import logging
from torch import nn, no_grad, cdist, zeros_like
from torch import zeros as torch_zeros
from torch.nn.functional import relu

from scripts.train_utils import initialize_xavier_uniform_weight_zero_bias
from scripts.regularizers import TSG
from scripts import model_pieces


logger = logging.getLogger(__name__)


class EEGNet_Modular(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.F1 = kwargs['params.F1']
        self.F2 = kwargs['params.F2']
        self.D = kwargs['params.D']

        self.sample_rate = kwargs['data.sample_rate']
        self.n_channels = kwargs['data.n_channels']
        self.n_classes = kwargs['data.n_classes']

        self.model_name = kwargs['params.model_name']
        self.kwargs = kwargs
        self.layers = nn.ModuleDict()
        self.regularization_loss = None
        self.final_dim_size = (kwargs['data.n_timestamps'] //
                               (kwargs['params.depthwise_pool_width'] * kwargs['params.separable_pool_width']))


        self.build_model_by_name()

        self.initialize_weights()  # TODO: add more initializations?

    def initialize_weights(self):
        initialize_xavier_uniform_weight_zero_bias(self)

    def build_model_by_name(self) -> None:
        if self.model_name == "EEGNet":
            self.build_standard_eegnet()
        elif self.model_name == "EEGNetConv":
            self.build_conv_eegnet()
        elif self.model_name == "EEGNetProto":
            self.build_eegnet_proto()
        elif self.model_name == "TSGLEEGNet":
            self.build_tsgl_eegnet()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def _build_model_base(self) -> None:
        self.layers.update(model_pieces.EEGNet_temporal_conv_with_batchnorm(
            F1=self.F1,
            kernel_length=self.kwargs['params.temporal_kernel_length'],
            momentum=self.kwargs['params.temporal_momentum'],
            affine=self.kwargs['params.temporal_affine'],
            eps=self.kwargs['params.temporal_eps'],
        ))
        self.layers.update(model_pieces.EEGNet_depthwise_conv_with_batchnorm(
            F1=self.F1, D=self.D, n_channels=self.n_channels,
            momentum=self.kwargs['params.depthwise_momentum'],
            affine=self.kwargs['params.depthwise_affine'],
            eps=self.kwargs['params.depthwise_eps'],
            max_norm=self.kwargs['params.conv_depth_max_norm']
        ))
        self.layers.update(model_pieces.activation_pool_dropout(
            activation_name=self.kwargs['params.depthwise_activation_name'],
            pool_name=self.kwargs['params.depthwise_pool_name'],
            dropout_name=self.kwargs['params.depthwise_dropout_name'],
            pool_kernel_width=self.kwargs['params.depthwise_pool_width'],
            dropout_rate=self.kwargs['params.depthwise_dropout_rate'],
            prefix='depthwise'
        ))
    
    def _extend_with_separable_conv(self) -> None:
        self.layers.update(model_pieces.EEGNet_separable_conv_with_batchnorm(
            F1=self.F1, F2=self.F2, D=self.D,
            kernel_length=self.kwargs['params.separable_kernel_length'],
            momentum=self.kwargs['params.separable_momentum'],
            affine=self.kwargs['params.separable_affine'],
            eps=self.kwargs['params.separable_eps'],
        ))
        self.layers.update(model_pieces.activation_pool_dropout(
            activation_name=self.kwargs['params.separable_activation_name'],
            pool_name=self.kwargs['params.separable_pool_name'],
            dropout_name=self.kwargs['params.separable_dropout_name'],
            pool_kernel_width=self.kwargs['params.separable_pool_width'],
            dropout_rate=self.kwargs['params.separable_dropout_rate'],
            prefix='separable'
        ))

    def _extend_with_simple_conv(self) -> None:
        self.layers.update(model_pieces.EEGNet_TSGL_simple_cov_with_batchnorm(
            F1=self.F1, F2=self.F2, D=self.D,
            kernel_length=self.kwargs['params.separable_kernel_length'],
            momentum=self.kwargs['params.separable_momentum'],
            affine=self.kwargs['params.separable_affine'],
            eps=self.kwargs['params.separable_eps'],
        ))
        self.layers.update(model_pieces.activation_pool_dropout(  # TODO: not actually separable
            activation_name=self.kwargs['params.separable_activation_name'],
            pool_name=self.kwargs['params.separable_pool_name'],
            dropout_name=self.kwargs['params.separable_dropout_name'],
            pool_kernel_width=self.kwargs['params.separable_pool_width'],
            dropout_rate=self.kwargs['params.separable_dropout_rate'],
            prefix='tsgl_conv'
        ))

    def _extend_with_simple_classifier(self) -> None:
        self.layers.update(model_pieces.EEGNet_simple_classifier(
            F2=self.F2, n_classes=self.n_classes,
            final_dim_size=self.final_dim_size,
            max_norm=self.kwargs['params.linear_max_norm']
        ))

    def _extend_with_conv_classifier(self) -> None:
        self.layers.update(model_pieces.EEGNet_conv_classifier(
            F2=self.F2, n_classes=self.n_classes,
            final_dim_size=self.final_dim_size,
        ))

    def build_standard_eegnet(self) -> None:
        self._build_model_base()
        self._extend_with_separable_conv()
        self._extend_with_simple_classifier()

    def build_conv_eegnet(self) -> None:
        self._build_model_base()
        self._extend_with_separable_conv()
        self._extend_with_conv_classifier()

    def build_eegnet_proto(self) -> None:
        self._build_model_base()
        self._extend_with_separable_conv()
        self.layers.append(nn.Flatten())
        self.register_buffer('prototypes', torch_zeros(self.n_classes, self.F2 * self.final_dim_size))
        self.forward = self.proto_forward

    def build_tsgl_eegnet(self) -> None:
        self._build_model_base()
        self._extend_with_simple_conv()
        last_conv_layer = self.layers['tsgl_simple_conv']
        self._extend_with_simple_classifier()
        self._kernel_reg = TSG(l1=self.kwargs["params.l1"], l21=self.kwargs["params.l21"])
        self._activity_reg = TSG(tl1=self.kwargs["params.tl1"])
        self.regularization_loss = lambda x: self._kernel_reg(last_conv_layer.weight) + self._activity_reg(x)        

    def run_eegnet(self, x, n_layers: int | None = None):
        x = x.unsqueeze(1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Data unsqueezed: {x.shape}')
        for i, (layer_name, layer) in enumerate(self.layers.items()):
            if n_layers and i >= n_layers:
                break
            x = layer(x)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'After layer "{layer_name}": {x.shape}')
        if logger.isEnabledFor(logging.DEBUG):
            input('Press any key to continue...')
        return x

    def compute_prototypes(self, X, y):
        self.eval()
        with no_grad():
            embeddings = self.run_eegnet(X)
            for c in range(self.n_classes):
                mask = y == c
                if mask.sum() > 0:
                    self.prototypes[c] = embeddings[mask].mean(dim=0)
        self.train()

    def proto_forward(self, x):
        embeddings = self.run_eegnet(x)
        dists = cdist(embeddings, self.prototypes)
        return -dists / self.kwargs['params.proto_temperature']

    def forward(self, x):
        x = self.run_eegnet(x)
        return x


# class GradCAM:
#     def __init__(self, model, target_layer):
#         """Claude Opus 4.5"""
#         self.model = model
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks on the conv layer
#         target_layer.register_forward_hook(self.save_activation)
#         target_layer.register_full_backward_hook(self.save_gradient)
    
#     def save_activation(self, module, input, output):
#         self.activations = output
    
#     def save_gradient(self, module, grad_in, grad_out):
#         self.gradients = grad_out[0]
    
#     def __call__(self, x, class_idx: int | None = None, normalize: bool = False):
#         self.model.eval()
#         x = x.clone()
#         x.requires_grad_(True)
#         output = self.model(x)
        
#         if class_idx is None:
#             class_idx = output.argmax(dim=1)
        
#         self.model.zero_grad()
#         one_hot = zeros_like(output)
#         one_hot[0, class_idx] = 1
#         output.backward(gradient=one_hot)
        
#         # Compute weights and Grad-CAM
#         weights = self.gradients.mean(dim=(2, 3), keepdim=True)
#         gradcam = (weights * self.activations).sum(dim=1, keepdim=True)
#         gradcam = relu(gradcam)
        
#         if normalize:
#             gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
#         return gradcam