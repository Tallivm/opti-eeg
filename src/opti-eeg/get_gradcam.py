print('Loading imports...')

import os, argparse, logging
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from flatten_dict import unflatten

from models.eegnet_modular import EEGNet_Modular
from scripts import data_utils, train_utils
from scripts.utils import setup_logger, set_all_seeds


import warnings
warnings.filterwarnings("ignore", message="Full backward hook is firing when gradients are computed")

logger = logging.getLogger(__name__)


class EEGClassificationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor):
        output = self.model(input_tensor)
        predicted_classes = torch.argmax(output, dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for idx, pred_class in enumerate(predicted_classes):
            one_hot[idx, pred_class] = 1.0
        loss = (output * one_hot).sum()
        loss.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=[0, 3])
        weighted_activations = weights.unsqueeze(0).unsqueeze(-1) * self.activations
        cam = torch.sum(weighted_activations, dim=1)
        cam = torch.maximum(cam, torch.zeros_like(cam))
    
        for i in range(cam.shape[0]):
            cam_min, cam_max = cam[i].min(), cam[i].max()
            if cam_max > cam_min:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy()


def load_checkpoint(checkpoint_path: str, device) -> tuple[torch.nn.Module, dict]:
    logger.info(f'Loading model from checkpoint: {checkpoint_path}')    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['hyper_parameters']
    model = EEGNet_Modular(**config).to(device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError:
        logger.warning('Layer names are different - this may be an older version of the model. Trying to match the layers...')

        old_sd = checkpoint['model_state_dict']
        new_sd = model.state_dict()
        if len(old_sd) != len(new_sd):
            raise RuntimeError(
                f"Parameter count mismatch: checkpoint has {len(old_sd)}, "
                f"model expects {len(new_sd)}. This model was probably not trained with opti-eeg, otherwise please contact the author."
            )
        remapped = {new_key: old_val for (new_key, _), (_, old_val) in zip(new_sd.items(), old_sd.items())}
        model.load_state_dict(remapped, strict=True)
        logger.info('Successfully loaded checkpoint with remapped layer names.')

    return model, config


def generate_and_save_mean_gradcam_plot(gradcams: np.ndarray, config: dict, savename: str) -> None:
    ch_names = [ch for ch in config['data']['channel_names'] if ch not in config['data']['omit_channels']]
    assert len(ch_names) == gradcams.shape[1],  f'Error: produced gradCAM matrix has incorrect shapes, with {gradcams.shape[1]} channels instead of {len(ch_names)}.'

    mean_gradcam = gradcams.mean(axis=0)

    plt.figure(figsize=(16, 8))
    plt.imshow(mean_gradcam, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Gradient-weighted Class Activation')
    plt.yticks(range(mean_gradcam.shape[0]), ch_names, fontsize=8)
    n_seconds = mean_gradcam.shape[1] // config['data']['sample_rate']
    plt.xticks(range(0, mean_gradcam.shape[1]+1, config['data']['sample_rate']), np.linspace(0, n_seconds, n_seconds+1))
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(savename)


def main(checkpoint_path: str, gradcam_path: str, save_image: bool, device) -> None:
    model, model_config = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    model_config = unflatten(model_config, splitter='dot')

    set_all_seeds(model_config['train']['model_seed'])

    # Load data
    data_path = model_config["path"]["test_data_path"]
    labels_path = model_config["path"]["test_labels_path"]
    if not data_path:
        data_path = model_config["path"]["data_path"]
        labels_path = model_config["path"]["labels_path"]
    assert os.path.exists(data_path), f'Error: file "{data_path}" (data) does not exist.'
    assert os.path.exists(labels_path), f'Error: file "{labels_path}" (labels) does not exist.'
    logger.info(f'Data will be loaded from: {data_path} (labels: {labels_path})')

    _, _, test_data, test_labels = data_utils.load_and_validate_data(config=model_config)
    logger.info(f'Loaded data shape: {test_data.shape}')
    logger.info(f'Loaded labels shape: {test_labels.shape}')        

    test_dataloader = train_utils.create_dataloader(data=test_data, labels=test_labels,
                                                   batch_size=model_config["params"]["batch_size"], num_workers=0)

    # Run gradCAM extraction
    all_cams = []
    gradcam_runner = EEGClassificationGradCAM(model=model, target_layer=model.layers['temporal_conv'])

    logger.info('Producing gradCAMs...')
    for input_data, _ in test_dataloader:

        input_data = input_data.to(device)
        cam = gradcam_runner.generate_cam(input_data)
        all_cams.append(cam)
    
    if len(all_cams) > 1:
        all_cams = np.concatenate(all_cams, axis=0)
    else:
        all_cams = np.array(all_cams)

    logger.info(f'Produced {len(all_cams)} GradCAM matrices of shape {all_cams.shape[1:]}')
    with open(gradcam_path, 'wb') as f:
        np.save(f, all_cams)

    if save_image:
        plot_filename = gradcam_path.rsplit('.', 1)[0] + '.pdf'
        generate_and_save_mean_gradcam_plot(gradcams=all_cams, config=model_config, savename=plot_filename)

    logger.info('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Generation of GradCAM matrices for EEGNet models',
        description='Select a pretrained EEGNet model to generate gradCAM matrices and their visualizations.',
    )
    parser.add_argument('checkpoint', help='Path to model checkpoint.')
    parser.add_argument('-o', '--outdir', default='gradcam', help='Directory to save output.')
    parser.add_argument('-p', '--plot', action='store_true', help='Also generate a GradCAM plot (in a PDF file).')
    parser.add_argument('--loglevel', choices=['debug', 'info', 'warn'], default='info', help='Logging level during code execution: "debug", "info" (default), or "warn".')
    args = parser.parse_args()
    setup_logger(logger=logger, loglevel=args.loglevel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device is: {device}')

    assert os.path.exists(args.checkpoint), f'Error: checkpoint "{args.checkpoint}" does not exist.'
    assert args.checkpoint.endswith(".ckpt"), f'Error: checkpoint filename should end with ".ckpt".'

    os.makedirs(args.outdir, exist_ok=True)
    gradcam_path = os.path.join(args.outdir, Path(args.checkpoint).stem + '.npy')

    main(checkpoint_path=args.checkpoint, gradcam_path=gradcam_path, save_image=args.plot, device=device)
