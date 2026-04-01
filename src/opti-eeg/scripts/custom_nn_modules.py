import torch
from torch.nn import functional as F
from torch.nn.utils.parametrize import register_parametrization


class PyTorchKNN:
    """GPU-Accelerated KNN. Claude Opus 4.5."""
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, X, y):
        # X: [n_samples, features], y: [n_samples]
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        # X: [n_query, features]
        # Compute pairwise distances
        dists = torch.cdist(X, self.X_train)  # [n_query, n_train]
        
        # Get k nearest neighbors
        _, indices = dists.topk(self.k, largest=False, dim=1)  # [n_query, k]
        
        # Get labels of neighbors
        neighbor_labels = self.y_train[indices]  # [n_query, k]
        
        # Majority vote
        preds = torch.mode(neighbor_labels, dim=1).values
        return preds


class SoftKNNLayer(torch.nn.Module):
    """Replace hard voting with soft, temperature-scaled distances. Claude Opus 4.5."""
    def __init__(self, k=5, temperature=1.0, num_classes=4):
        super().__init__()
        self.k = k
        self.temperature = torch.nn.Parameter(torch.tensor(temperature))
        self.num_classes = num_classes
        # These will be set from training data
        self.register_buffer('support_embeddings', None)
        self.register_buffer('support_labels', None)
    
    def set_support_set(self, embeddings, labels):
        """Call this with your training data before forward pass"""
        self.support_embeddings = embeddings
        self.support_labels = labels
    
    def forward(self, x):
        # x: [batch, features]
        # Compute distances to all support samples
        dists = torch.cdist(x, self.support_embeddings)  # [batch, n_support]
        
        # Get k nearest neighbors
        knn_dists, indices = dists.topk(self.k, largest=False, dim=1)
        
        # Soft weights (differentiable!)
        weights = torch.softmax(-knn_dists / self.temperature, dim=1)  # [batch, k]
        
        # Get one-hot labels of neighbors
        neighbor_labels = self.support_labels[indices]  # [batch, k]
        one_hot = F.one_hot(neighbor_labels, self.num_classes).float()  # [batch, k, num_classes]
        
        # Weighted vote (soft probabilities)
        probs = torch.einsum('bk,bkc->bc', weights, one_hot)  # [batch, num_classes]
        
        return probs  # Use with cross-entropy loss
    

class PrototypicalClassifier(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(temperature))
        self.register_buffer('prototypes', None)
    
    def compute_prototypes(self, embeddings, labels, num_classes):
        """Compute mean embedding per class"""
        prototypes = torch.zeros(num_classes, embeddings.size(1), device=embeddings.device)
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                prototypes[c] = embeddings[mask].mean(dim=0)
        self.prototypes = prototypes
    
    def forward(self, x):
        # Negative squared distance as logits
        dists = torch.cdist(x, self.prototypes)  # [batch, num_classes]
        logits = -dists / self.temperature
        return logits  # Use with cross-entropy
    

class MaxNormConstraint:
    # Clips the norm of weight vectors to be <= max_value. Claude Opus 4.5
    def __init__(self, max_value: float = 1.0, axis: int = 0):
        self.max_value = max_value
        self.axis = axis
    
    def __call__(self, module: torch.nn.Module, name: str = 'weight'):
        if hasattr(module, name):
            weight = getattr(module, name)
            with torch.no_grad():
                norms = torch.norm(weight, dim=self.axis, keepdim=True)
                desired = torch.clamp(norms, max=self.max_value)
                scale = desired / (norms + 1e-8)
                weight.mul_(scale)


class SeparableConv2d(torch.nn.Module):
    # adapted from https://stackoverflow.com/a/65155106
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], 
                 stride: int = 1, padding: int | str | tuple = 0, bias: bool = False):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size, 
            groups=in_channels,
            bias=bias,
            padding=padding,
            stride=stride
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1), 
            bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class LinearWithConstraint(torch.nn.Linear):
    """Adapted from braindecode."""
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))


class MaxNormParametrize(torch.nn.Module):
    """Adapted from braindecode."""
    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        self.max_norm = max_norm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.renorm(p=2, dim=0, maxnorm=self.max_norm)
    

class Conv2dWithConstraint(torch.nn.Conv2d):
    """Adapted from braindecode."""
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        # initialize the weights
        torch.nn.init.xavier_uniform_(self.weight, gain=1)
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))