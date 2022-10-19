import torch.nn as nn
import torch
import numpy as np
from src.models import *
from torch.nn import functional as F
import os

def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate, bn_track_running=True):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out, track_running_stats=bn_track_running)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out, track_running_stats=bn_track_running)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y




class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8, remove_relu=False, bn_track_running=True):
        super(ResNet1d, self).__init__()
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out, track_running_stats=bn_track_running)
        self.relu = nn.ReLU()
        self.remove_relu = remove_relu

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate, bn_track_running=bn_track_running)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

        self.l1_loss = nn.L1Loss(reduction="sum")

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)
        if not self.remove_relu:
            x = self.relu(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x

    def predict(self, x:torch.Tensor) -> torch.Tensor:
        return self.forward(x)

        
    def load_feature_extractor(self, state_dict_path:str):
        state_dict = torch.load(os.path.join(state_dict_path, 'model.pth'), map_location=lambda storage, loc: storage)
        # filter the state dict:
        feature_extractor_state_dict = {k:v for k, v in state_dict.items() if not "lin" in k}
        self.load_state_dict(feature_extractor_state_dict, strict=False)

    def create_ll_optimizer(self, optimizer_type=torch.optim.Adam, **optimizer_kwargs):
        self.optimizer = optimizer_type(list(self.lin.parameters()), **optimizer_kwargs)

class OrdinalResNet1d(ResNet1d):
    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8, remove_relu=False, bn_track_running=True):
        print(f"Since ordinal regression uses one less class, decreasing class from {n_classes} to {n_classes - 1} in order to be consistent with normal implementation")
        n_classes -= 1
        super().__init__(input_dim, blocks_dim, n_classes, kernel_size, dropout_rate, remove_relu, bn_track_running)
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes, bias=False)
        self.lin_bias = nn.Parameter(torch.zeros((1, n_classes)).float())

    def forward(self, x) -> torch.Tensor:
        return super().forward(x) + self.lin_bias

    def predict(self, x:torch.Tensor) -> torch.Tensor:
        prediction_logit = self.forward(x)
        return self.logits_to_prediction(prediction_logit)

    def prediction_class_to_regression(self, pred_class, buckets, mean=True):
        if mean:
            buckets = torch.cat((buckets[0].unsqueeze(dim=0), buckets.unfold(dimension=0, size=2, step=1).mean(dim=-1), buckets[-1].unsqueeze(dim=0)))
        return buckets[pred_class]

    def logits_to_prediction(self, logits:torch.Tensor) -> torch.Tensor:
        return (logits.sigmoid() > 0.5).sum(dim=-1).long()



class ResNet1dMetaData(ResNet1d):
    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8, remove_relu=False, bn_track_running=True, meta_features_dim=512):
        super().__init__(input_dim, blocks_dim, n_classes, kernel_size, dropout_rate, remove_relu, bn_track_running)
        self.meta_to_features = nn.Sequential(
                                nn.Linear(2, 64), nn.ReLU(),
                                nn.Linear(64, 128), nn.ReLU(),
                                nn.Linear(128, meta_features_dim)
                                )
        # reinit last layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last + meta_features_dim
        self.lin = nn.Identity()
        self.combined_lin = nn.Sequential(
                            nn.Linear(last_layer_dim, int(last_layer_dim/2)), nn.ReLU(),
                            nn.Linear(int(last_layer_dim/2), n_classes)
                            )
        
    def forward(self, x):
        traces, age, sex = x
        traces_features = super().forward(traces)
        meta_features = self.meta_to_features(torch.cat((age, sex), dim=-1))
        prediction = self.combined_lin(torch.cat((traces_features, meta_features), dim=-1))
        return prediction


class ProbResNet1d(ResNet1d):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, kernel_size=17, dropout_rate=0.8, bn_track_running=True, **kwargs):
        remove_relu = False
        n_classes = 1 # fixed for prob regression
        super(ProbResNet1d, self).__init__(input_dim, blocks_dim, n_classes, kernel_size, dropout_rate, remove_relu, bn_track_running)
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out, track_running_stats=bn_track_running)
        self.relu = nn.ReLU()

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate, bn_track_running=bn_track_running)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        self.n_blk = len(blocks_dim)
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, 1) # the linear layer for the mean
        self.lin_log_var_1 = nn.Linear(last_layer_dim, last_layer_dim)
        self.lin_log_var_2 = nn.Linear(last_layer_dim, 1)
        self.lin_relu = nn.ReLU()

    def create_linear_optimizer(self, optimizer_type=torch.optim.Adam, **optimizer_kwargs):
        self.optimizer = optimizer_type(list(self.lin.parameters()) + list(self.lin.parameters()) + list(self.lin_log_var_1.parameters()) + list(self.lin_log_var_2.parameters()), **optimizer_kwargs)

    def create_var_optimizer(self, optimizer_type=torch.optim.Adam, **optimizer_kwargs):
        self.optimizer = optimizer_type(list(self.lin_log_var_1.parameters()) + list(self.lin_log_var_2.parameters()), **optimizer_kwargs)


    def reformat_Laplace(self):
        self.forward = self.forward_mean

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        x = self.features(x)
        # Fully connected layer
        mean, log_var = self.features_to_mean(x), self.features_to_log_var(x)
        return torch.cat([mean, log_var], dim=-1)

    def features_to_mean(self, x):
        return self.lin(x)

    def features_to_log_var(self, x):
        return self.lin_log_var_2(self.relu(self.lin_log_var_1(x)))

    def forward_mean(self, x):
        x = self.features(x)
        return self.features_to_mean(x)

    def forward_log_var(self, x):
        x = self.features(x)
        return self.features_to_log_var(x)

    def features(self, x):
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)
        return x

