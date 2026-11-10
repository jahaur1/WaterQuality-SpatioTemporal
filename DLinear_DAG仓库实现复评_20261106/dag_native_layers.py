import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(configs.input_dim, configs.mlp_hidden_dims)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(configs.mlp_hidden_dims, configs.output_dim)

    def forward(self, x_dec, output):
        x_dec = x_dec.float()
        output = output.float()
        mlp_input = torch.cat((output, x_dec), dim=-1)
        mlp_input = self.linear1(mlp_input)
        mlp_input = self.relu(mlp_input)
        mlp_input = self.linear2(mlp_input)
        return mlp_input
