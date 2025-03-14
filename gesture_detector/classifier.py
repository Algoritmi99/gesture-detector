import numpy as np
from light import nn
from overrides import override


class FFNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        )

        self.in_dim = input_size
        self.out_dim = output_size

    @override
    def forward(self, arg: np.ndarray) -> np.ndarray:
        return self.net(arg)

    @override
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return self.net.backward(d_out)
