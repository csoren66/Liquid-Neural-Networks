import torch
import torch.nn as nn

class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size):
        super(ESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.W_in = nn.Linear(input_size, reservoir_size)
        self.W_res = nn.Linear(reservoir_size, reservoir_size)
        self.W_out = nn.Linear(reservoir_size, output_size)

    def forward(self, input):
        reservoir = torch.zeros((input.size(0), self.reservoir_size))
        for i in range(input.size(1)):
            input_t = input[:, i, :]
            reservoir = torch.tanh(self.W_in(input_t) + self.W_res(reservoir))
        output = self.W_out(reservoir)
        return output

# Example usage
input_size = 10
reservoir_size = 100
output_size = 1

model = ESN(input_size, reservoir_size, output_size)