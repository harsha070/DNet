import copy
from easydict import EasyDict
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_loss import _loss


Flow = EasyDict({
    'n': 64,
    'h': 1.0,
    'w': 1.0,
    'lid_velocity': 1.0,
    'density': 1000.0,
    'viscosity': 0.0098,
    'gravity': 9.80
})
Flow.reynolds_number = Flow.density * Flow.lid_velocity * Flow.h / Flow.viscosity
Flow.delta_x = Flow.h / Flow.n
Flow.delta_y = Flow.w / Flow.n


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.dim = Flow.n * Flow.n * 32 // 16
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, input_simulation):
        
        encoded_vector = self.encoder(input_simulation)
        encoded_vector = encoded_vector.view(1, -1)
        encoded_vector = self.fc_layer(encoded_vector)
        encoded_vector = encoded_vector.view(-1, 32, Flow.n // 4, Flow.n // 4)
        output_simulation = self.decoder(encoded_vector)
        
        return output_simulation
    
def initialize_solver(flow):
    _initial = np.random.normal(size=(1, 3, flow.n, flow.n))
    # u-0, v-1, p-2
    for i in range(flow.n):
        _initial[0][0][i][0] = 0.0
        _initial[0][1][i][0] = 0.0
        _initial[0][0][i][flow.n-1] = 0.0
        _initial[0][1][i][flow.n-1] = 0.0
    for j in range(flow.n):
        _initial[0][0][0][j] = 0.0
        _initial[0][0][flow.n-1][j] = flow.lid_velocity
        _initial[0][1][0][j] = 0.0
        _initial[0][1][flow.n-1][j] = 0.0
    return _initial
    
    
def initial_solution(flow: EasyDict):
    init = torch.from_numpy(initialize_solver(flow))
    init = init.float()
    return init
    

model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
input_simulation = initial_solution(Flow)

def show(image):
    plt.imshow(image, vmin=np.amin(image), vmax=np.amax(image), cmap='hsv')
    plt.axis('off')
    plt.show()

for i in range(50):
    optimizer.zero_grad()
    input_simulation = model.forward(input_simulation)
    loss = _loss(Flow, input_simulation, input_simulation)
    loss.backward(retain_graph=True)
    optimizer.step()
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    logging.error("loss {}".format(loss.item()))
    _ = input_simulation.detach().numpy()
    if i % 5 == 4:
        show(_[0][1])
