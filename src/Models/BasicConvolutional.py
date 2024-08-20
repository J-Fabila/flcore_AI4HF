import torch
import torch.nn as nn
import torch.nn.functional as F

"""class BasicConvolutional(torch.nn.Module):
    def __init__(self, params):
        super(BasicConvolutional, self).__init__()
        self.conv1 = nn.Conv2d(params.n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicConvolutional(nn.Module):
    def __init__(self, params):
        super(BasicConvolutional, self).__init__()
        self.conv1 = nn.Conv2d(params.n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(" ENTRADA ", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print("PRIMER POOL",x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print("SEGUNDO POOL",x.shape)
        # Calcular el tamaño del tensor para la capa completamente conectada
        x_size = x.size(0)
        print(" SIZE",x_size)
        x = x.view(x_size, -1)
        print(" DESPUES DEL VIEW",x.shape)
        # Ajustar el tamaño del tensor para la capa completamente conectada
        x = F.relu(self.fc1(x))
        print(" SEGUDA RELU",x.shape)
        x = F.relu(self.fc2(x))
        print(" FINAL")
        x = self.fc3(x)
        return x"""

class BasicConvolutional(nn.Module):
    def __init__(self, params):
        super(BasicConvolutional, self).__init__()
        self.conv1 = nn.Conv2d(params.n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)  # 61 = (256 - 5 + 1) / 2
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Calcular el tamaño del tensor para la capa completamente conectada
        x_size = x.size(0)
        x = x.view(x_size, -1)
        
        # Ajustar el tamaño del tensor para la capa completamente conectada
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
