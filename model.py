from torch import nn
from torch.nn import functional as F

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.layer1 = nn.Linear(784,512)
        self.layer2 = nn.Linear(512,256)
        self.layer3 = nn.Linear(256,128)
        self.layer4 = nn.Linear(128,64)
        self.layer5 = nn.Linear(64,10)

    def forward(self,x):
        x = x.view(-1,784)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        return x


