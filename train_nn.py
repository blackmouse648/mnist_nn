import torch

from model import NN
import dataset
from torch import nn
import torch.optim as optim
from matplotlib import pyplot as plt

epoch = 20


model = NN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

def train():
    for i,data in enumerate(dataset.train_loader,0):

        var,label = data

        out = model(var)

        loss = criterion(out,label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset.test_loader:

            var,label = data

            out = model(var)

            _, predict = torch.max(out.data,dim=1)

            total += label.size(0)
            print(total)
            correct += (predict == label).sum().item()

            print(100*correct/total)

if __name__ == '__main__':
    for i in range(epoch):
        train()
        test()

