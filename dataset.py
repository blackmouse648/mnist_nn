from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307,std=0.3081)
])

train_set = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_set = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)


