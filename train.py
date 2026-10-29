import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
transform =transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])
train_ds=datasets.MNIST("./data",train=True,download=True,
                        transform=transform)
test_ds=datasets.MNIST("./data",train=False,download=True,
                       transform=transform)
train_loader = DataLoader(train_ds, batch_size=64,shuffle=True)
test_loader=DataLoader(test_ds,batch_size=1000)
model=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10),
)
loss_fn=nn.CrossEntropyLoss()
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
for epoch in range(3):
    model.train()
    for x,y in train_loader:
        pred=model(x)
        loss=loss_fn(pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    correct=0
    with torch.no_grad():
        for x,y in test_loader:
            pred=model(x).argmax(dim=1)
            correct+=(pred==y).sum().item()
        print(f"epoch{epoch+1},acc={correct/len(test_ds):.4f}")
