import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Grayscale
from pytorchfi.core import fault_injection

## Download the Dataset: Just for Test
torch.hub.set_dir("./")
train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([ToTensor()])
    )
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([ToTensor()])
    )
## Forming dataloader and minibatch size: we will re-use minibatch_size in Injection Model later
minibatch_size = 64
dataloader4Train = DataLoader(train_data, batch_size=minibatch_size)
dataloader4Test = DataLoader(test_data, batch_size=minibatch_size)

## GPGPU Accelaration
gpgpuAcc = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"
## DNN Model Definition, we will use EfficientNet V2
dnnModel = torchvision.models.efficientnet_v2_s(weights=None)

## The Train, Test Function and loss function definition
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            XX, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("------------------------------------------------------------------------------\n")
loss_fn = torch.nn.CrossEntropyLoss()
if os.path.exists("./model.pth"):
    dnnModel.load_state_dict(torch.load("model.pth"))
else:
    train(dataloader4Train, dnnModel, loss_fn, torch.optim.SGD(dnnModel.parameters(), lr=1e-3))
    torch.save(dnnModel.state_dict(), "model.pth")

## First Test without no perturbation
test(dataloader4Test, dnnModel, loss_fn)

## Now, Let's do the perturbation
## TODO :
## * Implement a single bit flip situation
## *
## * You must implement custom injection function below
## -------------------- Write Custom injection function here ------------------
class Custom_Injection(fault_injection):
    def __init__(self, model, batch_size, input_shape, layer_type, **ExtraArgs):
        super().__init__(model, batch_size, input_shape, layer_type, **ExtraArgs)
        # you can write some data structures here
    def inject_custom(self, module, input_data, output_data):
        # write something
        dummy_value = 1 # dummy code, preventing syntax error
## ----------------------------- Definition End -------------------------------
inj_model = Custom_Injection(dnnModel, minibatch_size, None, None, use_cuda=gpgpuAcc)
corrupted_model = inj_model.declare_neuron_fi(function=inj_model.inject_custom)
## Second Test after perturbation
test(dataloader4Test, corrupted_model, loss_fn)
print("The End")
