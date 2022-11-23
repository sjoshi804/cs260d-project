import argparse

import numpy.random as npr
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from tqdm import tqdm

from datasets import get_dataset
from models import get_model

# Arguments
parser = argparse.ArgumentParser(description="Prune model using Iterative Magnitude Pruning")
parser.add_argument("--model", default="resnet18", type=str, help="model to prune")
parser.add_argument("--dataset", default="cifar10", type=str, choices=["mnist", "cifar10"])
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch-size", default=256, type=int)
parser.add_argument("--lr", default=0.1)
parser.add_argument("--prune-iterations", default=5, type=int)
parser.add_argument("--prune-pct", default=0.9, type=float)
parser.add_argument("--rewind-epoch", default=5, type=int)
parser.add_argument("--seed", default=0)
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
npr.seed(args.seed)

# Get Dataset
trainset, testset = get_dataset(args.dataset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size)

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

# Model
model = get_model(args.model).to(device)

# Optimizer (hparams replicated from forgetting score paper)
optimizer = torch.optim.SGD(
    params=model.parameters(), 
    lr=args.lr,
    momentum=0.9,
    nesterov=True 
)

# LR Scheduler (replicated from forgetting score paper)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

# Loss Function 
criterion = F.cross_entropy

# Train for 1 Epoch
def train(model):
    model.train()
    train_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** | Train Acc: ****% ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    correct, total = 0, 0
    for batch_idx, (X, y) in t:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = logits.max(1)
        correct += (predicted.eq(y).sum().item() * 100)
        total += y.shape[0]
        t.set_description('Loss: %.3f | Train Acc: %.3f%% ' % (train_loss / (batch_idx + 1), correct / total))
    return train_loss / len(trainloader), correct / total

# Test for 1 Epoch
def test(model):
    model.eval()
    test_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** | Train Acc: ****% ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    correct = 0
    total = 0
    for batch_idx, (X, y) in t:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        test_loss += loss.item()
        _, predicted = logits.max(1)
        correct += (predicted.eq(y).sum().item() * 100)
        total += y.shape[0]
        t.set_description('Loss: %.3f | Test Acc: %.3f%% ' % (test_loss / (batch_idx + 1), correct / total))
    return test_loss / len(trainloader), correct / total


# Pruning, Rewinding and Retraining
for epoch in range(10):
    print(f"step: {epoch}")
    train_loss, train_acc = train(model)
    print(f"train_loss: {train_loss}")
    print(f"train_acc: {train_acc}")
    test_loss, test_acc = test(model)
    print(f"test_loss: {test_loss}")
    print(f"test_acc: {test_acc}")


parameters_to_prune = []
for module in model.named_modules():
    if isinstance(module, torch.nn.Sequential) or isinstance(module, torch.nn.BatchNorm2d):
        continue
    parameters_to_prune.append((module, 'weight'))


prune.global_unstructured(
    parameters=parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    prune=0.2
)

for epoch in range(10):
    print(f"step: {epoch}")
    train_loss, train_acc = train(model)
    print(f"train_loss: {train_loss}")
    print(f"train_acc: {train_acc}")
    test_loss, test_acc = test(model)
    print(f"test_loss: {test_loss}")
    print(f"test_acc: {test_acc}")

