import argparse

# Arguments
parser = argparse.ArgumentParser(description="Prune model using Iterative Magnitude Pruning")
parser.add_argument("--model", default="resnet18", type=str, help="model to prune")
parser.add_argument("--dataset", default="mnist", type=str, choices=["mnist", "cifar10"])
parser.add_argument("--epochs", default=200, type=int)
