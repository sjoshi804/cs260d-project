from torchvision import models

def get_model(model: str):
    if model == "resnet18":
        return models.resnet18()
    else:
        raise NotImplementedError(f"{model} not supported yet.")