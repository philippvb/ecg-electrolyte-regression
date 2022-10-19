from enum import Enum
from src.models.trainer import * 
from src.models.resnet import * 


def get_trainer(task:str) -> Trainer:
    if task == "classification":
        return ClassificationTrainer
    elif task == "binary":
        return BinaryClassificationTrainer
    elif task == "ordinal":
        return OrdinalRegressionTrainer
    elif task == "regression":
        return RegressionTrainer
    elif task == "gaussian":
        return ProbRegressionTrainer
    else:
        raise ValueError(f"Task {task} not implemented")

def get_model(task:str, model:str, metadata=False):
    if model == "resnet":
        if task in ["classification", "regression", "binary"]:
            if metadata:
                return ResNet1dMetaData
            return ResNet1d
        elif task == "ordinal":
            if metadata:
                raise NotImplementedError
            return OrdinalResNet1d
        elif task == "gaussian":
            if metadata:
                raise NotImplementedError
            return ProbResNet1d
        else:
            raise ValueError(f"ResNet for task {task} not implemented")
    else:
        raise ValueError(f"Model {model} not found.")
