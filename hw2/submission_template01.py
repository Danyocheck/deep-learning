import torch.nn as nn

def create_model():
    # Linear layer mapping from 784 features, so it should be 784->256->16->10

    NN = nn.Sequential(nn.Linear(784, 256, bias=True),
                   nn.ReLU(),
                   nn.Linear(256, 16, bias=True),
                   nn.ReLU(),
                   nn.Linear(16, 10, bias=True),
                   nn.ReLU())

    # return model instance (None is just a placeholder)
    return NN
    

def count_parameters(model):
    # верните количество параметров модели model
    return sum(p.numel() for p in model.parameters())