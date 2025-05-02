import torch

def save(model, path = 'model_weights.pth'):
    torch.save(model.state_dict(), path)
    print(f"saving model to {path}")

# Load method throws error
def load(path = 'model_weights.pth'):
    model = torch.load(path, weights_only=False)
    model.eval()
    print(f"loading model from {path}")
    return model