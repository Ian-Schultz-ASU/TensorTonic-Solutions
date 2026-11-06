import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    if method == "zeros":
        return torch.zeros(shape)
    if method == "ones":
        return torch.ones(shape)
    if method == "full":
        return torch.full(shape, value, dtype=torch.float32)
        # size=shape, )