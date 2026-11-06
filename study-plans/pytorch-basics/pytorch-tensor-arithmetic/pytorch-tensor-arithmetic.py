import torch

# add, multiply, matmul, power, or max
def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    if op == "add": # GOOD
        return torch.add(x, y)
    elif op == "multiply":
        return torch.multiply(x, y)
    elif op == "matmul": # GOOD
        return torch.matmul(x, y)
    elif op == "power": # GOOD
        return torch.pow(x, y)
    elif op == "max":
        return torch.maximum(x, y)