
import torch


def tensor_size_in_bit(tensor):
    n_element = torch.numel(tensor)
    dtype = tensor.dtype
    if dtype == torch.uint8:
        b = 8
    elif dtype == torch.float32:
        b = 32
    elif dtype == torch.float64:
        b = 64
    else:
        raise Exception(f"unsupported dtype {dtype}")
    return n_element * b