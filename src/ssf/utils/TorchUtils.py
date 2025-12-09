import torch
import gc

def print_torch_memory(tag=""):
  allocated = torch.cuda.memory_allocated() / 1e9
  reserved = torch.cuda.memory_reserved() / 1e9
  print(f"[{tag}] PyTorch memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def clear_torch_memory(generation_strategy=None):
  del generation_strategy.model
  del generation_strategy
  gc.collect()
  torch.cuda.empty_cache()
  return