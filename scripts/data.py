import torch
from torch.utils.data import Dataset
from torchtyping import TensorType
from typing import Dict, List, Tuple

def create_batches(data: TensorType[int], batch_size: int, context_length: int) -> Tuple[TensorType[int], TensorType[int]]:
    """Samples random chunks from the dataset to form a batch."""
    # Generate random starting indices for each batch
    ix = torch.randint(0, len(data) - context_length, (batch_size,))
    # Create input (x) and target (y) tensors by slicing the data
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+1+context_length] for i in ix])
    return x, y

class Tokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [self.vocab[c] for c in text]

    def decode(self, ids: TensorType[int]) -> str:
        # Handles both PyTorch tensors and standard python lists
        return ''.join([self.inverse_vocab[i.item() if isinstance(i, torch.Tensor) else i] for i in ids])
