import torch
import torch.nn.functional as F
import math

def get_lr(step: int, max_steps: int, learning_rate: float, warmup_steps: int):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * 0.1 + learning_rate * 0.9 * coeff

def train_step(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, max_norm: float = 1.0) -> float:
    """A single training step, correctly handling both CPU and CUDA."""
    # FIX: determine device_type from the actual input tensor, not hardcoded 'cuda'
    device_type = X.device.type
    use_amp = device_type == 'cuda'

    optimizer.zero_grad(set_to_none=True)
    
    # FIX: autocast and scaler are only active on CUDA.
    # On CPU, autocast(device_type='cuda') was a no-op and GradScaler was unreliable.
    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
    
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    
    return loss.item()
