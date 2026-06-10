import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Optional
from data import Tokenizer
from model import KVCache

@torch.no_grad()
def generate_text(
    model: nn.Module, 
    prompt: str, 
    tokenizer: Tokenizer, 
    max_new_tokens: int, 
    context_length: int, 
    temperature: float = 1.0, 
    top_k: Optional[int] = None, 
    device: str = "cpu"
) -> str:
    """Generates text using autoregressive prediction with optional Top-K sampling."""
    
    model.eval()
    
    # 1. Encode the prompt using the Tokenizer's vocab
    # Defaulting to 0 for unknown characters just as a fallback safety
    context = [tokenizer.vocab.get(c, 0) for c in prompt]
    x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, seq_len)

    # 2. Initialize KV Caches (One dedicated cache tracker per Transformer layer)
    # If your GPT model has N layers, we pass a list of N KVCache objects
    num_layers = len(model.blocks) if hasattr(model, 'blocks') else 2 # Defaulting to your spec
    # Pass context_length to KVCache so it can manage its size
    kv_caches = [KVCache(context_length=context_length) for _ in range(num_layers)]
    
    for step in range(max_new_tokens):
        # 3. Crop the sequence to the model's maximum context length
        x_cond = x[:, -context_length:]

        # Performance optimization: After the first complete forward pass,
        # we only need to pass the absolute newest token into the model.
        if step > 0:
            x_cond = x[:, -1:] # Shape: (1, 1)
        
        # 4. Forward pass to get logits
        logits = model(x_cond, kv_caches=kv_caches) 
        
        # 5. Pluck the logits at the final step and apply temperature
        # logits shape becomes (batch_size, vocab_size)
        logits = logits[:, -1, :] 
        
        if temperature > 0.0:
            logits = logits / temperature
            
        # 6. Apply Top-K filtering strategically
        if top_k is not None:
            # Find the value of the k-th largest logit
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            
            # v[:, [-1]] gets the smallest value in the top-k tensor. 
            # We set everything lower than this threshold to negative infinity.
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 7. Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # 8. Sample the next token from the probability distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 9. Append the generated token to the running sequence
        x = torch.cat((x, next_token), dim=1)
        
    # 10. Decode the final tensor back into a string using inverse_vocab
    generated_ids = x[0].tolist()
    generated_text = "".join([tokenizer.inverse_vocab.get(idx, "") for idx in generated_ids])
    
    return generated_text
