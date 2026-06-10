"""
infer.py — Load a saved GPT checkpoint and generate text interactively.
Run this after training is complete

Usage:
    python infer.py
    python infer.py --checkpoint my_checkpoint.pth --temperature 0.9 --tokens 200
"""

import torch
import argparse
from data import Tokenizer
from model import GPT
from generate import generate_text

def load_model(checkpoint_path: str, device: str):
    """Reconstruct the model from a checkpoint and return model + tokenizer."""
    print(f"Loading checkpoint: {checkpoint_path}")
    # map_location lets you load a GPU-trained model onto CPU and vice versa
    ckpt = torch.load(checkpoint_path, map_location=device)

    hp        = ckpt['hyperparams']
    vocab     = ckpt['vocab']
    tokenizer = Tokenizer(vocab)

    model = GPT(
        vocab_size     = hp['vocab_size'],
        model_dim      = hp['embed_dim'],
        num_heads      = hp['num_heads'],
        num_kv_heads   = hp['num_kv_heads'],
        context_length = hp['context_length'],
        num_blocks     = hp['num_blocks'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()   # turn off dropout for inference

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded  |  Parameters: {total_params:,}  |  Device: {device}")
    return model, tokenizer, hp['context_length']


def main():
    parser = argparse.ArgumentParser(description="GPT inference — load checkpoint and generate text")
    parser.add_argument('--checkpoint',   type=str,   default='gpt_checkpoint.pth')
    parser.add_argument('--temperature',  type=float, default=0.8,
                        help='Higher = more creative/random, lower = more focused')
    parser.add_argument('--tokens',       type=int,   default=150,
                        help='Number of new tokens to generate')
    parser.add_argument('--top_k',        type=int,   default=10,
                        help='Number of top tokens to consider')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, context_length = load_model(args.checkpoint, device)

    print("\n" + "="*55)
    print("  GPT Interactive Prompt  (type 'quit' to exit)")
    print("="*55)

    while True:
        prompt = input("\nPrompt > ").strip()
        if prompt.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if not prompt:
            continue

        output = generate_text(
            model          = model,
            prompt         = prompt,
            tokenizer      = tokenizer,
            max_new_tokens = args.tokens,
            context_length = context_length,
            top_k          = args.top_k,
            temperature    = args.temperature,
            device         = device,
        )
        print("\n--- Generated ---")
        print(output)
        print("-" * 40)


if __name__ == "__main__":
    main()
