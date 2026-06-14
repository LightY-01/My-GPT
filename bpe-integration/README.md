# Sub-Word Tokenization (BPE) Integration

This directory contains the pipeline upgrades required to transition the core GPT architecture from a character-level tokenizer to an industry-standard Byte-Pair Encoding (BPE) tokenizer. 

To achieve this, the custom vocabulary builder was replaced with OpenAI's `tiktoken` library (specifically the `gpt2` encoding). This shifts the model's objective from learning how to spell words letter-by-letter to learning how to arrange whole sub-word units (syllables and common words) based on context.  
> For a transparent look at how BPE works under the hood, a complete implementation of the token-merging mechanics can be found in [`scripts/custom_bpe.py`](./scripts/custom_bpe.py).

## 🧠 Parameter Scaling & Hardware

Transitioning to BPE drastically changes the mathematical footprint of the network. The vocabulary size balloons from 65 characters to 50,257 sub-words. This inflates the embedding layer and the final output projection layer, increasing the model's overall size by over 50 million parameters without adding any additional hidden layers.

Because of this massive increase in VRAM requirements, this version of the model was trained on a **Google Colab T4 GPU** rather than the local RTX 3050.

| Parameter | Configuration |
| :--- | :--- |
| **Total Parameters** | 63,149,137 |
| **Vocabulary Size** | 50,257 (tiktoken 'gpt2') |
| **Context Length** | 256 |
| **Hardware Used** | NVIDIA T4 GPU |

(Note: The core Transformer blocks, Grouped Query Attention, and KV-Cache logic in `model.py` remain entirely identical to the character-level build).*

## 📊 Results & Mathematical Analysis

* **Best Validation Loss:** 4.75

At first glance, a validation loss of 4.75 appears significantly worse than the 1.52 achieved by the character-level model. However, this is a mathematical inevitability based on the dataset size, not a bug.

**The Vocabulary Mismatch:** The TinyShakespeare dataset contains roughly 300,000 words. OpenAI's `tiktoken` vocabulary has 50,257 unique sub-word tokens. Because the vocabulary is massive and the dataset is tiny, the model sees most sub-word tokens infrequently. It lacks the volume of data necessary to generalize efficiently, leading to rapid overfitting and a higher calculated loss.

**The Output Paradox:** Despite the higher mathematical loss, the generated text is drastically more coherent. Because the model no longer has to learn character composition, it defaults to outputting perfectly spelled, valid English words. When it hallucinates, it hallucinates complete thoughts rather than spiraling into character-level gibberish.

## Sample Generation

The outputs demonstrate the immediate structural benefit of sub-word tokenization. The model produces valid English vocabulary and maintains formatting seamlessly, even when the global narrative drifts. 

**Prompt:** `To be or to be not, that is the question:`
> **Generated output:**
> To be or to be not, that is the question:  
> The heavens have made you were you  
> But not the people.  
>
> CAMILLO:  
> What news is a guest to do, but I'll be.  
>
> PERDITA little:  
> I'll be in all,  
> You shall have been too; for you will not, you  
> But, you have done to your own.  

**Prompt:** `JESUS:`
> **Generated output:**
> JESUS:
> I have had a foot to the market-day?  
> 
> ESCALUS:  
> Ay, sir, sir: I know you  
> not too, if they have been a bawd your  
> continency and detest: you were  
> do't, they are now going;  
