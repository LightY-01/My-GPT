from typing import Dict, List, Tuple

def get_merges(corpus: str, num_merges: int) -> List[List[str]]:
    # Split corpus into a list of individual characters
    tokens = list(corpus)
    merges = []
    for _ in range(num_merges):
        # Count frequency of all adjacent token pairs
        freq = {}
        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])
            freq[pair] = freq.get(pair, 0) + 1
        # Find the most frequent pair (break ties lexicographically)
        best_value = max(freq.values())
        freq_pairs = []
        for p, c in freq.items():
            if c == best_value:
                freq_pairs.append(p)
        freq_pairs.sort()
        best = freq_pairs[0]
        merges.append([best[0], best[1]])

        # Merge all non-overlapping occurrences left to right
        new_tokens = []
        i = 0
        while i < (len(tokens)-1):
            if tokens[i] == best[0] and tokens[i+1] == best[1]:
                new_tokens.append(tokens[i]+tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_tokens.append(tokens[len(tokens)-1])
        tokens = new_tokens
        print(tokens)

    return merges

def tokenize_numbers(numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
    # Tokenize each number using greedy left-to-right longest match.
    tokens_list = []
    for num in numbers:
        str_num = str(num)
        tokens_list.append(greedy_tokenize(str_num, vocab))
    return tokens_list

def count_tokens(text: str, vocab: Dict[str, int]) -> int:
    # Count how many tokens the text uses with greedy tokenization.
    return len(greedy_tokenize(text, vocab))

def fertility_score(text: str, vocab: Dict[str, int]) -> float:
    # Compute tokens-per-word ratio (fertility).
    # Higher = more expensive and less efficient.
    return round(len(greedy_tokenize(text, vocab)) / len(text.split()), 4)

def greedy_tokenize(text: str, vocab: Dict[str, int]) -> List[str]:
    tokens = []
    i = 0
    # Fix start and iteratively reduce length of substring
    while i < len(text):
        best = None
        for length in range(len(text) - i, 0, -1):
            substr = text[i:i + length]
            if substr in vocab:
                best = substr
                break
        if best is None:
            tokens.append(text[i])
            i += 1
        else:
            tokens.append(best)
            i += len(best)
    return tokens