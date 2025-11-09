import random
from repeated_token import check_repeated_token
from itertools import product, combinations_with_replacement

def generate_no_repeat_sequences(vocab=['a', 'b', 'c', 'd', 'e'], max_len=5):
    """Generate sequences with no repeated tokens."""
    sequences = []
    # Generate all permutations of different lengths
    for length in range(1, min(len(vocab) + 1, max_len + 1)):
        from itertools import permutations
        for perm in permutations(vocab, length):
            sequences.append(''.join(perm))
    return sequences

def generate_with_repeats_sequences(n, vocab=['a', 'b', 'c', 'd', 'e'], min_len=2, max_len=6):
    """Generate sequences that definitely have repeated tokens."""
    sequences = []
    for _ in range(n):
        length = random.randint(min_len, max_len)
        # Ensure at least one repeat by picking a token to repeat
        seq = []
        repeat_token = random.choice(vocab)
        num_repeats = random.randint(2, min(length, 4))

        # Add the repeated token
        for _ in range(num_repeats):
            seq.append(repeat_token)

        # Fill the rest with random tokens (might create more repeats)
        while len(seq) < length:
            seq.append(random.choice(vocab))

        # Shuffle to mix up positions
        random.shuffle(seq)
        sequences.append(''.join(seq))
    return sequences

def generate_all_same_sequences(vocab=['a', 'b', 'c', 'd', 'e'], max_len=6):
    """Generate sequences where all tokens are the same."""
    sequences = []
    for token in vocab:
        for length in range(2, max_len + 1):
            sequences.append(token * length)
    return sequences

def generate_all_repeated_token_testcases():
    """Generate test cases for the repeated token detection algorithm."""
    all_cases = []
    bos = "BOS"
    model = check_repeated_token()

    vocab = ['a', 'b', 'c', 'd', 'e']

    # 1. Sequences with no repeated tokens (all unique)
    no_repeat_seqs = generate_no_repeat_sequences(vocab, max_len=5)
    for seq in no_repeat_seqs[:30]:  # Limit to avoid too many cases
        word = [bos] + list(seq)
        out = model.apply(word)
        all_cases.append((word, out.transformer_output))

    # 2. Sequences with all same token
    all_same_seqs = generate_all_same_sequences(vocab, max_len=5)
    for seq in all_same_seqs:
        word = [bos] + list(seq)
        out = model.apply(word)
        all_cases.append((word, out.transformer_output))

    # 3. Generate all short sequences (length 2-3) for comprehensive testing
    for length in range(2, 4):
        for seq_tuple in product(vocab, repeat=length):
            seq = ''.join(seq_tuple)
            word = [bos] + list(seq)
            out = model.apply(word)
            all_cases.append((word, out.transformer_output))

    # 4. Random sequences with guaranteed repeats
    with_repeat_seqs = generate_with_repeats_sequences(50, vocab, min_len=4, max_len=8)
    for seq in with_repeat_seqs:
        word = [bos] + list(seq)
        out = model.apply(word)
        all_cases.append((word, out.transformer_output))

    # 5. Some random sequences (may or may not have repeats)
    for _ in range(30):
        length = random.randint(3, 7)
        seq = ''.join(random.choice(vocab) for _ in range(length))
        word = [bos] + list(seq)
        out = model.apply(word)
        all_cases.append((word, out.transformer_output))

    return all_cases

if __name__ == "__main__":
    test_cases = generate_all_repeated_token_testcases()
    print(f"Generated {len(test_cases)} test cases")

    # Print a few examples
    for i, (input_seq, output) in enumerate(test_cases[:5]):
        print(f"Example {i+1}:")
        print(f"  Input: {input_seq}")
        print(f"  Output: {output}")
