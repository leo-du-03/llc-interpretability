import random
from reverse import check_reverse
from itertools import product

def generate_random_sequences(n, vocab=['a', 'b', 'c', 'd', 'e'], min_len=1, max_len=6):
    """Generate n random sequences from the given vocab."""
    sequences = []
    for _ in range(n):
        length = random.randint(min_len, max_len)
        seq = ''.join(random.choice(vocab) for _ in range(length))
        sequences.append(seq)
    return sequences

def generate_all_reverse_testcases():
    """Generate test cases for the reverse algorithm."""
    all_cases = []
    bos = "BOS"
    model = check_reverse()

    vocab = ['a', 'b', 'c', 'd', 'e']

    # Generate all sequences up to length 3 for comprehensive testing
    for length in range(1, 4):
        for seq_tuple in product(vocab, repeat=length):
            seq = ''.join(seq_tuple)
            word = [bos] + list(seq)
            out = model.apply(word)
            all_cases.append((word, out.transformer_output))

    # Generate random sequences of varying lengths
    random_seqs = generate_random_sequences(100, vocab=vocab, min_len=4, max_len=8)
    for seq in random_seqs:
        word = [bos] + list(seq)
        out = model.apply(word)
        all_cases.append((word, out.transformer_output))

    return all_cases

if __name__ == "__main__":
    test_cases = generate_all_reverse_testcases()
    print(f"Generated {len(test_cases)} test cases")

    # Print a few examples
    for i, (input_seq, output) in enumerate(test_cases[:5]):
        print(f"Example {i+1}:")
        print(f"  Input: {input_seq}")
        print(f"  Output: {output}")
