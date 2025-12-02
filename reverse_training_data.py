import torch
import random
from torch.utils.data import Dataset, DataLoader
import itertools

class ReverseDataset(Dataset):
    """Dataset for training reverse sequence models."""
    
    def __init__(self, vocab, max_seq_len=10, num_samples=1000):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.bos_token = "BOS"
        self.samples = self._generate_samples(num_samples)
        
        # Create vocabulary mapping
        self.vocab_to_idx = {token: idx for idx, token in enumerate([self.bos_token] + list(vocab))}
        self.idx_to_vocab = {idx: token for token, idx in self.vocab_to_idx.items()}
        
    def _generate_samples(self, num_samples):
        """Generate training samples for reverse task."""
        samples = []
        
        # Generate all possible sequences up to length 3
        for length in range(1, 4):
            for seq_tuple in itertools.product(self.vocab, repeat=length):
                if len(samples) >= num_samples:
                    break
                seq = list(seq_tuple)
                input_seq = [self.bos_token] + seq
                target_seq = [self.bos_token] + seq[::-1]  # Reverse the sequence
                samples.append((input_seq, target_seq))
                
        # Fill remaining samples with random sequences
        while len(samples) < num_samples:
            length = random.randint(1, min(self.max_seq_len - 1, 6))
            seq = [random.choice(list(self.vocab)) for _ in range(length)]
            input_seq = [self.bos_token] + seq
            target_seq = [self.bos_token] + seq[::-1]
            samples.append((input_seq, target_seq))
            
        return samples[:num_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        
        # Convert to indices
        input_indices = [self.vocab_to_idx[token] for token in input_seq]
        target_indices = [self.vocab_to_idx[token] for token in target_seq]
        
        # Keep sequences shorter to fit max_seq_len
        if len(input_indices) > self.max_seq_len:
            input_indices = input_indices[:self.max_seq_len]
            target_indices = target_indices[:self.max_seq_len]
        
        # Pad sequences to max_seq_len
        input_padded = input_indices + [0] * (self.max_seq_len - len(input_indices))
        target_padded = target_indices + [0] * (self.max_seq_len - len(target_indices))
        
        return (
            torch.tensor(input_padded, dtype=torch.long),
            torch.tensor(target_padded, dtype=torch.long),
            len(input_indices)  # actual length for masking
        )

def create_reverse_dataloader(vocab, max_seq_len=10, num_samples=1000, batch_size=32):
    """Create a DataLoader for reverse sequence training."""
    dataset = ReverseDataset(vocab, max_seq_len, num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_reverse_accuracy(model, vocab, max_seq_len=10, num_test=100):
    """Test the accuracy of a trained reverse model."""
    model.eval()
    correct = 0
    total = 0
    
    vocab_to_idx = {token: idx for idx, token in enumerate(["BOS"] + list(vocab))}
    idx_to_vocab = {idx: token for token, idx in vocab_to_idx.items()}
    
    with torch.no_grad():
        for _ in range(num_test):
            # Generate random test sequence
            length = random.randint(1, min(max_seq_len - 1, 4))
            seq = [random.choice(list(vocab)) for _ in range(length)]
            input_seq = ["BOS"] + seq
            expected_seq = ["BOS"] + seq[::-1]
            
            # Convert to indices and pad
            input_indices = [vocab_to_idx[token] for token in input_seq]
            input_padded = input_indices + [0] * (max_seq_len - len(input_indices))
            input_tensor = torch.tensor([input_padded], dtype=torch.long)
            
            # Get model prediction
            output = model(input_tensor)
            predicted_indices = torch.argmax(output[0], dim=-1)[:len(input_seq)]
            
            # Convert back to tokens
            predicted_seq = [idx_to_vocab[idx.item()] for idx in predicted_indices]
            
            # Check if correct (only compare non-padding positions)
            sequence_correct = True
            for i in range(len(expected_seq)):
                if predicted_seq[i] != expected_seq[i]:
                    sequence_correct = False
                    break
            
            if sequence_correct:
                correct += 1
            total += 1
    
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    # Test the data generator
    vocab = {"a", "b", "c", "d", "e"}
    dataloader = create_reverse_dataloader(vocab, max_seq_len=10, num_samples=100, batch_size=4)
    
    print("Sample batches from reverse dataloader:")
    for i, (inputs, targets, lengths) in enumerate(dataloader):
        if i >= 2:
            break
        print(f"Batch {i+1}:")
        print(f"  Inputs: {inputs}")
        print(f"  Targets: {targets}")
        print(f"  Lengths: {lengths}")
        print()
