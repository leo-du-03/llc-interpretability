import torch
from torch.utils.data import Dataset, DataLoader
from datasets.palindrome_data import generate_all_palindrome_testcases
from datasets.peak_data import get_peak_test_cases
from datasets.fractok_data import generate_all_prev_fraction_tokens_x_testcases, generate_fractok_training_data
from datasets.histogram_data import get_histogram_test_cases
from datasets.sort_data import get_sort_test_cases
from datasets.reverse_data import get_reverse_test_cases
from datasets.triplets_data import get_triplets_test_cases

VOCAB = ['BOS'] + list("abcdefghijklmnopqrstuvwxyz")  # include all chars you expect
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}

def encode_sequence(seq):
    """Convert a list of characters to integer IDs"""
    return [CHAR2IDX[ch] if isinstance(ch, str) else ch for ch in seq]

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        '''
        sequences should be a list of input, output should be the true output value
        should be raw outputs, not decoded, of shape (1, SEQ_LEN, OUT_DIM)
        '''
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input, truth = self.sequences[idx]
        return input, truth
    
def identityCollator(batch):
    return batch
    
def getSequenceDataLoader(sequences):
    # return DataLoader(SequenceDataset(sequences=sequences), collate_fn=tensor_collate, shuffle=True)
    return DataLoader(SequenceDataset(sequences=sequences), batch_size=1, collate_fn=identityCollator, shuffle=True)

def makePalindromeDataLoader(num_palin):
    data = generate_all_palindrome_testcases(num_palin)
    return getSequenceDataLoader(data)

def makePeakDataLoader():
    data = get_peak_test_cases(0)
    return getSequenceDataLoader(data)

def makeDomPeakDataLoader():
    data = get_peak_test_cases(1)
    return getSequenceDataLoader(data)

def makeFractokDataLoader(max_seq_len=10, vocab_size='medium'):
    # Placeholder: Replace with actual fractok data generation
    data = generate_all_prev_fraction_tokens_x_testcases(max_seq_len=max_seq_len, vocab_size=vocab_size)  # Should be list of (input, truth) pairs
    return getSequenceDataLoader(data)

def makeHistogramDataLoader(vocab=None, max_seq_len=10):
    data = get_histogram_test_cases(vocab=vocab, max_seq_len=max_seq_len)
    return getSequenceDataLoader(data)

def makeSortDataLoader(vocab=None, max_seq_len=10):
    data = get_sort_test_cases(vocab=vocab, max_seq_len=max_seq_len)
    return getSequenceDataLoader(data)

def makeReverseDataLoader(vocab=None, max_seq_len=10):
    data = get_reverse_test_cases(vocab=vocab, max_seq_len=max_seq_len)
    return getSequenceDataLoader(data)

def makeTripletsDataLoader(vocab=None, max_seq_len=10):
    data = get_triplets_test_cases(vocab=vocab, max_seq_len=max_seq_len)
    return getSequenceDataLoader(data)
    
def tensor_collate(batch):
    inputs, targets = zip(*batch)
    
    # encode inputs (keep BOS in inputs if your model expects it)
    # inputs = [encode_sequence(seq) for seq in inputs]
    # inputs = torch.tensor(inputs, dtype=torch.long)

    # convert targets to floats, skip 'BOS' if present
    processed_targets = []
    for t in targets:
        t_numeric = []
        for x in t:
            if x == 'BOS':
                continue  # skip BOS in targets
            if isinstance(x, bool):
                t_numeric.append(float(x))
            else:
                # if x is a number in string form
                t_numeric.append(float(x))
        processed_targets.append(t_numeric)

    targets = torch.tensor(processed_targets, dtype=torch.float)
    return inputs, targets

def makeFractokTrainDataLoader(max_seq_len=10, vocab_size='medium', batch_size=32):
    # Use the NEW function for training
    data = generate_fractok_training_data(n=1000, max_seq_len=max_seq_len, vocab_size=vocab_size)
    
    def collate_for_training(batch):
        inputs_list, targets_list = [], []
        
        for inputs, targets in batch:
            input_ids = torch.tensor([CHAR2IDX.get(token, 0) for token in inputs], dtype=torch.long)
            inputs_list.append(input_ids)
            targets_list.append(targets.float())
        
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs_list, batch_first=True, padding_value=0)
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets_list, batch_first=True, padding_value=0.0)
        
        return inputs_padded, targets_padded
    
    return DataLoader(SequenceDataset(sequences=data), batch_size=batch_size, collate_fn=collate_for_training, shuffle=True)

if __name__ == "__main__":
    makePalindromeDataLoader()
    makePeakDataLoader()
    makeFractokDataLoader()
    makeDomPeakDataLoader()
    makeHistogramDataLoader()
    makeSortDataLoader()
    makeReverseDataLoader()
    makeTripletsDataLoader()