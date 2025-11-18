import torch
from torch.utils.data import Dataset, DataLoader
from test_datasets.generate_tests import generate_all_palindrome_testcases
from test_datasets.test_peak import get_peak_test_cases

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
    data = get_peak_test_cases()
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


if __name__ == "__main__":
    makePalindromeDataLoader()
    makePeakDataLoader()