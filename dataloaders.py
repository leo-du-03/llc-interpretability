import torch
from torch.utils.data import Dataset, DataLoader
from test_datasets.generate_tests import generate_all_palindrome_testcases

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
    return DataLoader(SequenceDataset(sequences=sequences), batch_size=1, collate_fn=identityCollator, shuffle=True)

def makePalindromeDataLoader():
    data = generate_all_palindrome_testcases()
    return getSequenceDataLoader(data)

if __name__ == "__main__":
    makePalindromeDataLoader()