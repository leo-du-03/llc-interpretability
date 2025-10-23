import torch
from torch.utils.data import Dataset, DataLoader
from test_datasets.generate_tests import generate_all_palindrome_testcases
from test_datasets.test_peak import get_peak_test_cases
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
    
def getSequenceDataLoader(sequences):
    return DataLoader(SequenceDataset(sequences=sequences), shuffle=True)

def makePalindromeDataLoader():
    data = generate_all_palindrome_testcases()
    return getSequenceDataLoader(data)

def makePeakDataLoader():
    data = get_peak_test_cases()
    return getSequenceDataLoader(data)
    


if __name__ == "__main__":
    makePalindromeDataLoader()