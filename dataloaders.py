import torch
from torch.utils.data import Dataset, DataLoader
from test_datasets.generate_tests import generate_all_palindrome_testcases

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        '''
        sequences should be a list of input, output should be the true output value
        Example: [([BOS, 1, 2, 3], [BOS, 3, 2, 1]), ([BOS, 3, 1, 2], [BOS, 2, 1, 3])]
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

if __name__ == "__main__":
    makePalindromeDataLoader()