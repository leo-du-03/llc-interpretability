import random
from rasp_models.palindrome import check_palindrome
import numpy as np
import torch

def generate_palindromes(n, end_char='z'):
    palindromes = []
    for _ in range(n):
        word = ""
        length = random.randint(1, 4)
        for _ in range(length):
            word += chr(random.randint(ord('a'), ord(end_char)))
        word += word[::-1]
        rem_mid = bool(random.randint(0, 1))
        if (rem_mid and len(word) > 1) or len(word) == 10:
            half = len(word) // 2
            word = word[0:half] + word[half + 1:len(word)]
        palindromes.append(word)
    return palindromes

def generate_non_palindromes(n, end_char='z'):
    non_palindromes = []
    for _ in range(n):
        word = ""
        length = random.randint(1, 9)
        for _ in range(length):
            word += chr(random.randint(ord('a'), ord(end_char)))
        if word != word[::-1]:
            non_palindromes.append(word)
    return non_palindromes

def generate_all_palindrome_testcases(num_palin, end_char='z'):
    all_cases = []
    bos = "BOS"
    model = check_palindrome()

    rand_pals = generate_palindromes(num_palin, end_char)
    for line in rand_pals:
        word = [bos] + list(line)
        out = model.apply(word)
        all_cases.append((word, torch.tensor(np.array(out.transformer_output), dtype=torch.float64)))

    rand_non_pals = generate_non_palindromes(num_palin, end_char)
    for line in rand_non_pals:
        line_list = list(line)
        word = [bos] + line_list
        out = model.apply(word)
        all_cases.append((word, torch.tensor(np.array(out.transformer_output), dtype=torch.float64)))
    return all_cases