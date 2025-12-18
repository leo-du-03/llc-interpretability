"""
Contains tests for the palindrome detection model.
Ensures that the model is accurate.
"""

from rasp_models.palindrome import check_palindrome
from datasets.palindrome_data import generate_non_palindromes, generate_palindromes

def palindromesTests():
    '''
    Makes sure the palindrome model is outputing the correct answers
    '''
    bos = "BOS"
    model = check_palindrome()

    test_cases = []

    print("Compiling test cases:")
    with open("tests/palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        test_cases.append(word)

    rand_pals = generate_palindromes(50)
    for line in rand_pals:
        word = [bos] + list(line)
        test_cases.append(word)
    
    with open("tests/non_palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        test_cases.append(word)

    rand_non_pals = generate_non_palindromes(50)
    for line in rand_non_pals:
        word = [bos] + list(line)
        test_cases.append(word)
    print("done.")

    for i, word in enumerate(test_cases, 1):
        out = model.apply(word)
        target = ['BOS'] + [a == b for a, b in zip(word[1:], word[1:][::-1])]
        
        if target == out.decoded:
            print(".", end="", flush=True)
        else:
            print(
                f"\n--- Decoded Output Mismatch in Test Case {i} ---\n"
                f"Input:    {word}\n"
                f"Expected: {target}\n"
                f"Got:      {out.decoded}\n"
            )
    print("\n")