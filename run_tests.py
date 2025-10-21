from palindrome import check_palindrome
from test_datasets.generate_tests import generate_non_palindromes, generate_palindromes

def palindromesTests():
    # NOTE: RASP doesn't seem to be able to properly calculate the length of any array with length >= 10, always too low by 1 in this case.
    bos = "BOS"
    model = check_palindrome()

    out = model.apply([bos, "a", "b", "a"])

    with open("test_datasets/palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        out = model.apply(word)
        assert((not False in out.decoded) == True)

    rand_pals = generate_palindromes(50)
    for line in rand_pals:
        word = [bos] + list(line)
        out = model.apply(word)
        assert((not False in out.decoded) == True)

    with open("test_datasets/non_palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        out = model.apply(word)
        line_list = list(line)
        word = [bos] + line_list
        truth = [bos]
        wordreverse = line_list + line_list[::-1]
        for i in range(0, len(line_list)):
            truth.append(wordreverse[i] == wordreverse[i + len(line_list)])
        assert(out.decoded == truth)

    rand_non_pals = generate_non_palindromes(50)
    for line in rand_non_pals:
        word = [bos] + list(line)
        out = model.apply(word)
        line_list = list(line)
        word = [bos] + line_list
        truth = [bos]
        wordreverse = line_list + line_list[::-1]
        for i in range(0, len(line_list)):
            truth.append(wordreverse[i] == wordreverse[i + len(line_list)])
        assert(out.decoded == truth)

from peak import get_peak_model
from test_datasets.test_peak import get_peak_test_cases
def peakTests():
    test_cases = get_peak_test_cases()
    bos = "BOS"
    model = get_peak_model()

    for i, (input_seq, expected_output) in enumerate(test_cases, 1):
        out = model.apply(input_seq)
        decoded = out.decoded

        assert decoded == expected_output, (
            # f"\n--- Test Case {i} Failed ---\n"
            # f"Input:    {input_seq}\n"
            # f"Expected: {expected_output}\n"
            # f"Got:      {decoded}\n"
        )
        # print(f"--- Test Case {i} ---")
        # print(f"Input:    {input_seq}")
        # print(f"Expected: {expected_output}")
        # print(f"Output:   {decoded}")
        # print("Passed!\n")

if __name__ == "__main__":
    palindromesTests()
    peakTests()