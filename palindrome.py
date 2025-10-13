from tracr.rasp import rasp
from tracr.compiler import compiling
import random

def generate_palindromes(n):
    palindromes = []
    for _ in range(n):
        word = ""
        length = random.randint(1, 4)
        for _ in range(length):
            word += chr(random.randint(ord('a'), ord('z')))
        word += word[::-1]
        rem_mid = bool(random.randint(0, 1))
        if (rem_mid and len(word) > 1) or len(word) == 10:
            half = len(word) // 2
            word = word[0:half] + word[half + 1:len(word)]
        palindromes.append(word)
    return palindromes

def generate_non_palindromes(n):
    non_palindromes = []
    for _ in range(n):
        word = ""
        length = random.randint(1, 9)
        for _ in range(length):
            word += chr(random.randint(ord('a'), ord('z')))
        if word != word[::-1]:
            non_palindromes.append(word)
    return non_palindromes

def is_palindrome(tokens):
    all_true_selector = rasp.Select(tokens, tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true_selector)

    opp_idx = (length - rasp.indices).named("opp_idx")
    opp_idx = (opp_idx - 1).named("opp_idx-1")
    reverse_selector = rasp.Select(rasp.indices, opp_idx, rasp.Comparison.EQ)
    reverse = rasp.Aggregate(reverse_selector, tokens)

    check = rasp.SequenceMap(lambda x, y: x == y, tokens, reverse)

    return check

def check_palindrome():
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        program=is_palindrome(rasp.tokens),
        vocab={"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "},
        max_seq_len=50,
        compiler_bos=bos,
    )

    return model

# NOTE: RASP doesn't seem to be able to properly calculate the length of any array with length >= 10, always too low by 1 in this case.
bos = "BOS"
model = check_palindrome()
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
    assert((not False in out.decoded) == False)

rand_non_pals = generate_non_palindromes(50)
for line in rand_non_pals:
    word = [bos] + list(line)
    out = model.apply(word)
    assert((not False in out.decoded) == False)
