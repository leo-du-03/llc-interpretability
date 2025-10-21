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

def generate_all_palindrome_testcases():
    all_cases = []
    bos = "BOS"
    with open("test_datasets/palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        all_cases.append((word, [bos] + [True] * len(list(line))))

    rand_pals = generate_palindromes(50)
    for line in rand_pals:
        word = [bos] + list(line)
        all_cases.append((word, [bos] + [True] * len(list(line))))

    with open("test_datasets/non_palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        line_list = list(line)
        word = [bos] + line_list
        truth = [bos]
        wordreverse = line_list + line_list[::-1]
        for i in range(0, len(line_list)):
            truth.append(wordreverse[i] == wordreverse[i + len(line_list)])
        all_cases.append((word, truth))

    rand_non_pals = generate_non_palindromes(50)
    for line in rand_non_pals:
        line_list = list(line)
        word = [bos] + line_list
        truth = [bos]
        wordreverse = line_list + line_list[::-1]
        for i in range(0, len(line_list)):
            truth.append(wordreverse[i] == wordreverse[i + len(line_list)])
        all_cases.append((word, truth))
    return all_cases