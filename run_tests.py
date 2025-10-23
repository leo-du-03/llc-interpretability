from palindrome import check_palindrome
from tracr.haiku_to_pytorch import haiku_to_pytorch, apply, outputs_equal
from test_datasets.generate_tests import generate_non_palindromes, generate_palindromes

def palindromesTests():
    # NOTE: RASP doesn't seem to be able to properly calculate the length of any array with length >= 10, always too low by 1 in this case.
    bos = "BOS"
    model = check_palindrome()
    torch_model = haiku_to_pytorch(model)

    with open("test_datasets/palindromes.txt") as file:
        lines = [line.rstrip() for line in file]
    for line in lines:
        word = [bos] + list(line)
        out = model.apply(word)
        pytorch_out = apply(torch_model, word)
        assert(outputs_equal(out.transformer_output, pytorch_out))
        assert((not False in out.decoded) == True)

    rand_pals = generate_palindromes(50)
    for line in rand_pals:
        word = [bos] + list(line)
        out = model.apply(word)
        pytorch_out = apply(torch_model, word)
        assert(outputs_equal(out.transformer_output, pytorch_out))
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
        pytorch_out = apply(torch_model, word)
        assert(outputs_equal(out.transformer_output, pytorch_out))
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
        pytorch_out = apply(torch_model, word)
        assert(outputs_equal(out.transformer_output, pytorch_out))
        assert(out.decoded == truth)

if __name__ == "__main__":
    palindromesTests()
    peakTests()