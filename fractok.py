from tracr.rasp import rasp
from tracr.compiler import compiling

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)

#fraction of prev tokens that are x
def rasp_fraction_x_prev_tokens():
  is_x = rasp.Map(lambda t: 1 if t == "x" else 0, rasp.tokens)
  is_x = rasp.numerical(is_x)
  prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ) #select prev tokens
  frac_prevs = rasp.Aggregate(prevs, is_x, 0) #get mean of prev tokens
  frac_prevs = rasp.numerical(frac_prevs)
  return frac_prevs

def check_fractok(max_seq_len=10, vocab_size='medium'):
  bos = "BOS"
  if vocab_size == 'small':
    vocab = {"x"} | set(chr(i) for i in range(ord('a'), ord('f')))  # a-e
  elif vocab_size == 'medium':
    vocab = {"x"} | set(chr(i) for i in range(ord('a'), ord('n')))  # a-m
  elif vocab_size == 'large':
    vocab = {"x"} | set(chr(i) for i in range(ord('a'), ord('z') + 1))  # a-z
  else:
    raise ValueError(f"Unknown vocab_size: {vocab_size}")
  model = compiling.compile_rasp_to_model(
      program = rasp_fraction_x_prev_tokens(),
      vocab=vocab,
      max_seq_len=max_seq_len,
      compiler_bos=bos,
  )
  return model

# out = model.apply([bos, "x", "a", "a", "x"])
# print(out.decoded)
