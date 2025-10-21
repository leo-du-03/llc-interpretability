from tracr.rasp import rasp
from tracr.compiler import compiling

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)

#fraction of prev tokens that are x
is_x = rasp.Map(lambda t: 1 if t == "x" else 0, rasp.tokens)
is_x = rasp.numerical(is_x)
prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ) #select prev tokens
frac_prevs = rasp.Aggregate(prevs, is_x, 0) #get mean of prev tokens
frac_prevs = rasp.numerical(frac_prevs)

bos = "BOS"
model = compiling.compile_rasp_to_model(
    frac_prevs,
    vocab={"x", "a", "c"},
    max_seq_len=10,
    compiler_bos=bos,
)

out = model.apply([bos, "x", "a", "a", "x"])
print(out.decoded)
