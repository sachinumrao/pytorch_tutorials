import spacy
import torch
from numpy.testing import assert_almost_equal

is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

nlp = spacy.load("en_pytt_bertbaseuncased_lg")
doc = nlp("Here is some text to encode.")

assert doc.tensor.shape == (7, 768)  # Always has one row per token

doc._.pytt_word_pieces_  # String values of the wordpieces
doc._.pytt_word_pieces  # Wordpiece IDs (note: *not* spaCy's hash values!)
doc._.pytt_alignment  # Alignment between spaCy tokens and wordpieces

# The raw transformer output has one row per wordpiece.
assert len(doc._.pytt_last_hidden_state) == len(doc._.pytt_word_pieces)

# To avoid losing information, we calculate the doc.tensor attribute such that
# the sum-pooled vectors match (apart from numeric error)
assert_almost_equal(doc.tensor.sum(axis=0), doc._.pytt_last_hidden_state.sum(axis=0))
span = doc[2:4]

# Access the tensor from Span elements (especially helpful for sentences)
assert span.tensor == doc.tensor[2:4]

# .vector and .similarity use the transformer outputs
apple1 = nlp("Apple shares rose on the news.")
apple2 = nlp("Apple sold fewer iPhones this quarter.")
apple3 = nlp("Apple pie is delicious.")

print(apple1[0].similarity(apple2[0]))  # 0.73428553
print(apple1[0].similarity(apple3[0]))  # 0.43365782