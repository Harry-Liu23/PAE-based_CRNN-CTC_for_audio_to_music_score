from pae_preprocess import normalize_pae_text
from pae_tokenizer import PAETokenizer

train_pae_texts = [
    normalize_pae_text("G-2 4/4 xF c d e f | g a b c"),
    normalize_pae_text("F-4 3/4 bB 4c 4d 4e | 2f"),
]

tokenizer = PAETokenizer.from_texts(train_pae_texts)

print("Vocab size including blank:", tokenizer.vocab.vocab_size)
print("Blank ID:", tokenizer.vocab.blank_id)

sample = train_pae_texts[0]
ids = tokenizer.encode(sample)
print("Text:", sample)
print("Encoded:", ids)
print("Decoded raw:", tokenizer.decode_raw(ids))