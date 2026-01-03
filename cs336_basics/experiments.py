from tokenizer import Tokenizer

vocab, merges = Tokenizer.deserialize_with_pickle('saves/owt_first.pkl')

sorted_merges = sorted(merges.items(), key=lambda x: x[1])
merges_list = []
for (id0, id1), new_id in sorted_merges:
    bytes0 = vocab[id0]
    bytes1 = vocab[id1]
    merges_list.append((bytes0, bytes1))

tok = Tokenizer(vocab=vocab, merges=merges_list, special_tokens=['<|endoftext|>'])

max_token = max(vocab.values(), key=len)
print(max_token.decode("utf-8"))
print(len(max_token))

sorted_vocab = sorted(vocab.values(), key=len, reverse=True)
for x in sorted_vocab[:20]:
    print(x.decode("utf-8"))
    print(len(x))