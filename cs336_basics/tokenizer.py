import regex as re
from typing import Iterable, Iterator
from cs336_basics.pretokenization_example import find_chunk_boundaries
from  multiprocessing import Pool
from sortedcontainers import SortedSet
import os
import pickle
from time import time
from tqdm import tqdm
import json

def get_stats(
    ids: tuple[int, ...] | list[int], 
    counts: dict[tuple[int, int], int] | None = None,
    multiply: int = 1
    ) -> dict[tuple[int, int], int]:
    if counts is None:
        counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + multiply
    return counts

def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    result = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i != len(ids) - 1 and ids[i + 1] == pair[1]:
            result.append(idx)
            i+=2
        else:
            result.append(ids[i])
            i +=1
    return result

def pair_in_tuple(tup: tuple[int, ...], pair: tuple[int, int]) -> bool:
    for i in range(len(tup) - 1):
        if (tup[i], tup[i+1]) == pair:
            return True
    return False

PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def get_chunk_freq(text: str, special_tokens : list[str]) -> dict[tuple[int, ...], int]:
    PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    if special_tokens:
        special_pattern = "|".join(re.escape(t) for t in special_tokens)
        text_chunks = re.split(special_pattern, text)
    else:
        text_chunks = [text]
    chunks: list[str] = []
    for text_chunk in text_chunks:
        if text_chunk: 
            chunks.extend(PATTERN.findall(text_chunk))
    chunk_freq: dict[tuple[int, ...], int] = {} # {(1, 2, 3) : 52}
    for chunk in chunks:
        if chunk:
            chunk_tuple = tuple(chunk.encode('utf-8'))
            chunk_freq[chunk_tuple] = chunk_freq.get(chunk_tuple, 0) + 1
    return chunk_freq

class Tokenizer:
    def __init__(
        self,
        merges: list[tuple[bytes, bytes]] = None,
        vocab:  dict[int, bytes] = None,
        special_tokens: list[str] | None = None
        ) -> None:
        self.trained: bool = bool(vocab and merges) 
        if vocab and not merges or not vocab and merges:
            raise ValueError("Provide both vocab and merges")
        self.vocab: dict[int, bytes] = vocab or {}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.merges: dict[tuple[int, int], int] = {}
        self.special_token_mapping: dict[str, int] = {}
        for i, (p0, p1) in enumerate(merges or []):
            id0 = self.bytes_to_id[p0]
            id1 = self.bytes_to_id[p1]
            self.merges[(id0, id1)] = i + 256
        
        self.special_tokens: list[str] = sorted(special_tokens, key = lambda x: (-len(x), x)) if special_tokens else []
        self._max_token_length: int | None = None 
        self.encode_cash = {}
        self.create_special_token_mapping()
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Construct a Tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to vocabulary JSON file
            merges_filepath: Path to merges text file
            special_tokens: Optional list of special tokens
            
        Returns:
            Tokenizer instance
        """
        # Load vocabulary from JSON file
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_dict_str_keys = json.load(f)
        
        # Convert string keys to int keys
        vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else v 
                 for k, v in vocab_dict_str_keys.items()}
        
        # Load merges from text file
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse merge line, expecting format like "token1 token2"
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        token1 = parts[0].encode('utf-8')
                        token2 = parts[1].encode('utf-8')
                        merges.append((token1, token2))
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def create_special_token_mapping(self):
        self.special_token_mapping: dict[str, int] = {}
        if self.special_tokens and self.vocab:
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            self.special_pattern = re.compile(f"({pattern})")
            for token in self.special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes in self.bytes_to_id:
                    self.special_token_mapping[token] = self.bytes_to_id[token_bytes]
                else:
                    new_token_id = max(self.vocab.keys(), default=255) + 1
                    self.special_token_mapping[token] = new_token_id
                    self.vocab[new_token_id] = token_bytes
                    self.bytes_to_id[token_bytes] = new_token_id
        else:
            self.special_pattern = None
    
    def _compute_max_token_length(self):
        if self._max_token_length is not None:
            return self._max_token_length
        
        if not self.vocab:
            self._max_token_length = 100 
            return self._max_token_length
        
        max_vocab_token_len = max(len(token_bytes) for token_bytes in self.vocab.values())
        max_special_token_len = max(len(token.encode('utf-8')) for token in self.special_tokens) if self.special_tokens else 0
        self._max_token_length = max(max_vocab_token_len, max_special_token_len) + 50
        return self._max_token_length

    def create_vocab(self):
        self.vocab = {i : bytes([i]) for i in range(256)}
        for (p0, p1), ind in self.merges.items():
            self.vocab[ind] = self.vocab[p0] + self.vocab[p1]
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256
        if self.trained:
            raise ValueError("Tokenizer is already trained")
        self.trained = True
        if self.special_tokens:
            special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
            text_chunks = re.split(special_pattern, text)
        else:
            text_chunks = [text]
        
        chunks: list[str] = []
        for text_chunk in text_chunks:
            if text_chunk: 
                chunks.extend(PATTERN.findall(text_chunk))
        chunk_freq: dict[tuple[int, ...], int] = {} # {(1, 2, 3) : 52}
        for chunk in chunks:
            if chunk:
                chunk_tuple = tuple(chunk.encode('utf-8'))
                chunk_freq[chunk_tuple] = chunk_freq.get(chunk_tuple, 0) + 1

        self.vocab = {i : bytes([i]) for i in range(256)}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        self.create_special_token_mapping()
        next_id = max(self.vocab.keys()) + 1 
        num_merges = vocab_size - next_id
        
        for i in range(num_merges):
            stats = {}
            for chunk_tuple in chunk_freq:
                stats = get_stats(ids=chunk_tuple, counts=stats, multiply=chunk_freq[chunk_tuple])
            pair = max(stats.keys(), key=lambda x: (stats[x], self.vocab[x[0]], self.vocab[x[1]]))
            new_chunk_freq = {}
            for chunk_tuple, freq in chunk_freq.items():
                if pair_in_tuple(chunk_tuple, pair):
                    merged_chunk = tuple(merge(list(chunk_tuple), pair, next_id))
                    new_chunk_freq[merged_chunk] = new_chunk_freq.get(merged_chunk, 0) + freq
                else:
                    new_chunk_freq[chunk_tuple] = freq
            chunk_freq = new_chunk_freq
            self.merges[pair] = next_id
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            next_id += 1
        
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self._max_token_length = None 

    def encode_casual(self, text):
        ids = [self.bytes_to_id[bytes([byte])] for byte in text.encode('utf-8')]
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key = lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_with_regex_split(self, text):
        chunks = PATTERN.findall(text)
        result = []
        for chunk in chunks:
            if chunk not in self.encode_cash:
                encoded = self.encode_casual(chunk)
                result.extend(encoded)
                self.encode_cash[chunk] = encoded
            else:
                result.extend(self.encode_cash[chunk])
        return result
        
    def decode(self, ids: list[int]) -> str:
        tokens = b"".join([self.vocab[idx] for idx in ids])
        return tokens.decode('utf-8', errors='replace')
    
    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_with_regex_split(text)
        else:
            if not self.special_pattern:
                raise ValueError("Tokenizer's vocab is not created, but special tokens are set")
            
            parts = self.special_pattern.split(text)
            # Result: ["text", "<|endoftext|>", "more text", "<|endoftext|>", ...]
            result = []
            for part in parts:
                if part not in self.special_token_mapping and part: #O(1) search in a mapping and part != ""
                    encoded = self.encode_with_regex_split(part)
                    result.extend(encoded)
                elif part: # is special token and != ""
                    result.append(self.special_token_mapping[part])
            return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files.
        
        Args:
            iterable: An iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        CHUNK_SIZE = 1024 * 4
        SLICE_BACK_TOKENS = self._compute_max_token_length()
        
        buffer = ""
        for chunk in iter(lambda: iterable.read(CHUNK_SIZE), ""):
            buffer += chunk
            if chunk != "":
                # encode all text in buffer
                encoded = self.encode(buffer)
                # get all tokens except the last SLICE_BACK_TOKENS
                to_yield_from = encoded[:-SLICE_BACK_TOKENS]
                # last SLICE_BACK_TOKENS tokens
                to_buffer_tokens = encoded[-SLICE_BACK_TOKENS:]
                # decode them back, they are our new buffer
                buffer = self.decode(to_buffer_tokens)
                if to_yield_from:
                    for token_id in to_yield_from:
                        yield token_id
        
        # Encode and yield any remaining buffer
        if buffer:
            ids = self.encode(buffer)
            for token_id in ids:
                yield token_id
        

    def train_from_file(self, filename, vocab_size):
        assert vocab_size >= 256
        if self.trained:
            raise ValueError("Tokenizer is already trained")
        self.trained = True
        args_list = []
        t0 = time()
        chunk_freq = {}
        with open(filename, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes * 10, b"<|endoftext|>")
            print(f"Num bouns: {len(boundaries)}")
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            with Pool(processes=num_processes) as p:
                boundary_pairs = list(zip(boundaries[:-1], boundaries[1:]))
                for start, end in tqdm(boundary_pairs, desc="Pretokenization", unit="chunk"):
                    f.seek(start)
                    part = f.read(end - start).decode("utf-8", errors="ignore")
                    args_list.append((part, self.special_tokens))
                    if len(args_list) == num_processes: #heuristic for ~8GB of RAM available
                        
                        list_of_freqs = p.starmap(get_chunk_freq, args_list)
                        for dct in list_of_freqs:
                            for k, v in dct.items():
                                chunk_freq[k] = chunk_freq.get(k, 0) + v
                        args_list = []

                if args_list:
                    list_of_freqs = p.starmap(get_chunk_freq, args_list)
                    for dct in list_of_freqs:
                        for k, v in dct.items():
                            chunk_freq[k] = chunk_freq.get(k, 0) + v  

        print(f"Pretokenization time: {time() - t0}")
        self.vocab = {i : bytes([i]) for i in range(256)}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        pair_to_chunk_map: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for chunk in chunk_freq:
            for pair in zip(chunk, chunk[1:]):
                if pair not in pair_to_chunk_map:
                    pair_to_chunk_map[pair] = set()
                pair_to_chunk_map[pair].add(chunk)
        
        self.create_special_token_mapping()
        next_id = max(self.vocab.keys()) + 1 
        num_merges = vocab_size - next_id
        
        map_pair_to_count = {}
        for chunk in chunk_freq:
            for pair in zip(chunk, chunk[1:]):
                map_pair_to_count[pair] = map_pair_to_count.get(pair, 0) + chunk_freq[chunk]
        
        pair_counts_sorted = SortedSet()
        for pair, count in map_pair_to_count.items():
            pair_counts_sorted.add((count, self.vocab[pair[0]], self.vocab[pair[1]]))
        
        for i in range(num_merges):
            top_entry = pair_counts_sorted.pop()
            count, tok1, tok2 = top_entry
            tok1 = self.bytes_to_id[tok1]
            tok2 = self.bytes_to_id[tok2]
            pair = (tok1, tok2)
            self.merges[pair] = next_id
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.bytes_to_id[self.vocab[next_id]] = next_id
            touched_chunks = pair_to_chunk_map[pair]
            pairs_before = {} 
            pairs_after = {}
            for chunk_tuple in list(touched_chunks): 
                freq = chunk_freq[chunk_tuple]
                for p in zip(chunk_tuple, chunk_tuple[1:]):
                    pairs_before[p] = pairs_before.get(p, 0) + freq

                for p in zip(chunk_tuple, chunk_tuple[1:]):
                    pair_to_chunk_map[p].discard(chunk_tuple)
                
                merged = tuple(merge(list(chunk_tuple), pair, next_id))
                
                for p in zip(merged, merged[1:]):
                    pairs_after[p] = pairs_after.get(p, 0) + freq
                
                for p in zip(merged, merged[1:]):
                    pair_to_chunk_map.setdefault(p, set()).add(merged)
                
                chunk_freq[merged] = chunk_freq.get(merged, 0) + freq
                chunk_freq.pop(chunk_tuple)

            pair_to_chunk_map.pop(pair, None)
            
            all_affected_pairs = set(pairs_before.keys()) | set(pairs_after.keys())
            for p in all_affected_pairs:
                old_count = map_pair_to_count.get(p, 0)
                before_count = pairs_before.get(p, 0)
                after_count = pairs_after.get(p, 0)
                new_count = old_count - before_count + after_count

                if old_count > 0:
                    pair_counts_sorted.discard((old_count, self.vocab[p[0]], self.vocab[p[1]]))
                
                if new_count > 0:
                    map_pair_to_count[p] = new_count
                    pair_counts_sorted.add((new_count, self.vocab[p[0]], self.vocab[p[1]]))
                elif p in map_pair_to_count:
                    del map_pair_to_count[p]
            next_id += 1
        
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self._max_token_length = None 

    def serialize_with_pickle(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        merges_list = []
        for (id0, id1), new_id in sorted_merges:
            bytes0 = self.vocab[id0]
            bytes1 = self.vocab[id1]
            merges_list.append((bytes0, bytes1))
        model = {"vocab": self.vocab, "merges": merges_list}
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    def deserialize_with_pickle(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model['vocab'], model['merges']

if __name__ == "__main__":
    tok = Tokenizer(special_tokens=['<|endoftext|>'])
    t0 = time()
    tok.train_from_file(filename="data/owt_train.txt", vocab_size=32000)
    print(f"Trained for {time() - t0} seconds")
    tok.serialize_with_pickle(filename="saves/owt_first.pkl")