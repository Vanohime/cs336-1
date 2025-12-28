import regex as re
from typing import Iterable
import pickle
import os
import multiprocessing
from time import time

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

def get_stats_from_list(chunks: list[tuple[int, ...]], freqs: list[int]):
    stats = {}
    for chunk, freq in zip(chunks, freqs):
        stats = get_stats(ids=chunk, counts=stats, multiply=freq)
    return stats

def prepare_chunks_for_multiproc(chunk_freq: dict[tuple, int], num_processes: int = 4):
    total_len = sum(len(x) for x in chunk_freq)
    len_per_process = total_len // num_processes
    chunk_lists = [([], []) for _ in range(num_processes)]
    for i, chunk in enumerate(chunk_freq):
        idx = i % num_processes
        chunk_lists[idx][0].append(chunk)
        chunk_lists[idx][1].append(chunk_freq[chunk])
    assert len(chunk_lists) == num_processes
    return chunk_lists
 
PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(
        self,
        merges: list[tuple[bytes, bytes]] = [],
        vocab:  dict[int, bytes] = {},
        special_tokens: list[str] | None = None
        ) -> None:
        self.trained: bool = bool(vocab and merges) 
        if vocab and not merges or not vocab and merges:
            raise ValueError("Provide both vocab and merges")
        self.vocab: dict[int, bytes] = vocab
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.merges: dict[tuple[int, int], int] = {}
        self.special_token_mapping: dict[str, int] = {}
        for i, (p0, p1) in enumerate(merges):
            id0 = self.bytes_to_id[p0]
            id1 = self.bytes_to_id[p1]
            self.merges[(id0, id1)] = i + 256
        
        self.special_tokens: list[str] = sorted(special_tokens, key = lambda x: (-len(x), x)) if special_tokens else []
        self._max_token_length: int | None = None 
        self.create_special_token_mapping()
    
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
            t0 = time()
            special_pattern = "|".join(re.escape(t) for t in self.special_tokens)
            text_chunks = re.split(special_pattern, text)
            print(f"Removing special tokens: {time() - t0} sec")
        else:
            text_chunks = [text]
        
        # Применяем PATTERN к каждому чанку отдельно
        chunks: list[str] = []
        t0 = time()
        for text_chunk in text_chunks:
            if text_chunk:  # фильтруем пустые строки
                chunks.extend(PATTERN.findall(text_chunk))
        print(f"Chunking with regex: {time() - t0}")
        chunk_freq: dict[tuple[int, ...], int] = {} # {(1, 2, 3) : 52}
        t0 = time()
        for chunk in chunks:
            if chunk:
                chunk_tuple = tuple(chunk.encode('utf-8'))
                chunk_freq[chunk_tuple] = chunk_freq.get(chunk_tuple, 0) + 1
        print(f"Conputing frequensies for each chunk: {time() - t0}")
        self.vocab = {i : bytes([i]) for i in range(256)}
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        self.create_special_token_mapping()
        next_id = max(self.vocab.keys()) + 1 
        num_merges = vocab_size - next_id
        
        stats_latency = 0
        merge_latency = 0
        num_processes = 1
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i in range(num_merges):
                stats = {}
                t0 = time()
                arg_list = prepare_chunks_for_multiproc(chunk_freq, num_processes)
                
                results = pool.starmap(get_stats_from_list, arg_list)
                stats = results[0]
                for res in results[1:]:
                    for stat in res:
                        stats[stat] = stats.get(stat, 0) + res[stat]
                stats_latency += time() - t0

                pair = max(stats.keys(), key=lambda x: (stats[x], self.vocab[x[0]], self.vocab[x[1]]))
                new_chunk_freq = {}
                t0 = time()
                for chunk_tuple, freq in chunk_freq.items():
                    if pair_in_tuple(chunk_tuple, pair):
                        merged_chunk = tuple(merge(list(chunk_tuple), pair, next_id))
                        new_chunk_freq[merged_chunk] = new_chunk_freq.get(merged_chunk, 0) + freq
                    else:
                        new_chunk_freq[chunk_tuple] = freq
                merge_latency += time() - t0
                chunk_freq = new_chunk_freq
                self.merges[pair] = next_id
                self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
                next_id += 1
        
        print(f"Stats total latency: {stats_latency}")
        print(f"Merges total latency: {merge_latency}")
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
            result.extend(self.encode_casual(chunk))
        return result
        
    def decode(self, ids):
        tokens = b"".join([self.vocab[idx] for idx in ids])
        return tokens.decode('utf-8', errors='replace')
    
    def encode(self, text):
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

    def encode_iterable(self, iterable):
        CHUNK_SIZE = 1024
        SLICE_BACK_TOKENS = self._compute_max_token_length()
        
        buffer = ""
        for chunk in iter(lambda: iterable.read(CHUNK_SIZE), ""):
            buffer += chunk
            if chunk != "":
                # encode all text
                encoded = self.encode(buffer)
                # get all tokens except the last SLICE_BACK_TOKENS
                to_yield_from = encoded[:-SLICE_BACK_TOKENS]
                # last SLICE_BACK_TOKENS tokens
                to_buffer_tokens = encoded[-SLICE_BACK_TOKENS:]
                # decode them back, they are our new buffer
                decoded_back = self.decode(to_buffer_tokens)
                buffer = decoded_back
                if to_yield_from:
                    for token_id in to_yield_from:
                        yield token_id
        
        if buffer:
            ids = self.encode(buffer)
            for token_id in ids:
                yield token_id
    
    def serialize_with_pickle(vocab, merges, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        model = {"vocab": vocab, "merges": merges}
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    def deserialize_with_pickle(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model['vocab'], model['merges']
        

