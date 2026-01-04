"""
Benchmark script to measure tokenizer throughput.
Measures encoding speed in bytes per second for different tokenizers.
"""
import os
from time import time
from tokenizer import Tokenizer


def load_tokenizer_from_pickle(pkl_path: str, special_tokens: list[str] | None = None) -> Tokenizer:
    """Load a tokenizer from a pickle file."""
    vocab, merges_list = Tokenizer.deserialize_with_pickle(pkl_path)
    
    # Check if merges_list is in the correct format (bytes, bytes) or old format (int, int)
    if merges_list and len(merges_list) > 0:
        first_merge = merges_list[0]
        if isinstance(first_merge[0], int):
            # Old format: convert (int, int) to (bytes, bytes)
            print("  Note: Converting old merge format to new format...")
            new_merges = []
            for id0, id1 in merges_list:
                bytes0 = vocab[id0]
                bytes1 = vocab[id1]
                new_merges.append((bytes0, bytes1))
            merges_list = new_merges
    
    return Tokenizer(vocab=vocab, merges=merges_list, special_tokens=special_tokens)


def benchmark_tokenizer(
    tokenizer: Tokenizer, 
    text_file: str, 
    name: str,
    num_bytes: int | None = None,
    chunk_size: int = 8192,
    num_warmup_runs: int = 1
) -> dict:
    """
    Benchmark a tokenizer on a text file using encode_iterable.
    
    Args:
        tokenizer: The tokenizer to benchmark
        text_file: Path to the text file
        name: Name of the dataset/tokenizer for display
        num_bytes: Number of bytes to read (None = read all)
        chunk_size: Size of chunks to yield to encode_iterable
        num_warmup_runs: Number of warmup runs before timing
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"File: {text_file}")
    print(f"{'='*70}")
    
    # Get file size
    file_size = os.path.getsize(text_file)
    if num_bytes is not None:
        text_size_bytes = min(num_bytes, file_size)
    else:
        text_size_bytes = file_size
    
    text_size_mb = text_size_bytes / (1024 ** 2)
    
    print(f"Text size: {text_size_bytes:,} bytes ({text_size_mb:.2f} MB)")
    print(f"Chunk size: {chunk_size:,} bytes")
    
    # Helper function to create chunk iterator (same as tokenize_datasets.py)
    def read_chunks(file_path, chunk_size, max_bytes=None):
        """Generator that yields chunks of text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            bytes_read = 0
            while True:
                # Determine how much to read
                if max_bytes is not None:
                    remaining = max_bytes - bytes_read
                    if remaining <= 0:
                        break
                    read_size = min(chunk_size, remaining)
                else:
                    read_size = chunk_size
                
                chunk = f.read(read_size)
                if not chunk:
                    break
                
                bytes_read += len(chunk.encode('utf-8'))
                yield chunk
    
    # Warmup runs
    if num_warmup_runs > 0:
        print(f"Running {num_warmup_runs} warmup iteration(s)...", end=' ', flush=True)
        for _ in range(num_warmup_runs):
            warmup_tokens = list(tokenizer.encode_iterable(read_chunks(text_file, chunk_size, num_bytes)))
        print("Done")
    
    # Benchmark encoding with encode_iterable
    print("Benchmarking encoding (encode_iterable)...", end=' ', flush=True)
    t0 = time()
    tokens = list(tokenizer.encode_iterable(read_chunks(text_file, chunk_size, num_bytes)))
    elapsed = time() - t0
    print("Done")
    
    # Calculate metrics
    throughput_bytes_per_sec = text_size_bytes / elapsed
    throughput_mb_per_sec = throughput_bytes_per_sec / (1024 ** 2)
    num_tokens = len(tokens)
    tokens_per_sec = num_tokens / elapsed
    bytes_per_token = text_size_bytes / num_tokens if num_tokens > 0 else 0
    
    # Print results
    print(f"\n{'-'*70}")
    print(f"Results:")
    print(f"{'-'*70}")
    print(f"  Total tokens:        {num_tokens:,}")
    print(f"  Encoding time:       {elapsed:.3f} seconds")
    print(f"  Throughput:          {throughput_bytes_per_sec:,.0f} bytes/sec")
    print(f"                       {throughput_mb_per_sec:.2f} MB/sec")
    print(f"  Tokens per second:   {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Bytes per token:     {bytes_per_token:.2f}")
    print(f"{'-'*70}")
    
    # Benchmark decoding
    print("Benchmarking decoding...", end=' ', flush=True)
    t0 = time()
    decoded_text = tokenizer.decode(tokens)
    decode_elapsed = time() - t0
    print("Done")
    
    decode_throughput_bytes_per_sec = len(decoded_text.encode('utf-8')) / decode_elapsed
    decode_throughput_mb_per_sec = decode_throughput_bytes_per_sec / (1024 ** 2)
    
    print(f"  Decoding time:       {decode_elapsed:.3f} seconds")
    print(f"  Decode throughput:   {decode_throughput_bytes_per_sec:,.0f} bytes/sec")
    print(f"                       {decode_throughput_mb_per_sec:.2f} MB/sec")
    print(f"{'-'*70}")
    
    return {
        'name': name,
        'text_size_bytes': text_size_bytes,
        'num_tokens': num_tokens,
        'encode_time': elapsed,
        'throughput_bytes_per_sec': throughput_bytes_per_sec,
        'throughput_mb_per_sec': throughput_mb_per_sec,
        'tokens_per_sec': tokens_per_sec,
        'bytes_per_token': bytes_per_token,
        'decode_time': decode_elapsed,
        'decode_throughput_bytes_per_sec': decode_throughput_bytes_per_sec,
        'decode_throughput_mb_per_sec': decode_throughput_mb_per_sec,
    }


def main():
    """Main benchmarking function."""
    print("\n" + "="*70)
    print("TOKENIZER THROUGHPUT BENCHMARK")
    print("="*70)
    
    # Configuration
    # Use validation sets or limit the number of bytes from train sets
    # to keep benchmark time reasonable
    benchmark_configs = [
        {
            'tokenizer_path': 'saves/ts_first.pkl',
            'data_file': 'data/TinyStories-valid.txt',
            'name': 'TinyStories Tokenizer (validation set)',
            'num_bytes': None,  # Use entire validation set
        },
        {
            'tokenizer_path': 'saves/owt_first.pkl',
            'data_file': 'data/owt_valid.txt',
            'name': 'OWT Tokenizer (validation set)',
            'num_bytes': None,  # Use entire validation set
        },
    ]
    
    # If validation sets don't exist or are too small, use samples from train sets
    fallback_configs = [
        {
            'tokenizer_path': 'saves/ts_first.pkl',
            'data_file': 'data/TinyStories-train.txt',
            'name': 'TinyStories Tokenizer (10MB sample)',
            'num_bytes': 10 * 1024 * 1024,  # 10 MB
        },
        {
            'tokenizer_path': 'saves/owt_first.pkl',
            'data_file': 'data/owt_train.txt',
            'name': 'OWT Tokenizer (10MB sample)',
            'num_bytes': 10 * 1024 * 1024,  # 10 MB
        },
    ]
    
    results = []
    
    for config in benchmark_configs:
        if not os.path.exists(config['data_file']):
            print(f"\nWarning: {config['data_file']} not found, skipping...")
            continue
        
        # Check if file is too small
        file_size = os.path.getsize(config['data_file'])
        if file_size < 1024:  # Less than 1KB
            print(f"\nWarning: {config['data_file']} is too small ({file_size} bytes), skipping...")
            continue
        
        # Load tokenizer
        print(f"\nLoading tokenizer from {config['tokenizer_path']}...")
        tokenizer = load_tokenizer_from_pickle(
            config['tokenizer_path'],
            special_tokens=['<|endoftext|>']
        )
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")
        
        # Run benchmark
        result = benchmark_tokenizer(
            tokenizer=tokenizer,
            text_file=config['data_file'],
            name=config['name'],
            num_bytes=config.get('num_bytes'),
            num_warmup_runs=1
        )
        results.append(result)
    
    # If no benchmarks were run, try fallback configs
    if not results:
        print("\n" + "="*70)
        print("No validation sets found. Using training set samples...")
        print("="*70)
        
        for config in fallback_configs:
            if not os.path.exists(config['data_file']):
                print(f"\nWarning: {config['data_file']} not found, skipping...")
                continue
            
            # Load tokenizer
            print(f"\nLoading tokenizer from {config['tokenizer_path']}...")
            tokenizer = load_tokenizer_from_pickle(
                config['tokenizer_path'],
                special_tokens=['<|endoftext|>']
            )
            print(f"Tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")
            
            # Run benchmark
            result = benchmark_tokenizer(
                tokenizer=tokenizer,
                text_file=config['data_file'],
                name=config['name'],
                num_bytes=config.get('num_bytes'),
                num_warmup_runs=1
            )
            results.append(result)
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n{'Tokenizer':<45} {'Throughput (bytes/s)':<20} {'MB/s':<10}")
        print("-"*70)
        for result in results:
            print(f"{result['name']:<45} {result['throughput_bytes_per_sec']:>18,.0f}  {result['throughput_mb_per_sec']:>8.2f}")
        print("-"*70)
        
        # Find fastest
        if len(results) > 1:
            fastest = max(results, key=lambda x: x['throughput_bytes_per_sec'])
            print(f"\nFastest: {fastest['name']}")
            print(f"         {fastest['throughput_bytes_per_sec']:,.0f} bytes/sec ({fastest['throughput_mb_per_sec']:.2f} MB/sec)")
    else:
        print("\n" + "="*70)
        print("No benchmarks were run. Please check that data files exist.")
        print("="*70)


if __name__ == "__main__":
    main()

