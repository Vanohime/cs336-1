"""
Script to tokenize TinyStories and OWT datasets using trained tokenizers.
Saves the tokenized data as numpy arrays for efficient loading during training.
"""
import numpy as np
import os
from time import time
from tqdm import tqdm
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


def tokenize_file(tokenizer: Tokenizer, input_path: str, output_path: str, chunk_size: int = 8192, temp_file_size_mb: int = 100):
    """
    Tokenize a text file and save as numpy array.
    Saves to temporary files every temp_file_size_mb MB to avoid memory issues.
    
    Args:
        tokenizer: Trained tokenizer
        input_path: Path to input text file
        output_path: Path to save tokenized numpy array
        chunk_size: Size of chunks to read at a time (in characters)
        temp_file_size_mb: Save temporary file every N MB of source data
    """
    print(f"\nTokenizing {input_path}...")
    print(f"Output will be saved to {output_path}")
    
    t0 = time()
    
    # Get file size for progress bar
    file_size = os.path.getsize(input_path)
    print(f"File size: {file_size / (1024**3):.2f} GB")
    print(f"Saving temp files every {temp_file_size_mb} MB")
    
    # Create temp directory
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_tokenize')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Tokenize using encode_iterable for memory efficiency
    current_tokens = []
    temp_files = []
    bytes_processed = 0
    total_tokens_processed = 0  # Track total tokens across all batches
    temp_file_size_bytes = temp_file_size_mb * 1024 * 1024
    temp_file_counter = 0
    
    # Tokenize entire file with progress tracking
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing") as pbar:
        with open(input_path, 'r', encoding='utf-8') as f:
            for token_id in tokenizer.encode_iterable(f):
                current_tokens.append(token_id)
                total_tokens_processed += 1
                
                # Update progress periodically (every 8192 tokens)
                if total_tokens_processed % 8192 == 0:
                    # Simple heuristic: assume ~1.8 bytes per token on average
                    estimated_bytes = total_tokens_processed * 1.8
                    if estimated_bytes > bytes_processed:
                        pbar.update(int(estimated_bytes - bytes_processed))
                        bytes_processed = estimated_bytes
                
                # Save to temp file if we've accumulated enough tokens
                if len(current_tokens) >= (temp_file_size_bytes // 2):  # ~2 bytes per token
                    temp_file_path = os.path.join(temp_dir, f'temp_{temp_file_counter:04d}.npy')
                    token_array = np.array(current_tokens, dtype=np.uint16)
                    np.save(temp_file_path, token_array)
                    temp_files.append(temp_file_path)
                    
                    tqdm.write(f"  Saved temp file {temp_file_counter}: {len(current_tokens):,} tokens ({token_array.nbytes / (1024**2):.2f} MB)")
                    
                    # Reset for next batch
                    current_tokens = []
                    temp_file_counter += 1
            
            # Ensure progress bar reaches 100%
            pbar.update(file_size - bytes_processed)
    
    # Save any remaining tokens
    if current_tokens:
        temp_file_path = os.path.join(temp_dir, f'temp_{temp_file_counter:04d}.npy')
        token_array = np.array(current_tokens, dtype=np.uint16)
        np.save(temp_file_path, token_array)
        temp_files.append(temp_file_path)
        print(f"  Saved temp file {temp_file_counter}: {len(current_tokens):,} tokens ({token_array.nbytes / (1024**2):.2f} MB)")
        current_tokens = []
    
    # Merge all temp files into final output
    print(f"\nMerging {len(temp_files)} temporary files...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Count total tokens first
    total_tokens = 0
    for temp_file in temp_files:
        temp_array = np.load(temp_file, mmap_mode='r')
        total_tokens += len(temp_array)
    
    print(f"Total tokens: {total_tokens:,}")
    
    # Create memory-mapped output file
    final_array = np.lib.format.open_memmap(
        output_path, 
        mode='w+', 
        dtype=np.uint16, 
        shape=(total_tokens,)
    )
    
    # Copy data from temp files to final file
    offset = 0
    for i, temp_file in enumerate(tqdm(temp_files, desc="Merging files")):
        temp_array = np.load(temp_file)
        final_array[offset:offset + len(temp_array)] = temp_array
        offset += len(temp_array)
    
    # Flush to disk
    del final_array
    
    # Clean up temp files
    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        os.remove(temp_file)
    os.rmdir(temp_dir)
    
    elapsed = time() - t0
    final_size = os.path.getsize(output_path)
    print(f"Array size: {final_size / (1024**2):.2f} MB")
    print(f"Tokenization completed in {elapsed:.2f} seconds ({file_size / elapsed / (1024**2):.2f} MB/s)")
    print(f"Saved to {output_path}")
    
    return total_tokens


def main():
    """Main function to tokenize all datasets."""
    
    # Define paths
    datasets = [
        {
            'name': 'TinyStories',
            'tokenizer_path': 'saves/ts_first.pkl',
            'train_input': 'data/TinyStories-train.txt',
            'train_output': 'data/tokenized/tinystories_train_tokens.npy',
            'valid_input': 'data/TinyStories-valid.txt',
            'valid_output': 'data/tokenized/tinystories_valid_tokens.npy',
        },
        {
            'name': 'OWT (OpenWebText)',
            'tokenizer_path': 'saves/owt_first.pkl',
            'train_input': 'data/owt_train.txt',
            'train_output': 'data/tokenized/owt_train_tokens.npy',
            'valid_input': 'data/owt_valid.txt',
            'valid_output': 'data/tokenized/owt_valid_tokens.npy',
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        print("\n" + "="*80)
        print(f"Processing {dataset['name']} dataset")
        print("="*80)
        
        # Load tokenizer
        print(f"\nLoading tokenizer from {dataset['tokenizer_path']}...")
        tokenizer = load_tokenizer_from_pickle(
            dataset['tokenizer_path'], 
            special_tokens=['<|endoftext|>']
        )
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")
        
        # Tokenize training set
        if os.path.exists(dataset['train_input']):
            tokenize_file(tokenizer, dataset['train_input'], dataset['train_output'])
        else:
            print(f"Warning: {dataset['train_input']} not found, skipping...")
        
        # Tokenize validation set
        if os.path.exists(dataset['valid_input']):
            tokenize_file(tokenizer, dataset['valid_input'], dataset['valid_output'])
        else:
            print(f"Warning: {dataset['valid_input']} not found, skipping...")
    
    print("\n" + "="*80)
    print("All datasets tokenized successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

