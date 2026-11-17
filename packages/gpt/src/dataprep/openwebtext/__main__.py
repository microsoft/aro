# type: ignore
# MIT License
# Original Author: Andrej Karpathy
# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

DATA_PATH = Path("data") / "openwebtext"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Parquet copy of OpenWebText on the Hugging Face Hub (the legacy "openwebtext" alias and its
# loading script are no longer supported by recent `datasets`/`huggingface_hub` releases).
DATASET_ID = "Skylion007/openwebtext"

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    # Get subset fraction from environment variable or use default
    subset_fraction = float(os.environ.get("SUBSET_FRACTION", "0.29"))

    # downloads ~24GB of parquet files (about 8M documents, 8,013,769) into the Hugging Face
    # cache; the full dataset is downloaded even when only a subset is tokenized below
    dataset = load_dataset(DATASET_ID, num_proc=num_proc_load_dataset)

    if subset_fraction < 1.0:
        print(f"Using subset fraction: {subset_fraction:.3f}")
        # For 2.6B tokens, we need approximately 28-30% of the original dataset
        # Since original has ~9B tokens, we take about 0.29 of the data

        # Take a subset of the dataset first
        dataset_subset = dataset["train"].train_test_split(
            train_size=subset_fraction, seed=2357, shuffle=True
        )["train"]

        # Now create train/val split from this subset
        split_dataset = dataset_subset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

        print("Dataset subset info:")
        print(f"Train documents: {len(split_dataset['train']):,}")
        print(f"Val documents: {len(split_dataset['val']):,}")
        print(
            f"Expected tokens: ~{int(9_000_000_000 * subset_fraction / 1_000_000_000 * 1000)}M tokens"
        )
    else:
        print("Using full dataset")
        # owt by default only contains the 'train' split, so create a test split
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example["text"])  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = DATA_PATH / f"{split}.bin"
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = min(1024, len(dset))  # small subsets have fewer documents than shards

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
