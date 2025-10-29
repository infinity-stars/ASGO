"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
import requests
from omegaconf import OmegaConf
import pdb
import tiktoken
from tqdm import tqdm
import random
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist


def load_train_test_loaders(cfg):
    if 'shakespeare' in cfg.dataset.dataset_name:
        if not os.path.exists(os.path.join(cfg.dataset.dataset_path, cfg.dataset.dataset_name, 'train.bin')):
            prepare_shakespeare_dataset(cfg.dataset.dataset_name, cfg.dataset.dataset_path)
        with open(os.path.join(cfg.dataset.dataset_path, cfg.dataset.dataset_name, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        OmegaConf.update(cfg, 'dataset.vocab_size', meta['vocab_size'])
    elif cfg.dataset.dataset_name == 'openwebtext':
        if not os.path.exists(os.path.join(cfg.dataset.dataset_path, cfg.dataset.dataset_name, 'val.bin')):
            prepare_openwedtext_dataset(cfg.dataset.dataset_name, cfg.dataset.dataset_path)
    else:
        raise ValueError(f"Unexpected dataset_name: {cfg.dataset.dataset_name}")
    if cfg.train.DDP:
        dist.barrier()
    Train_Dataset = LanguageModelDataset(cfg.dataset.dataset_name, cfg.dataset.dataset_path, 
                                        'train', cfg.model.block_size, seed = cfg.train.seed)
    Test_Dataset = LanguageModelDataset(cfg.dataset.dataset_name, cfg.dataset.dataset_path, 'val', 
                                        cfg.model.block_size, seed =  cfg.train.seed)
    train_loader = DataLoader(Train_Dataset, batch_size = cfg.train.batch_size, num_workers = 4)
    
    test_loader = DataLoader(Test_Dataset, batch_size=cfg.train.test_batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    if cfg.train.DDP:
        dist.barrier()

    return train_loader, test_loader

class LanguageModelDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, data_path, split, block_size, seed = None):
        super().__init__()
        self.block_size = block_size
        data_file = 'train.bin' if split == 'train' else 'val.bin'
        self.data_path = os.path.join(data_path, dataset_name, data_file)
        self.seed = seed
        
    def __iter__(self):
        # Multi-CPU setting
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        data = np.memmap(self.data_path, dtype = np.uint16, mode = 'r')
        total_workers = world_size * num_workers
        worker_global_id = rank * num_workers + worker_id

        step = 0
        while True:
            seed = self.seed + worker_global_id + step * total_workers
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(data) - self.block_size)
            x = torch.from_numpy(data[idx:idx+self.block_size].astype(np.int64))
            y = torch.from_numpy(data[idx+1:idx+1+self.block_size].astype(np.int64))
            step += 1
            yield x, y
    
def prepare_openwedtext_dataset(dataset_name, data_path):
    print('==> Preparing OpenWebText dataset...')
    file_path = os.path.join(data_path, dataset_name)
    os.makedirs(os.path.join(data_path, dataset_name), exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("openwebtext", num_proc = 8, trust_remote_code=True)
    split_dataset = dataset['train'].train_test_split(test_size = 0.0005, seed = 2357, shuffle = True)
    split_dataset['val'] = split_dataset.pop('test')
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out
    tokenized = split_dataset.map(
        process,
        remove_columns = ['text'],
        desc = "tokenizing the splits",
        num_proc = 8,
    )
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype = np.uint64)
        filename = os.path.join(file_path, f'{split}.bin')
        arr = np.memmap(filename, dtype = np.uint16, mode = 'w+', shape = (arr_len, ))
        idx = 0
        for batch_idx in tqdm(range(1024), desc = f'writing {filename}'):
            batch = dset.shard(num_shards=1024, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

def prepare_c4_dataset(dataset_name, data_path):
    print('==> Preparing C4 dataset...')
    file_path = os.path.join(data_path, dataset_name)
    os.makedirs(os.path.join(data_path, dataset_name), exist_ok=True)
    tiktoken.get


def prepare_shakespeare_dataset(dataset_name, data_path):
    print('==> Preparing Shakespeare dataset...')

    input_file_path = os.path.join(data_path, dataset_name, 'input.txt')
    os.makedirs(os.path.join(data_path, dataset_name), exist_ok=True)
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    # Read Data
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f'length of dataset in characters:{len(data):,}')

    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    # Here {vocab_size:,} :, means the format 1000 --> 1,000
    print(f'vocab size: {vocab_size: ,}')

    # Create Mappings
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


    # Create train and validation splits
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n*0.9):]

    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f'Train data has: {len(train_ids): ,} tokens')
    print(f'Validation data has: {len(val_ids): ,} tokens')
    if dataset_name == 'shakespeare_char':
        train_ids = np.array(train_ids, dtype =np.uint16)
        val_ids = np.array(val_ids, dtype =np.uint16)
        train_ids.tofile(os.path.join(data_path, 'shakespeare_char', 'train.bin'))
        val_ids.tofile(os.path.join(data_path, 'shakespeare_char', 'val.bin'))

        # Save meta information
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(os.path.join(data_path,'shakespeare_char','meta.pkl'), 'wb') as f:
            # This is save meta into binary data
            pickle.dump(meta, f)
    elif dataset_name == 'shakespeare':
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(data_path, 'shakespeare', 'train.bin'))
        val_ids.tofile(os.path.join(data_path, 'shakespeare', 'val.bin'))
        meta = {
            'vocab_size': enc.n_vocab,
            'encoder': enc  
        }
        with open(os.path.join(data_path,'shakespeare','meta.pkl'), 'wb') as f:
            # This is save meta into binary data
            pickle.dump(meta, f)
    else:
        raise ValueError(f"Unexpected dataset_name: {dataset_name}")

    return
# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

