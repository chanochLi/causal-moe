import torch
import tiktoken
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, path: str, block_size: int=512):
        super().__init__()
        
        # gpt官方tokenizer
        self.enc = tiktoken.get_encoding('gpt2')
        self.block_size = block_size    # pos最大长度

        # 特殊分割符号来分割不同训练文本
        # gpt2使用<|endoftext|>来分割 # [50256]
        self.encoded_data = []
        self.eos_toekn = self.enc.encode(
            '<|endoftext|>', 
            allowed_special={'<|endoftext|>'}
            )[0]
        
        # 获取训练数据
        import json
        raw_data = []
        self.max_line = 1000
        with open(path, 'r') as f:
            for index, line in enumerate(f):
                if index >= self.max_line:
                    break
                try:
                    # 由于数据不等长，需要pad数据，然后存成numpy等等形式，而非同时处理数据
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except Exception as e:
                    continue
        
        full_encoded = []
        for text in raw_data:
            # 把所有文本拼接，然后用特殊符号分割
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_toekn])
        
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i: i + self.block_size + 1] # 在此处+1，而不在计算loss的时候shift，每一行实际是513
            if len(chunk) != self.block_size + 1:
                chunk = chunk + [self.eos_toekn] * (self.block_size + 1 - len(chunk))

            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, index):
        chunk = self.encoded_data[index]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
        
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, tokens):
        return self.enc.decode(tokens)