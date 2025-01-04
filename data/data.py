import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from gensim.models import Word2Vec

# 加载语料
with open('Transformer\data\\train.en-bn.en.tok.bpe', 'r', encoding='utf-8') as f:
    sentences = [line.strip().split() for line in f]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)
model_path = 'Transformer\data\\srcWord2Vec.bin'
model.save(model_path)

print(model.wv.most_similar('you', topn=1))

# vectors = []
# for word in word_list:
#     if word in model.wv:
#         vectors.append(model.wv[word])
#     else:
#         vectors.append([0] * 100)

# # 自定义数据集类
# class CustomDataset(Dataset):
#     def __init__(self, src_file, tgt_file, max_samples=None):
#         # 读取源语言和目标语言的数据文件
#         with open(src_file, 'r', encoding='utf-8') as f:
#             self.src_data = f.readlines()
#         with open(tgt_file, 'r', encoding='utf-8') as f:
#             self.tgt_data = f.readlines()
        
#         # 截取前 max_samples 条数据
#         if max_samples is not None:
#             self.src_data = self.src_data[:max_samples]
#             self.tgt_data = self.tgt_data[:max_samples]
    
#     def __len__(self):
#         return len(self.src_data)
    
#     def __getitem__(self, index):
#         # 获取对应索引位置的源语言和目标语言数据，并返回
#         src_sent = self.src_data[index].strip()
#         tgt_sent = self.tgt_data[index].strip()
#         return {'src': src_sent, 'tgt': tgt_sent}

# # 源语言和目标语言数据文件路径
# src_file = 'Transformer\data\\train.en-bn.bn.tok.bpe'
# tgt_file = 'Transformer\data\\train.en-bn.bn.tok.bpe'

# n = 200  # 假设你想要截取前 100 条数据
# # 创建自定义数据集对象
# dataset = CustomDataset(src_file, tgt_file, max_samples=n)

# # 创建 dataload 格式的数据加载器
# batch_size = 32
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # 遍历数据加载器，获取批次数据
# for batch in data_loader:
#     src_batch = batch['src']  # 获取源语言批次数据
#     tgt_batch = batch['tgt']  # 获取目标语言批次数据
#     # print(src_batch)
#     # 在这里进行训练代码的编写，使用 src_batch 和 tgt_batch 进行训练
