import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from tests.Data_driven import *
import faiss
import gc


# 定义线性选择函数
def linear_selection_function(i, N, M):
    return np.ceil((M - i) * N / (M * (M + 1) / 2))

# 定义指数选择函数
def exponential_selection_function(i, N, M):
    weights = np.exp(-np.arange(M) / M)
    weight_sum = np.sum(weights)
    return np.ceil(np.exp(-i / M) * N / weight_sum)

def select_elements_with_target_sum(arr, target_sum):
    total = 0  # 初始化总和
    selected_elements = []  # 存储满足条件的前 m 个元素
    
    for i, num in enumerate(arr):
        # 如果当前总和加上 num 会超过目标和
        if total + num > target_sum:
            # 调整当前元素，使总和等于目标和
            selected_elements.append(target_sum - total)
            break
        else:
            # 否则将当前元素加入结果中
            selected_elements.append(num)
            total += num
            
        # 如果总和等于目标和，提前结束
        if total == target_sum:
            break
            
    return selected_elements

def build_faiss_index(data_vectors):
    dimension = data_vectors.shape[1]  # 向量维度
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离来构建索引
    index.add(data_vectors)  # 向索引中添加数据
    return index


def find_similar_documents(query_vectors, index, excluded_indices, k=5):
    # 假设初始搜索量为k的两倍，保留空间排除一些索引
    num_search = k * 2
    # print("query_vectors, num_search",type(query_vectors), num_search)
    distances, indices = index.search(query_vectors, num_search)
    
    # 初始化结果存储
    final_indices = []
    final_distances = []

    # 过滤掉已经存在于 excluded_indices 中的索引
    for dist, idx in zip(distances.flatten(), indices.flatten()):
        if idx not in excluded_indices:
            final_indices.append(idx)
            final_distances.append(dist)
            if len(final_indices) == k:
                break
    
    # 如果初始搜索结果不足以提供k个有效结果，则需要继续扩展搜索范围
    while len(final_indices) < k:
        num_search *= 2  # 搜索范围加倍
        distances, indices = index.search(query_vectors, num_search)
        for dist, idx in zip(distances.flatten(), indices.flatten()):
            if idx not in excluded_indices and idx not in final_indices:
                final_indices.append(idx)
                final_distances.append(dist)
                if len(final_indices) == k:
                    break
    
    # 返回足够数量的最近邻距离和索引
    return np.array(final_distances[:k]), np.array(final_indices[:k])

def select_d2d(data0, data1, exist_index, sel_size_list):
    excluded_indices = exist_index[:]
    del exist_index
    print("data0.shape, data1.shape",data0.shape, data1.shape)
    index = build_faiss_index(data1)
    print(index)
    # n = k // len(data0)  # 想要查询的相似文档的数量
    similar_docs = {}
    for i, (query_vector, n) in enumerate(zip(data0, sel_size_list)):
        n = int(n)
        # if n == 0:
        #     n = 1
        print("n: ",n)
        query_vector = query_vector.reshape(1, -1)  # 转换成二维数组以匹配FAISS的API
        distances, indices = find_similar_documents(query_vector, index, excluded_indices, k=n)
        print("indices:",indices)
        similar_docs[i] = indices  # 保存索引
        excluded_indices.extend(indices)
        excluded_indices = list(set(excluded_indices))

    # 输出查询结果
    for query_idx, similar_indices in similar_docs.items():
        print(f"Document {query_idx} in dataset1 is most similar to documents {similar_indices} in dataset2")

    print("Unique indices of similar documents len:", len(excluded_indices), excluded_indices[0],excluded_indices[1])
    with open('./indices.txt', 'w', encoding='utf-8') as file:
        # 写入内容
        file.write(f"Unique indices of similar documents len:{len(excluded_indices), excluded_indices[0],excluded_indices[1]}" )
    return excluded_indices

def _calculate_entropy(logits, seq_len):
    print("logits.shape:",logits.shape)
    # batch_size = logits.size(0) // seq_len
    # logits = logits.view(batch_size, seq_len, -1)
    probabilities = F.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities + 1e-10)  # 避免对0取对数
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    entropy = torch.sum(entropy, dim=1)  # 将所有时间步的熵求和
    return entropy

def calculate_entropy(padded_logits, batch_size=32):
    num_batches = (len(padded_logits) + batch_size - 1) // batch_size
    total_entropy = []  # 用来收集所有批次的熵

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(padded_logits))
        batch_logits = padded_logits[start_idx:end_idx]
        max_size = max(logit.shape[0] for logit in batch_logits) 
        batch_logits = [F.pad(logit, (0, 0, 0, max_size - logit.shape[0])) if logit.shape[0] < max_size else logit for logit in batch_logits]
        # batch_logits = torch.cat(batch_logits, dim=0)
        # print(len(batch_logits),batch_logits[0].shape)
        batch_logits = torch.stack(batch_logits, dim=0)
        mask = batch_logits != 0    # 创建掩码，假设填充值为0
        probabilities = F.softmax(batch_logits, dim=-1) # 计算概率
        masked_probabilities = probabilities * mask + (~mask).float()   # 应用掩码到概率
        log_probabilities = torch.log(masked_probabilities + 1e-10) # 计算对数概率

        entropy = -torch.sum(masked_probabilities * log_probabilities, dim=-1)  # 计算熵
        entropy = torch.sum(entropy, dim=1)  # 将所有时间步的熵求和

        # 收集这批次的熵
        total_entropy.append(entropy)

    # 将所有批次的熵合并
    total_entropy = torch.cat(total_entropy, dim=0).to("cuda:0")
    return total_entropy

def train_model(model, dataloader, tgt_vocab, NUM_EPOCHS):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    # DEVICE = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])#ignore_index=tgt_vocab['<PAD>']
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-5, eps=1e-9, betas=(0.9,0.98))
    for epoch in range(0, NUM_EPOCHS):
        model.train()
        all_logits = []

        for i, (src, tgt) in enumerate(dataloader):
            enc_inputs = src
            dec_inputs = tgt[:,:-1]
            dec_outputs = tgt[:,1:]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(DEVICE), dec_inputs.to(DEVICE), dec_outputs.to(DEVICE)
            # print("enc_inputs, dec_inputs shape",enc_inputs.shape, dec_inputs.shape,enc_inputs[0], dec_inputs[0])
            logits, outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print(i,'Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logits = logits.detach()
            for logit in logits:
                all_logits.append(logit)
    
    # # 假设 all_logits 是一个包含多个形状不同的张量的列表
    # max_size = max(logit.shape[1] for logit in all_logits)  # 假设所有张量的形状在第1维度上不同

    # # 填充所有张量到相同的最大尺寸
    # padded_logits = [F.pad(logit, (0, 0, 0, max_size - logit.shape[1])) if logit.shape[1] < max_size else logit for logit in all_logits]
    # all_logits = torch.cat(padded_logits, dim=0)

    return model, all_logits

def OURS_AL(model, dataloader, vectors, tgt_vocab, exist_index, k):
    data1_vectors = vectors[:]
    # data1_vectors = vectors[1]
    data0_vectors = vectors[np.array(exist_index)]
    # print("data1_vectors.shape, data0_vectors.shape",data1_vectors.shape, data0_vectors.shape)
    torch.cuda.empty_cache() 
    model, logits = train_model(model, dataloader, tgt_vocab, 50)
    del model
    entropy = calculate_entropy(logits)
    _, sort_uncertain_indices = torch.topk(entropy, len(logits)) # 将样本不确定性的索引排序
    del logits, entropy
    gc.collect()  # 垃圾收集
    torch.cuda.empty_cache() 
    M = len(sort_uncertain_indices)
    sel_size_list = [linear_selection_function(i, k, M) for i in range(M)]
    # print(f"------------------sel_size_list:{sel_size_list}---------------------------")
    sel_size_list = select_elements_with_target_sum(sel_size_list, k)
    print(f"结果列表的总和: {sum(sel_size_list)}")
    lengs = len(sel_size_list)
    selected_indices_list = select_d2d(data0_vectors[:lengs], data1_vectors, exist_index, sel_size_list)
    print(f"selected_indices_list: {sum(selected_indices_list)}")
    return selected_indices_list
