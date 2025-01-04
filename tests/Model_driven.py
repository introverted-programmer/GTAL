import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from models import transformer1
from data.Get_Data import Data_Load
import math
from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from tests.Data_driven import *
import faiss

# 定义一个简单的Transformer模型用于文本分类
# class Discriminative_model(nn.Module):
#     def __init__(self, input_dim, num_labels):
#         super(Discriminative_model, self).__init__()
#         self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
#         self.classifier = nn.Linear(input_dim, num_labels)

#     def forward(self, x):
#         x = self.transformer_layer(x)
#         x = torch.mean(x, dim=1)  # Pooling
#         return self.classifier(x)

# def train_discriminative_model(X_labeled, X_unlabeled, input_dim, num_labels, device):
#     classifier = LogisticRegression()
#     for _ in range(100):
#         # 准备向量
#         labeled_vectors, unlabeled_vectors = prepare_data(labeled_data, unlabeled_data, doc2vec_model)

#         # 为已标注数据和未标注数据设置标签
#         labels = np.append(np.ones(len(labeled_vectors)), np.zeros(len(unlabeled_vectors)))

#         # 合并数据进行打乱
#         all_vectors = np.vstack((labeled_vectors, unlabeled_vectors))
#         all_vectors, labels = shuffle(all_vectors, labels, random_state=0)

#         # 训练分类器
#         classifier.fit(all_vectors, labels)

#         # 选择数据
#         selected_indices = select_data(classifier, unlabeled_vectors, n_samples)
#         labeled_data += [unlabeled_data[i] for i in selected_indices]
#         unlabeled_data = [doc for i, doc in enumerate(unlabeled_data) if i not in selected_indices]


#     X_combined = torch.cat([X_labeled, X_unlabeled], dim=0)
#     y_combined = torch.cat([torch.zeros(X_labeled.size(0), dtype=torch.long),
#                             torch.ones(X_unlabeled.size(0), dtype=torch.long)], dim=0)

#     dataset = TensorDataset(X_combined, y_combined)
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     model.train()
#     for epoch in range(10):
#         for X_batch, y_batch in loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#     return model

# def get_unlabeled_idx(X_train, labeled_idx):
#     all_idx = np.arange(len(X_train))
#     return np.setdiff1d(all_idx, labeled_idx)


# class DiscriminativeSampling:
#     def __init__(self, model, input_dim, num_labels, device='cpu'):
#         self.model = model
#         self.input_dim = input_dim
#         self.num_labels = num_labels
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.sub_batches = 10

#     def query(self, X_train, Y_train, labeled_idx, amount):
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         # labeled_idx = np.array(labeled_idx)
#         unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

#         labeled_so_far = 0
#         sub_sample_size = int(amount / self.sub_batches)
#         while labeled_so_far < amount:
#             if labeled_so_far + sub_sample_size > amount:
#                 sub_sample_size = amount - labeled_so_far

#             discriminative_model = train_discriminative_model(X_train[labeled_idx], X_train[unlabeled_idx], self.input_dim, self.num_labels, self.device)
#             predictions = discriminative_model(X_train[unlabeled_idx]).softmax(dim=1)
#             _, selected_indices = torch.topk(predictions[:, 1], sub_sample_size)
#             labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices.numpy()]))
#             labeled_so_far += sub_sample_size
#             unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

#             del discriminative_model
#             torch.cuda.empty_cache()

#         return labeled_idx


def build_faiss_index(data_vectors):
    dimension = data_vectors.shape[1]  # 向量维度
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离来构建索引
    index.add(data_vectors)  # 向索引中添加数据
    return index


def find_similar_documents(query_vectors, index, k=5):
    """ 查询索引以找到最相似的k个文档 """
    distances, indices = index.search(query_vectors, k)
    return distances, indices

def calculate_entropy(logits):
    batch_size = logits.size(0) // 37
    logits = logits.view(batch_size, 37, -1)
    probabilities = F.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities + 1e-10)  # 避免对0取对数
    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    entropy = torch.sum(entropy, dim=1)  # 将所有时间步的熵求和
    return entropy


def DDCS_AL(model, data,vocab, data_vectors, k):   #动态判别式核心集   主动学习方法
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lab_data = data[0]
    unlab_vectors = data[1]
    # src_vocab = vocab[0]
    # tgt_vocab = vocab[1]
    data_iter = iter(lab_data)
    src, tgt = next(data_iter)
    enc_inputs = src
    dec_inputs = tgt[:,:-1]
    dec_outputs = tgt[:,1:]
    enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(DEVICE), dec_inputs.to(DEVICE), dec_outputs.to(DEVICE)
    # label_sentence_vectors = data_vectors[0]
    # unlabel_sentence_vectors= data_vectors[1]
    # label_labels = np.zeros(len(label_sentence_vectors))
    # unlabel_labels = np.ones(len(unlabel_sentence_vectors))
    # all_sentence_vectors = np.array(label_sentence_vectors+unlabel_sentence_vectors)
    logits, _, _, _ = model(enc_inputs, dec_inputs)
    print(logits.shape)
    entropy = calculate_entropy(logits)
    print(entropy.shape)
    _, top_uncertain_indices = torch.topk(entropy, 150)  # 获取不确定性最高的n个样本的索引
    # top_uncertain_indices = torch.tensor(top_uncertain_indices, dtype=torch.long)
    top_uncertain_indices = top_uncertain_indices[:]
    top_uncertain_indices = top_uncertain_indices.cpu().numpy()
    a = enc_inputs[:]
    a = a.cpu().numpy()
    print("top_uncertain_indices", top_uncertain_indices)
    print(a.shape,a[0],top_uncertain_indices.shape)
    uncertain_datas = a[top_uncertain_indices]
    model_path = '/data/lyf/Transformer/data/T_doc2vec.bin'
    model = Doc2Vec.load(model_path)  


    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    SIZE_B = 0
    SIZE_E = 20000
    LR = 1e-5
    L1 = 1e-7
    L2 = 1e-7

    VAL_SCALE=0.2
    src_ss_path = "/data/lyf/Transformer/data/train.en-bn.en.tok.bpe"
    tgt_ss_path = "/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe"
    src_vocab_path = "/data/lyf/Transformer/data/vocab.en.txt"
    tgt_vocab_path = "/data/lyf/Transformer/data/vocab.bn.txt"

    sel_sc = 0.4


    train_dataloader, vocab, centers, ori_data = Data_Load.data_driven_select_data(src_ss_path, tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=sel_sc, is_training=False, BATCH_SIZE=BATCH_SIZE, SIZE_B=SIZE_B, SIZE_E=math.floor(SIZE_E*0.05))
    src_vocab, tgt_vocab, src_index_2_word, tgt_index_2_word  = vocab[0], vocab[1], vocab[2], vocab[3]
    print("uncertain_datas", uncertain_datas, uncertain_datas.shape)
    uncertain_sens = [' '.join([src_index_2_word[token] for token in sen]) for sen in uncertain_datas]
    uncertain_vectors = [model.infer_vector(sentence.split()) for sentence in uncertain_sens]
    # centers = DAL_select((uncertain_vectors, unlab_vectors), k)   # DAL结合模型不确定性

    # 模型不确定性结合多查询最近邻搜索
    uncertain_vectors = np.array(uncertain_vectors)
    unlab_vectors = np.array(unlab_vectors)

    index = build_faiss_index(unlab_vectors)
    print(index)
    n = k // len(uncertain_vectors)  # 想要查询的相似文档的数量
    similar_docs = {}
    for i, query_vector in enumerate(uncertain_vectors):
        query_vector = query_vector.reshape(1, -1)  # 转换成二维数组以匹配FAISS的API
        distances, indices = find_similar_documents(query_vector, index, k=n)
        print(indices)
        similar_docs[i] = indices[0]  # 保存索引

    # 输出查询结果
    for query_idx, similar_indices in similar_docs.items():
        print(f"Document {query_idx} in dataset1 is most similar to documents {similar_indices} in dataset2")

    # 使用集合来去除重复的索引
    unique_indices = set()
    for indices in similar_docs.values():
        unique_indices.update(indices)

    unique_indices_list = list(unique_indices)
    print("Unique indices of similar documents:", len(unique_indices_list))



    return unique_indices_list

# 数据选择函数
def select_data(classifier, unlabeled_vectors, exclude_indices, n_samples):
    probs = classifier.predict_proba(unlabeled_vectors)[:, 0]  # 类别0的概率
    mask = np.ones(len(unlabeled_vectors), dtype=bool)  # 创建一个布尔掩码，初始化时所有值都为True
    mask[exclude_indices] = False    # 将排除的索引设置为False
    filtered_probs = probs[mask]    # 仅在未被排除的索引中进行选择
    filtered_indices = np.argsort(filtered_probs)[:n_samples]
    # 将相对索引转换为原始数组的索引
    # 注意: np.nonzero返回的是元组，我们需要第一个元素
    original_indices = np.nonzero(mask)[0][filtered_indices]

    return original_indices

# if __name__ == "__main__":
#     DEVICE = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
#     NUM_EPOCHS = 2
#     BATCH_SIZE = 32
#     SIZE_B = 0
#     SIZE_E = 20000
#     LR = 1e-5
#     L1 = 1e-7
#     L2 = 1e-7

#     VAL_SCALE=0.2

#     src_ss_path = "/data/lyf/Transformer/data/train.en-bn.en.tok.bpe"
#     tgt_ss_path = "/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe"
#     src_vocab_path = "/data/lyf/Transformer/data/vocab.en.txt"
#     tgt_vocab_path = "/data/lyf/Transformer/data/vocab.bn.txt"

#     sel_sc = 0.4


#     train_dataloader, vocab, centers, ori_data = Data_Load.data_driven_select_data(src_ss_path, tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=sel_sc, is_training=False, BATCH_SIZE=BATCH_SIZE, SIZE_B=SIZE_B, SIZE_E=math.floor(SIZE_E*0.05))
#     src_vocab, tgt_vocab, src_index_2_word, tgt_index_2_word  = vocab[0], vocab[1], vocab[2], vocab[3]
#     train_sel_dataloader, vocab, centers, ori_sel_data = Data_Load.data_driven_select_data(src_ss_path, tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=sel_sc, is_training=False, BATCH_SIZE=BATCH_SIZE, SIZE_B=math.floor(SIZE_E*0.05), SIZE_E=math.floor(SIZE_E*0.05)+20000)

#     valid_dataloader, _, _, _ = Data_Load.data_driven_select_data(src_ss_path, tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=None, is_training=True, BATCH_SIZE=BATCH_SIZE, SIZE_B=math.floor(-VAL_SCALE*(SIZE_E-SIZE_B)*sel_sc*0.05), SIZE_E=None)
#     model = transformer1.Transformer(len(src_vocab), len(tgt_vocab), d_model=512, d_k = 64, d_v = 64, n_heads = 8, n_layers=6, dropout=0.1)
#     model = model.to(DEVICE)
    
#     labeled_data = [[src_vocab.get(token, src_vocab['<UNK>']) for token in sentence] for sentence in ori_data[0]]
#     labeled_labels = [[tgt_vocab.get(token, tgt_vocab['<UNK>']) for token in sentence] for sentence in ori_data[1]]
#     # labeled_data, labeled_labels = ori_data[0], ori_data[1] #1k
#     unlabeled_data = [[src_vocab.get(token, src_vocab['<UNK>']) for token in sentence] for sentence in ori_sel_data[0]]    #20k
#     unlabeled_labels = [[tgt_vocab.get(token, tgt_vocab['<UNK>']) for token in sentence] for sentence in ori_sel_data[1]]
#     src_lens = [len(seq) for seq in labeled_data]
#     tgt_lens = [len(seq) for seq in labeled_labels]
#     und_lens = [len(seq) for seq in unlabeled_data]
#     und_lab_lens = [len(seq) for seq in unlabeled_labels]
    
#     max_src_len = max(src_lens)
#     max_tgt_len = max(tgt_lens)
#     max_und_len = max(und_lens)
#     max_und_lab_len = max(und_lab_lens)
#     max_len = max(max_src_len,max_tgt_len,max_und_len,max_und_lab_len)
#     print(max_len)

#     # print(torch.cat([torch.tensor(labeled_data[0], dtype=torch.long),torch.tensor([0]*5, dtype=torch.long)]))
#     labeled_data = [ seq+[src_vocab['<PAD>']] * (max_len - len(seq))  for seq in labeled_data]
#     # print(labeled_data)
#     labeled_labels = [ seq+[tgt_vocab['<PAD>']] * (max_len - len(seq))  for seq in labeled_labels]
#     unlabeled_data = [ seq+[src_vocab['<PAD>']] * (max_len - len(seq))  for seq in unlabeled_data]
#     unlabeled_labels = [ seq+[tgt_vocab['<PAD>']] * (max_len - len(seq))  for seq in unlabeled_labels]
#     # labeled_labels = [torch.cat([torch.tensor(seq, dtype=torch.long), torch.tensor([tgt_vocab['<PAD>']] * (max_tgt_len - len(seq)), dtype=torch.long)]) for seq in labeled_labels]
#     # unlabeled_data = [torch.cat([torch.tensor(seq, dtype=torch.long), torch.tensor([src_vocab['<PAD>']] * (max_und_len - len(seq)), dtype=torch.long)]) for seq in unlabeled_data]

#     label_datas = ori_data[0]
#     unlabel_datas = ori_sel_data[0]

#     label_datas_sens = [' '.join(line) for line in label_datas]
#     unlabel_datas_sens = [' '.join(line) for line in unlabel_datas]

#     label_labels = np.zeros(len(label_datas_sens))
#     unlabel_labels = np.ones(len(unlabel_datas_sens))
#     all_sens = label_datas_sens+unlabel_datas_sens
#     all_sens = np.array(all_sens)
#     model_path = '/data/lyf/Transformer/tests/doc2vec.bin'
#     # documents = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(data)]
#     # model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4, epochs=60)
#     # model.save(model_path)
#     model = Doc2Vec.load(model_path)
#     # 将句子转换为向量
#     label_sentence_vectors = [model.infer_vector(sentence.split()) for sentence in label_datas_sens]
#     unlabel_sentence_vectors = [model.infer_vector(sentence.split()) for sentence in unlabel_datas_sens]
#     all_sentence_vectors = label_sentence_vectors+unlabel_sentence_vectors
#     # all_sentence_vectors = [model.infer_vector(sentence.split()) for sentence in all_sens]
#     all_sentence_vectors = np.array(all_sentence_vectors)
#     print("infer_vector Done")

#     classifier = LogisticRegression()
#     s = []
#     X_train = np.array(all_sentence_vectors[:])#np.concatenate([label_datas_sens,unlabel_datas_sens])
#     Y_train = np.concatenate([label_labels,unlabel_labels], axis=0)  # 如果您有监督数据的标签
       
#     for _ in tqdm(range(100)):
#         labeled_idx = np.where(Y_train == 0)[0] # 已标记数据的初始索引

#         indices = np.arange(len(X_train))
#         np.random.shuffle(indices)
#         # print(indices)
#         X_train = X_train[indices]
#         Y_train = Y_train[indices]
#         all_sens = all_sens[indices]

#         # 训练分类器
#         classifier.fit(X_train, Y_train)
#         # print("train classifier Done")

#         # 选择数据
#         selected_indices = select_data(classifier, unlabel_sentence_vectors, s,  80)
#         s.append(selected_indices)
#         selected_vectors =np.array(unlabel_sentence_vectors)[selected_indices]

#         # X_train = np.concatenate([X_train, selected_vectors], axis=0)#[unlabel_sentence_vectors[i] for i in selected_indices]
#         # Y_train = np.concatenate(Y_train,np.ones(len(selected_indices)))
#         # unlabel_sentence_vectors = [doc for i, doc in enumerate(unlabel_sentence_vectors) if i not in selected_indices]
        
#         # Y_train[np.array([np.where(np.allclose(X_train, target)) for target in selected_vectors])]=0
#         result_indices = [np.where([np.array_equal(x, target) for x in X_train])[0] for target in selected_vectors]
#         Y_train[result_indices]=0

#     s = np.concatenate(s)
#     print(s)
    
#     print(len(set(s)), len(s))
#     pca = PCA(n_components=2)   # 使用PCA将数据降维到二维
#     reduced_data = pca.fit_transform(unlabel_sentence_vectors)
    
#     all_points = reduced_data   # 获取所有点和选中点的二维坐标
#     selected_points = reduced_data[s]

#     # 绘制图像
#     plt.figure(figsize=(10, 10))
#     plt.scatter(all_points[:, 0], all_points[:, 1], c='blue', label='All points')
#     plt.scatter(selected_points[:, 0], selected_points[:, 1], c='red', label='Selected points', marker='x')
#     plt.legend()
#     plt.title(f"DAL")
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.savefig(f'./DAL.png')

#     # discriminative_sampling = DiscriminativeSampling(model, input_dim, num_labels,device=DEVICE)
#     # c = discriminative_sampling.query(X_train,Y_train,labeled_idx,amount=10)