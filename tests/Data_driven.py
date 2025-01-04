# from gurobipy import *
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import torch
from sklearn.linear_model import LogisticRegression

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def gpu_distance_matrix(X, Y):
#     # 将数据转移到 GPU
#     X_gpu = cp.asarray(X)
#     Y_gpu = cp.asarray(Y)
    
#     # 计算每对点之间的差
#     diff = X_gpu[:, cp.newaxis, :] - Y_gpu[cp.newaxis, :, :]
    
#     # 计算欧氏距离，使用 linalg.norm
#     dist_mat = cp.linalg.norm(diff, axis=-1)
    
#     return dist_mat

# 随机选择方法
def random_select(data_vectors,k):
    n = data_vectors.shape[0]
    idx = np.random.choice(n, k, replace=False)
    lis = idx.tolist()
    return lis

# core-set 求解 k-center 问题
def core_set(data_vectors, k):
    print(f"Calculating minkowski distance")
    dist_mat = distance_matrix(data_vectors, data_vectors)
    print(f"Done")
    dist_mat_gpu = torch.tensor(dist_mat).to(DEVICE)
    
    num_points = dist_mat_gpu.shape[0]
    centers = []
    centers.append(torch.randint(0, num_points, (1,)).item())
    
    for _ in tqdm(range(1, k)):
        dist_to_centers = torch.min(dist_mat_gpu[:, centers], axis=1).values
        new_center = torch.argmax(dist_to_centers).item()
        centers.append(new_center)
    # print(len(centers), centers[:5])
    return centers

#最大误差最小化求解
def maximal_error_minimization(data_vectors, k):
    n = data_vectors.shape[0]
    idx = np.random.choice(n, 1, replace=False)  # Start with one random point
    centers_data = data_vectors[idx]
    centers = [idx[0]]
    for _ in tqdm(range(k - 1)):
        dist = cdist(data_vectors, centers_data, 'euclidean')
        max_dist_idx = np.argmax(np.min(dist, axis=1))
        centers_data = np.vstack([centers_data, data_vectors[max_dist_idx]])
        centers.append(max_dist_idx)
    return centers

# 数据选择函数
def select_data(classifier, unlabeled_vectors, exclude_indices, n_samples):
    probs = classifier.predict_proba(unlabeled_vectors)[:, 0]  # 类别0的概率
    mask = np.ones(len(unlabeled_vectors), dtype=bool)  # 创建一个布尔掩码，初始化时所有值都为True
    mask[exclude_indices] = False    # 将排除的索引设置为False
    filtered_probs = probs[mask]    # 仅在未被排除的索引中进行选择
    filtered_indices = np.argsort(filtered_probs)[:n_samples]
    # filtered_indices = np.argsort(filtered_probs)[-n_samples:]
    # 将相对索引转换为原始数组的索引
    # 注意: np.nonzero返回的是元组，我们需要第一个元素
    original_indices = np.nonzero(mask)[0][filtered_indices]

    return original_indices

def DAL_select(data_vectors, k):
    label_sentence_vectors = data_vectors[0]
    unlabel_sentence_vectors= data_vectors[1]
    label_labels = np.zeros(len(label_sentence_vectors))
    unlabel_labels = np.ones(len(unlabel_sentence_vectors))
    all_sentence_vectors = np.array(label_sentence_vectors+unlabel_sentence_vectors)

    print("infer_vector Done")

    classifier = LogisticRegression()
    s = []
    X_train = np.array(all_sentence_vectors[:])#np.concatenate([label_datas_sens,unlabel_datas_sens])
    Y_train = np.concatenate([label_labels,unlabel_labels], axis=0)  # 如果您有监督数据的标签
       
    for _ in range(100):
        labeled_idx = np.where(Y_train == 0)[0] # 已标记数据的初始索引

        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        X_train = X_train[indices]
        Y_train = Y_train[indices]
        # all_sens = all_sens[indices]

        # 训练分类器
        classifier.fit(X_train, Y_train)
        print("train classifier Done")
        # print("len(s)",len(s),s)
        selected_indices = select_data(classifier, unlabel_sentence_vectors, s,  int(k/100))
        # print(type(selected_indices),type(s))
        s = np.concatenate((np.array(s), selected_indices)).astype(int).tolist()
        # s.append(selected_indices)
        selected_vectors =np.array(unlabel_sentence_vectors)[selected_indices]

        Y_train[np.array([np.where(np.allclose(X_train, target)) for target in selected_vectors])]=0

    s = np.array(s)
    return s