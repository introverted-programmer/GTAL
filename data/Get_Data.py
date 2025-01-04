from ctypes.wintypes import SIZE
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import TaggedDocument
import torch.nn as nn
import torch.optim as optim
import random
from models import transformer1
from tests.Data_driven import *




class BasicDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tensor = torch.tensor([self.src_vocab.get(token, self.src_vocab['<UNK>']) for token in self.src_sentences[idx]], dtype=torch.long)
        tgt_tensor = torch.tensor([self.tgt_vocab.get(token, self.tgt_vocab['<UNK>']) for token in self.tgt_sentences[idx]], dtype=torch.long)
        return src_tensor, tgt_tensor
    


class Data_Load():
    def padding(src_vocab, tgt_vocab):
        def collate_fn(batch):
            src_batch, tgt_batch = zip(*batch)
            
            src_lens = [len(seq) for seq in src_batch]
            tgt_lens = [len(seq) for seq in tgt_batch]
            
            max_src_len = max(src_lens)
            max_tgt_len = max(tgt_lens)
            
            src_batch_padded = [torch.cat([seq, torch.tensor([src_vocab['<PAD>']] * (max_src_len - len(seq)), dtype=torch.long)]) for seq in src_batch]
            tgt_batch_padded = [torch.cat([seq, torch.tensor([tgt_vocab['<PAD>']] * (max_tgt_len - len(seq)), dtype=torch.long)]) for seq in tgt_batch]
            
            src_batch_tensor = torch.stack(src_batch_padded)
            tgt_batch_tensor = torch.stack(tgt_batch_padded)

            return src_batch_tensor, tgt_batch_tensor
        return collate_fn
    
    def get_data(src_ss_path, tgt_ss_path, src_vocab_path, tgt_vocab_path, is_training, BATCH_SIZE=128,  SIZE_B=None, SIZE_E=None):
        src_ss, tgt_ss = [], []
        src_vocab, tgt_vocab = {}, {}

        with open(src_ss_path, 'r', encoding='utf-8') as f:
            src_ss = [line.strip()[:].split() for line in f]
        with open(tgt_ss_path, 'r', encoding='utf-8') as f:
            tgt_ss = [['<BOS>'] + line.strip()[:].split() + ['<EOS>'] for line in f]
        
        with open(src_vocab_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                word = line.strip()
                src_vocab[word] = idx
        with open(tgt_vocab_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                word = line.strip()
                tgt_vocab[word] = idx

        if SIZE_B!=None or SIZE_E!=None:
            src_ss = src_ss[SIZE_B:SIZE_E]
            tgt_ss = tgt_ss[SIZE_B:SIZE_E]

        src_max_length = max(len(seq) for seq in src_ss)
        tgt_max_length = max(len(seq) for seq in tgt_ss)

        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)
        src_vocab['<BOS>'] = src_vocab_size
        src_vocab['<PAD>'] = src_vocab_size+1
        src_vocab['<EOS>'] = src_vocab_size+2
        src_vocab['<UNK>'] = src_vocab_size+3
        tgt_vocab['<BOS>'] = tgt_vocab_size
        tgt_vocab['<PAD>'] = tgt_vocab_size+1
        tgt_vocab['<EOS>'] = tgt_vocab_size+2
        tgt_vocab['<UNK>'] = tgt_vocab_size+3
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)

        src_index_2_word = {value: key for key, value in src_vocab.items()}
        tgt_index_2_word = {value: key for key, value in tgt_vocab.items()}

        train_dataset = BasicDataset(src_ss,tgt_ss,src_vocab,tgt_vocab)
        train_dataloder = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=is_training, collate_fn=Data_Load.padding(src_vocab,tgt_vocab))

        return train_dataset, train_dataloder,  (src_vocab,tgt_vocab), (src_index_2_word, tgt_index_2_word), (src_max_length,tgt_max_length)
    
    def data_driven_select_data(src_ss_path, tgt_ss_path, src_vocab_path, tgt_vocab_path, method, SCALE, is_training, BATCH_SIZE=128,  SIZE_B=None, SIZE_E=None):
        src_ss, tgt_ss = [], []
        src_vocab, tgt_vocab = {}, {}
        filename1 = os.path.basename(src_ss_path)
        filename2 = os.path.basename(tgt_ss_path)

        src_language = filename1.split('.')[-3] 
        tgt_language = filename2.split('.')[-3] 
        with open(src_ss_path, 'r', encoding='utf-8') as f:
            src_sss = [line.strip()[:].split() for line in f]    #[['hello', 'world'], ['this', 'is', 'a', 'test']]
        with open(tgt_ss_path, 'r', encoding='utf-8') as f:
            tgt_sss = [['<BOS>'] + line.strip()[:].split() + ['<EOS>'] for line in f]
        
        with open(src_vocab_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                word = line.strip()
                src_vocab[word] = idx
        with open(tgt_vocab_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                word = line.strip()
                tgt_vocab[word] = idx

        src_ss = src_sss[SIZE_B:SIZE_E]
        tgt_ss = tgt_sss[SIZE_B:SIZE_E]

        src_max_length = max(len(seq) for seq in src_ss)
        tgt_max_length = max(len(seq) for seq in tgt_ss)

        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)
        src_vocab['<BOS>'] = src_vocab_size
        src_vocab['<PAD>'] = src_vocab_size+1
        src_vocab['<EOS>'] = src_vocab_size+2
        src_vocab['<UNK>'] = src_vocab_size+3
        tgt_vocab['<BOS>'] = tgt_vocab_size
        tgt_vocab['<PAD>'] = tgt_vocab_size+1
        tgt_vocab['<EOS>'] = tgt_vocab_size+2
        tgt_vocab['<UNK>'] = tgt_vocab_size+3
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)

        src_index_2_word = {value: key for key, value in src_vocab.items()}
        tgt_index_2_word = {value: key for key, value in tgt_vocab.items()}

        if method != None:
            data_src = [' '.join(line) for line in src_ss]    #['hello world', 'this is a test']

            Doc2Vec_model_path = f'./{src_language}-{tgt_language}-{method.__name__}_doc2vec.bin'
            if os.path.exists(Doc2Vec_model_path):
                Doc2Vec_model = Doc2Vec.load(Doc2Vec_model_path)
            else:
                print("Training Doc2Vec...")
                documents = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(data_src)]
                Doc2Vec_model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4, epochs=60)
                Doc2Vec_model.save(Doc2Vec_model_path)
                Doc2Vec_model = Doc2Vec.load(Doc2Vec_model_path)
                print("Done")
            # 将句子转换为向量
            sentence_vectors = [Doc2Vec_model.infer_vector(sentence.split()) for sentence in data_src]
            # print(sentence_vectors)
            # sentence_vectors = np.array(sentence_vectors)
            print("infer_vector Done")
            
            loader_size = int(SCALE * len(data_src))
            print(f"total {loader_size}, select strat: ")
            
            if method.__name__ == "DAL_select":
                unlabel_sentence_vectors = sentence_vectors[:]
                if SIZE_E==None:
                    SIZE_E = 0
                label_sentence = src_sss[SIZE_E:SIZE_E+1000]
                label_sentence = [' '.join(line) for line in label_sentence]
                label_sentence_vectors = [Doc2Vec_model.infer_vector(sentence.split()) for sentence in label_sentence]
                centers = method((label_sentence_vectors,unlabel_sentence_vectors),loader_size)
            elif method.__name__ == "OURS_AL":
                DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # data_src = [' '.join(line) for line in src_ss]    #['hello world', 'this is a test']
                # Doc2Vec_model_path = f'./{src_language}-{tgt_language}_doc2vec.bin'
                # Doc2Vec_model = Doc2Vec.load(Doc2Vec_model_path)
                # 将句子转换为向量
                # sentence_vectors = [Doc2Vec_model.infer_vector(sentence.split()) for sentence in data_src]
                # print("infer_vector Done")

                sentence_vectors = np.array(sentence_vectors)
                init_indices = core_set(sentence_vectors, 1000)
                print("core-set Done")
                # init_indices = [random.randint(0, 40000) for _ in range(1000)]
                init_src = [src_ss[i] for i in init_indices]
                init_tgt = [tgt_ss[i] for i in init_indices]
                print(len(init_src),len(init_tgt),init_src[:3],init_tgt[:3])
                init_dataset = BasicDataset(init_src,init_tgt,src_vocab,tgt_vocab)
                init_dataloder = DataLoader(init_dataset, batch_size=32, shuffle=False, collate_fn=Data_Load.padding(src_vocab,tgt_vocab))
                
                model = transformer1.Transformer(len(src_vocab), len(tgt_vocab), d_model=512, d_k = 64, d_v = 64, n_heads = 8, n_layers=6, dropout=0.1)
                
                centers = method(model, init_dataloder, sentence_vectors, tgt_vocab, init_indices, loader_size-1000)
                print("len(centers)",len(centers))
            elif method.__name__ == "DDCS_AL":
                DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                sentence_vectors = np.array(sentence_vectors)
                src = src_sss[SIZE_E:SIZE_E+200]
                tgt = tgt_sss[SIZE_E:SIZE_E+200]
                train_dataset = BasicDataset(src,tgt,src_vocab,tgt_vocab)
                train_dataloder = DataLoader(train_dataset, batch_size=len(src), shuffle=is_training, collate_fn=Data_Load.padding(src_vocab,tgt_vocab))
                
                model = transformer1.Transformer(len(src_vocab), len(tgt_vocab), d_model=512, d_k = 64, d_v = 64, n_heads = 8, n_layers=6, dropout=0.1)
                model = model.to(DEVICE)
                # weight_decay=1e-4,
                criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])#ignore_index=tgt_vocab['<PAD>']
                optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=1e-5, eps=1e-9, betas=(0.9,0.98))
                for epoch in range(1, 50):
                    model.train()
                    # total_loss = 0
                    # for enc_inputs, dec_inputs, dec_outputs in loader:
                    for i, (src, tgt) in enumerate(train_dataloder):
                        enc_inputs = src
                        dec_inputs = tgt[:,:-1]
                        dec_outputs = tgt[:,1:]
                        # print(f"enc_inputs_size:{enc_inputs.shape},dec_inputs_size:{dec_inputs.shape},dec_outputs_size:{dec_outputs.shape}")

                        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(DEVICE), dec_inputs.to(DEVICE), dec_outputs.to(DEVICE)
                        # outputs: [batch_size * tgt_max_length, tgt_vocab_size]
                        _, outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                        loss = criterion(outputs, dec_outputs.view(-1))  # dec_outputs.view(-1):[batch_size * tgt_max_length * tgt_vocab_size]
                        # loss = model.compute_loss(outputs, dec_outputs.view(-1), l1_lambda=L1, l2_lambda=L2)
                        print(i,'Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(loss))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                centers = method(model, (train_dataloder,sentence_vectors), None, None, loader_size)
            else:
                sentence_vectors = np.array(sentence_vectors)
                centers = method(sentence_vectors,loader_size)
                with open('./random_lens.txt', 'w', encoding='utf-8') as file:
                    # 写入内容
                    file.write(f"random select total list lens :{len(centers)}")

            sel_src = []
            sel_tgt = []
            sel_src.extend(src_ss[i] for i in centers)
            sel_tgt.extend(tgt_ss[i] for i in centers)
            print("===============================")
            print(sel_src[:5], sel_tgt[:5])
            print("===============================")
            print(sel_src[-5:], sel_tgt[-5:])
            print("===============================")

            # dist_mat = distance_matrix(sentence_vectors, sentence_vectors)
            # dist_to_centers = np.min(dist_mat[:, centers], axis=1)
            # max_distance = np.max(dist_to_centers)
            # print(f"{str(method.__name__)}: Maximum distance to nearest center: {max_distance}")
            
            pca = PCA(n_components=2)   # 使用PCA将数据降维到二维
            reduced_data = pca.fit_transform(sentence_vectors)
            
            all_points = reduced_data   # 获取所有点和选中点的二维坐标
            selected_points = reduced_data[centers]

            # 绘制图像
            plt.figure(figsize=(10, 10))
            plt.scatter(all_points[:, 0], all_points[:, 1], c='blue', label='All points')
            plt.scatter(selected_points[:, 0], selected_points[:, 1], c='red', label='Selected points', marker='x')
            plt.legend()
            plt.title(f"{str(method.__name__)} x {len(src_ss)} select {loader_size}")
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.savefig(f'./{str(method.__name__)}.png')

            train_dataset = BasicDataset(sel_src,sel_tgt,src_vocab,tgt_vocab)
            train_dataloder = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=is_training, collate_fn=Data_Load.padding(src_vocab,tgt_vocab))
            return train_dataloder, (src_vocab,tgt_vocab,src_index_2_word, tgt_index_2_word), centers, (src_ss, tgt_ss)
        
        train_dataset = BasicDataset(src_ss,tgt_ss,src_vocab,tgt_vocab)
        train_dataloder = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=is_training, collate_fn=Data_Load.padding(src_vocab,tgt_vocab))

        return train_dataloder, (src_vocab,tgt_vocab,src_index_2_word, tgt_index_2_word), None, (src_ss, tgt_ss)
    
# start = time.perf_counter()
# elapsed = time.perf_counter()  - start
# print(elapsed)
