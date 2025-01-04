import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import transformer1
from data.Get_Data import Data_Load
import numpy as np
import time
from tests.Data_driven import *
from tests.Model_driven import *
from tests.mod_driven import *
import os

def get_region(language_code):
    if language_code in ["ur", "bn", "ne", "ta", "te", "kn", "ps", "or"]:
        # ur(乌尔都语) bn(孟加拉语) ne(尼泊尔语) ta(泰米尔语) te(泰卢固语)
        # kn(卡纳达语) ps(普什图语) or(奥里亚语)
        return "south_asia"
    elif language_code in ["az", "he", "fa", "tk", "ar", "ku", "am"]:
        # az(阿塞拜疆) he(希伯来语) fa(波斯语) tk(土库曼语) ar(阿拉伯语)
        # ku(库尔德语) am(阿姆哈拉语)
        return "middle_east"
    elif language_code in ["km", "ph", "id", "ms", "vi", "th", "fil", "my", "lo"]:
        # km(高棉语(柬埔寨)) ph(菲律宾) id(印度尼西亚) ms(马来西亚)
        # vi(越南) th(泰语) fil(菲律宾) my(缅甸) lo(老挝)
        return "southeast_asia"
    else:
        return "UNKNOWN"

# DEVICE = transformer1.DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 150
BATCH_SIZE = 64
SIZE_B = 0
SIZE_E = 20000
LR = 1e-5
L1 = 1e-7
L2 = 1e-7

VAL_SCALE=0.2
src_lang = "en"
tgt_lang = "ar" #  fa ms th ur ne az
source_of = "TED2020" # CCAligned wmt19 TED2020
METHOD = OURS_AL   # random_select    core_set    DAL_select  OURS_AL

region = get_region(tgt_lang)

root_path = f"/home/user/lyf/data/dataset/{source_of}/{region}/{src_lang}-{tgt_lang}"

train_src_ss_path = root_path + f"/train/4w_train.{src_lang}-{tgt_lang}.{src_lang}.tok.bpe"    # "/data/lyf/dataset/CCAligned/middle_east/en-az/train/4w_train.en-az.en.tok.bpe"#"/data/lyf/Transformer/data/train.en-bn.en.tok.bpe"
train_tgt_ss_path = root_path + f"/train/4w_train.{src_lang}-{tgt_lang}.{tgt_lang}.tok.bpe"    # "/data/lyf/dataset/CCAligned/middle_east/en-az/train/4w_train.en-az.az.tok.bpe"#"/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe"
vaild_src_ss_path = root_path + f"/dev/3k_dev.{src_lang}-{tgt_lang}.{src_lang}.tok.bpe"
vaild_tgt_ss_path = root_path + f"/dev/3k_dev.{src_lang}-{tgt_lang}.{tgt_lang}.tok.bpe"
src_vocab_path = root_path + f"/vocab/vocab.{src_lang}.txt"
tgt_vocab_path = root_path + f"/vocab/vocab.{tgt_lang}.txt"


# train_src_ss_path = "/data/lyf/Transformer/data/train.en-bn.en.tok.bpe"
# train_tgt_ss_path = "/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe"
# src_vocab_path = "/data/lyf/Transformer/data/vocab.en.txt"
# tgt_vocab_path = "/data/lyf/Transformer/data/vocab.bn.txt"
filename1 = os.path.basename(train_src_ss_path)
filename2 = os.path.basename(train_tgt_ss_path)
src_language = filename1.split('.')[-3] 
tgt_language = filename2.split('.')[-3] 
src_tgt = f"{src_language}-{tgt_language}"


sel_sc = 0.4
model_path = f"./t1.1.2_{src_tgt}_{METHOD.__name__}_model.pt"        ##!!!
plt_path = f"./t1.1.2_{src_tgt}_{METHOD.__name__}_training_loss.png" ##!!!

train_dataloader, vocab, centers, ori_data = Data_Load.data_driven_select_data(train_src_ss_path, train_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=METHOD, SCALE=sel_sc, is_training=True, BATCH_SIZE=BATCH_SIZE)#, SIZE_B=SIZE_B, SIZE_E=SIZE_E
valid_dataloader, _, _, _ = Data_Load.data_driven_select_data(vaild_src_ss_path, vaild_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=None, is_training=True, BATCH_SIZE=BATCH_SIZE, SIZE_B=math.floor(-VAL_SCALE*(SIZE_E-SIZE_B)*sel_sc), SIZE_E=None)

# train_dataloader, vocab, centers, ori_data = Data_Load.data_driven_select_data(train_src_ss_path, train_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=METHOD, SCALE=sel_sc, is_training=True, BATCH_SIZE=BATCH_SIZE)
# valid_dataloader, _, _, _ = Data_Load.data_driven_select_data(vaild_src_ss_path, vaild_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=None, is_training=True, BATCH_SIZE=BATCH_SIZE)
src_vocab, tgt_vocab, src_index_2_word, tgt_index_2_word  = vocab[0], vocab[1], vocab[2], vocab[3]

model = transformer1.Transformer(len(src_vocab), len(tgt_vocab), d_model=512, d_k = 64, d_v = 64, n_heads = 8, n_layers=6, dropout=0.1)
model = model.to(DEVICE)
# weight_decay=1e-4,
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])#ignore_index=tgt_vocab['<PAD>']
optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=LR, eps=1e-9, betas=(0.9,0.98))
# optimizer = optim.SGD(model.parameters(), lr=7e-4, momentum=0.99)  # 用adam的话效果不好
# early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
time1 = time.perf_counter()

start_epoch = 0
# if os.path.exists(model_path):
#     checkpoint = torch.load(model_path, map_location=DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1

train_losses = []
valid_losses = []
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_loss = 0
    # for enc_inputs, dec_inputs, dec_outputs in loader:
    for i, (src, tgt) in enumerate(train_dataloader):
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

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    # avg_loss = VAL_SCALE * total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    model.eval()
    total_loss = 0
    valid_loss = 0
    with torch.no_grad():
        for src, tgt in valid_dataloader:
            valid_enc_inputs = src
            valid_dec_inputs = tgt[:,:-1]
            valid_dec_outputs = tgt[:,1:]
            valid_enc_inputs, valid_dec_inputs, valid_dec_outputs = valid_enc_inputs.to(DEVICE), valid_dec_inputs.to(DEVICE), valid_dec_outputs.to(DEVICE)
            _, valid_outputs, valid_enc_self_attns, valid_dec_self_attns, valid_dec_enc_attns = model(valid_enc_inputs, valid_dec_inputs)
            valid_loss = criterion(valid_outputs, valid_dec_outputs.view(-1))
            # valid_loss = model.compute_loss(valid_outputs, valid_dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'valid_loss =', '{:.6f}'.format(valid_loss))
            total_loss += valid_loss.item()
        
        avg_loss = total_loss / len(valid_dataloader)
        valid_losses.append(avg_loss)
        
        print('Epoch:', '%04d' % (epoch + 1), 'valid_loss =', '{:.6f}'.format(avg_loss))
        
        plt.figure()  # 设置图像大小
        plt.plot(range(start_epoch+1, len(train_losses) + 1), train_losses, 'b',label='Training Loss')  # 绘制损失曲线
        plt.plot(range(start_epoch+1, len(valid_losses) + 1), valid_losses, 'r',label='Validation Loss')  # 绘制损失曲线
        plt.title(f'{type(model).__name__} and {METHOD.__name__}')
        plt.xlabel('epoch')  # 设置x轴标签
        plt.ylabel('Loss')  # 设置y轴标签
        plt.legend()  # 显示图例
        plt.savefig(plt_path)  # 将图像保存为文件


        # early_stopping(avg_loss)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, model_path)

time2 = time.perf_counter() - time1
print("train total time: ", time2/60.0, "min")




# import math
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from models import transformer1
# from data.Get_Data import Data_Load
# import numpy as np
# import time
# from tests.Data_driven import *
# from tests.Model_driven import *
# from tests.mod_driven import *
# import os

# def get_region(language_code):
#     if language_code in ["ur", "bn", "ne", "ta", "te", "kn", "ps", "or"]:
#         # ur(乌尔都语) bn(孟加拉语) ne(尼泊尔语) ta(泰米尔语) te(泰卢固语)
#         # kn(卡纳达语) ps(普什图语) or(奥里亚语)
#         return "south_asia"
#     elif language_code in ["az", "he", "fa", "tk", "ar", "ku", "am"]:
#         # az(阿塞拜疆) he(希伯来语) fa(波斯语) tk(土库曼语) ar(阿拉伯语)
#         # ku(库尔德语) am(阿姆哈拉语)
#         return "middle_east"
#     elif language_code in ["km", "ph", "id", "ms", "vi", "th", "fil", "my", "lo"]:
#         # km(高棉语(柬埔寨)) ph(菲律宾) id(印度尼西亚) ms(马来西亚)
#         # vi(越南) th(泰语) fil(菲律宾) my(缅甸) lo(老挝)
#         return "southeast_asia"
#     else:
#         return "UNKNOWN"

# # DEVICE = transformer1.DEVICE
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# NUM_EPOCHS = 50
# BATCH_SIZE = 64
# SIZE_B = 0
# SIZE_E = 20000
# LR = 1e-5
# L1 = 1e-7
# L2 = 1e-7

# VAL_SCALE=0.2
# src_lang = "en"
# tgt_lang = "az" #  fa ms th ur ne az
# source_of = "CCAligned" # CCAligned wmt19
# METHOD = None   # random_select    core_set    DAL_select  OURS_AL
# region = get_region(tgt_lang)

# root_path = f"/home/user/lyf/data/dataset/{source_of}/{region}/{src_lang}-{tgt_lang}"

# train_src_ss_path = root_path + f"/train/train.{src_lang}-{tgt_lang}.{src_lang}.tok.bpe"    # "/data/lyf/dataset/CCAligned/middle_east/en-az/train/4w_train.en-az.en.tok.bpe"#"/data/lyf/Transformer/data/train.en-bn.en.tok.bpe"
# train_tgt_ss_path = root_path + f"/train/train.{src_lang}-{tgt_lang}.{tgt_lang}.tok.bpe"    # "/data/lyf/dataset/CCAligned/middle_east/en-az/train/4w_train.en-az.az.tok.bpe"#"/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe"
# vaild_src_ss_path = root_path + f"/dev/dev.{src_lang}-{tgt_lang}.{src_lang}.tok.bpe"
# vaild_tgt_ss_path = root_path + f"/dev/dev.{src_lang}-{tgt_lang}.{tgt_lang}.tok.bpe"
# src_vocab_path = root_path + f"/vocab/vocab.{src_lang}.txt"
# tgt_vocab_path = root_path + f"/vocab/vocab.{tgt_lang}.txt"


# # train_src_ss_path = "/data/lyf/Transformer/data/train.en-bn.en.tok.bpe"
# # train_tgt_ss_path = "/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe"
# # src_vocab_path = "/data/lyf/Transformer/data/vocab.en.txt"
# # tgt_vocab_path = "/data/lyf/Transformer/data/vocab.bn.txt"
# filename1 = os.path.basename(train_src_ss_path)
# filename2 = os.path.basename(train_tgt_ss_path)
# src_language = filename1.split('.')[-3] 
# tgt_language = filename2.split('.')[-3] 
# src_tgt = f"{src_language}-{tgt_language}"


# sel_sc = 1 #0.25
# model_path = f"./t1.1.2_{src_tgt}_all_None_model.pt"        ##!!!     METHOD.__name__
# plt_path = f"./t1.1.2_{src_tgt}_all_None_training_loss.png" ##!!!     METHOD.__name__

# train_dataloader, vocab, centers, ori_data = Data_Load.data_driven_select_data(train_src_ss_path, train_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=METHOD, SCALE=sel_sc, is_training=True, BATCH_SIZE=BATCH_SIZE)#, SIZE_B=SIZE_B, SIZE_E=SIZE_E
# valid_dataloader, _, _, _ = Data_Load.data_driven_select_data(vaild_src_ss_path, vaild_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=None, is_training=True, BATCH_SIZE=BATCH_SIZE, SIZE_B=math.floor(-VAL_SCALE*(SIZE_E-SIZE_B)*sel_sc), SIZE_E=None)

# # train_dataloader, vocab, centers, ori_data = Data_Load.data_driven_select_data(train_src_ss_path, train_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=METHOD, SCALE=sel_sc, is_training=True, BATCH_SIZE=BATCH_SIZE)
# # valid_dataloader, _, _, _ = Data_Load.data_driven_select_data(vaild_src_ss_path, vaild_tgt_ss_path, src_vocab_path, tgt_vocab_path, method=None, SCALE=None, is_training=True, BATCH_SIZE=BATCH_SIZE)
# src_vocab, tgt_vocab, src_index_2_word, tgt_index_2_word  = vocab[0], vocab[1], vocab[2], vocab[3]

# model = transformer1.Transformer(len(src_vocab), len(tgt_vocab), d_model=512, d_k = 64, d_v = 64, n_heads = 8, n_layers=6, dropout=0.1)
# model = model.to(DEVICE)
# # weight_decay=1e-4,
# criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])#ignore_index=tgt_vocab['<PAD>']
# optimizer = optim.Adam(model.parameters(), weight_decay=1e-4, lr=LR, eps=1e-9, betas=(0.9,0.98))
# # optimizer = optim.SGD(model.parameters(), lr=7e-4, momentum=0.99)  # 用adam的话效果不好
# # early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
# time1 = time.perf_counter()

# start_epoch = 0
# if os.path.exists(model_path):
#     checkpoint = torch.load(model_path, map_location=DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1

# train_losses = []
# valid_losses = []
# for epoch in range(start_epoch, NUM_EPOCHS):
#     model.train()
#     total_loss = 0
#     # for enc_inputs, dec_inputs, dec_outputs in loader:
#     for i, (src, tgt) in enumerate(train_dataloader):
#         enc_inputs = src
#         dec_inputs = tgt[:,:-1]
#         dec_outputs = tgt[:,1:]
#         # print(f"enc_inputs_size:{enc_inputs.shape},dec_inputs_size:{dec_inputs.shape},dec_outputs_size:{dec_outputs.shape}")

#         enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(DEVICE), dec_inputs.to(DEVICE), dec_outputs.to(DEVICE)
#         # outputs: [batch_size * tgt_max_length, tgt_vocab_size]
#         _, outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
#         loss = criterion(outputs, dec_outputs.view(-1))  # dec_outputs.view(-1):[batch_size * tgt_max_length * tgt_vocab_size]
#         # loss = model.compute_loss(outputs, dec_outputs.view(-1), l1_lambda=L1, l2_lambda=L2)
#         print(i,'Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(loss))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_dataloader)
#     # avg_loss = VAL_SCALE * total_loss / len(train_dataloader)
#     train_losses.append(avg_loss)

#     model.eval()
#     total_loss = 0
#     valid_loss = 0
#     with torch.no_grad():
#         for src, tgt in valid_dataloader:
#             valid_enc_inputs = src
#             valid_dec_inputs = tgt[:,:-1]
#             valid_dec_outputs = tgt[:,1:]
#             valid_enc_inputs, valid_dec_inputs, valid_dec_outputs = valid_enc_inputs.to(DEVICE), valid_dec_inputs.to(DEVICE), valid_dec_outputs.to(DEVICE)
#             _, valid_outputs, valid_enc_self_attns, valid_dec_self_attns, valid_dec_enc_attns = model(valid_enc_inputs, valid_dec_inputs)
#             valid_loss = criterion(valid_outputs, valid_dec_outputs.view(-1))
#             # valid_loss = model.compute_loss(valid_outputs, valid_dec_outputs.view(-1))
#             print('Epoch:', '%04d' % (epoch + 1), 'valid_loss =', '{:.6f}'.format(valid_loss))
#             total_loss += valid_loss.item()
        
#         avg_loss = total_loss / len(valid_dataloader)
#         valid_losses.append(avg_loss)
        
#         print('Epoch:', '%04d' % (epoch + 1), 'valid_loss =', '{:.6f}'.format(avg_loss))
        
#         # plt.figure()  # 设置图像大小
#         # plt.plot(range(start_epoch+1, len(train_losses) + 1), train_losses, 'b',label='Training Loss')  # 绘制损失曲线
#         # plt.plot(range(start_epoch+1, len(valid_losses) + 1), valid_losses, 'r',label='Validation Loss')  # 绘制损失曲线
#         # # plt.title(f'{type(model).__name__} and {METHOD.__name__}')
#         # plt.xlabel('epoch')  # 设置x轴标签
#         # plt.ylabel('Loss')  # 设置y轴标签
#         # plt.legend()  # 显示图例
#         # plt.savefig(plt_path)  # 将图像保存为文件


#         # early_stopping(avg_loss)
#         # if early_stopping.early_stop:
#         #     print("Early stopping")
#         #     break

#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss.item(),
#     }, model_path)

# time2 = time.perf_counter() - time1
# print("train total time: ", time2/60.0, "min")

