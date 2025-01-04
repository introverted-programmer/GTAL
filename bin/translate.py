import time
import torch
from torch import Tensor
from models import transformer1
from data.Get_Data import Data_Load
from data.Dec_Strategy import *
import os
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# with open('/data/lyf/Transformer/data/train.en-bn.en.tok.bpe', 'r', encoding='utf-8') as f:
#     src_ss = [line.strip() for line in f]

# with open('/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe', 'r', encoding='utf-8') as f:
#     tgt_ss = [line.strip() for line in f]

# TEST_SIZE_B = 50000
# TEST_SIZE_E = 51500

# src_test = src_ss[TEST_SIZE_B:TEST_SIZE_E]
# tgt_test = tgt_ss[TEST_SIZE_B:TEST_SIZE_E]

# = './1.5k_test_src.txt'
# = './1.5k_test_tgt.txt'
# 定义输出文件的路径

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


src_lang = "en"
tgt_lang = "ar"
source_of = "TED2020"
method = "OURS_AL"#"random_select"   # DAL_select random_select OURS_AL core_set
region = get_region(tgt_lang)

root_path = f"/home/user/lyf/data/dataset/{source_of}/{region}/{src_lang}-{tgt_lang}"

src_ss_path = root_path + f"/test/3k_test.{src_lang}-{tgt_lang}.{src_lang}.tok.bpe"
tgt_ss_path = root_path + f"/test/3k_test.{src_lang}-{tgt_lang}.{tgt_lang}.tok.bpe"
src_vocab_path = root_path + f"/vocab/vocab.{src_lang}.txt"
tgt_vocab_path = root_path + f"/vocab/vocab.{tgt_lang}.txt"

# src_ss_path ="/data/lyf/dataset/CCAligned/middle_east/en-az/test/3k_test.en-az.en.tok.bpe" #'/data/lyf/Transformer/data/all_test_src.txt'
# tgt_ss_path ="/data/lyf/dataset/CCAligned/middle_east/en-az/test/3k_test.en-az.az.tok.bpe" #'/data/lyf/Transformer/data/all_test_tgt.txt'

# src_vocab_path = "/data/lyf/dataset/CCAligned/middle_east/en-az/vocab/vocab.en.txt"
# tgt_vocab_path = "/data/lyf/dataset/CCAligned/middle_east/en-az/vocab/vocab.az.txt"

filename1 = os.path.basename(src_ss_path)
filename2 = os.path.basename(tgt_ss_path)
src_language = filename1.split('.')[-3] 
tgt_language = filename2.split('.')[-3] 
src_tgt = f"{src_language}-{tgt_language}"
directory_path = os.path.dirname(src_ss_path)

output_trans_path = f"{directory_path}/trans.t1.1.2_{method}"   ##!!!
model_path = f"/home/user/lyf/data/Transformer/bin/t1.1.2_{src_tgt}_{method}_model.pt"        ##!!!


data_set, dataload, vocab, index_2_word, max_length = Data_Load.get_data(src_ss_path, tgt_ss_path, src_vocab_path, tgt_vocab_path, is_training=False, BATCH_SIZE=512)
src_index_2_word, tgt_index_2_word = index_2_word[0],index_2_word[1]
src_vocab, tgt_vocab = vocab[0], vocab[1]

model = transformer1.Transformer(len(src_vocab), len(tgt_vocab), d_model=512, d_k = 64, d_v = 64, n_heads = 8, n_layers=6, dropout=0.1)
model = model.to(DEVICE)
checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

src_tensor = []
trans_ss = []
count = 1
time1 = time.perf_counter()
# print("dataload.len: ", len(dataload))
for j, (src, tgt) in enumerate(dataload):
    # if j>0:
    #     break
    # print(len(src))
    for n in range(len(src)):
        # print(src[n])
        dec_predict = beam_search_decoder(model, src[n].view(1, -1).to(DEVICE), tgt_vocab=tgt_vocab)
        dec_predict = dec_predict.squeeze()
        if dec_predict.dim()==0:
            dec_predict = Tensor([dec_predict])

        ss_list = [src_index_2_word[t.item()] for t in src[n]] 
        # print(' '.join([te for te in ss_list if te!='<PAD>']),'-->')
        se = ' '.join([tgt_index_2_word[i.item() if i.dim() == 0 else i] for i in dec_predict])
        print(count, se)
        count += 1
        trans_ss.append(se)
# [print(s) for s in trans_ss]

with open(output_trans_path, 'w', encoding='utf-8') as file:
    for sentence in trans_ss:
        file.write(sentence + '\n')

time2 = time.perf_counter() - time1
print("translate total time: ", time2/60.0, "min")
print(f"翻译句子已写入{output_trans_path}")