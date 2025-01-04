
import math

with open('/data/lyf/Transformer/data/train.en-bn.en.tok.bpe', 'r', encoding='utf-8') as f:
    src_ss = [line.strip() for line in f]

with open('/data/lyf/Transformer/data/train.en-bn.bn.tok.bpe', 'r', encoding='utf-8') as f:
    tgt_ss = [line.strip() for line in f]

# SIZE_B = 0
# SIZE_E = 20000

# ss_tarin_src = src_ss[SIZE_B:SIZE_E]
# ss_train_tgt = tgt_ss[SIZE_B:SIZE_E]
# ss_val_src = src_ss[math.floor(-0.2*SIZE_E):]
# ss_val_tgt = tgt_ss[math.floor(-0.2*SIZE_E):]

# # 定义输出文件的路径
# tarin_src = './tarin_src.txt'
# train_tgt = './train_tgt.txt'
# val_src = './val_src.txt'
# val_tgt = './val_tgt.txt'

# with open(tarin_src, 'w', encoding='utf-8') as file:
#     for sentence in ss_tarin_src:
#         file.write(sentence + '\n')

# # 将句子写入文件
# with open(train_tgt, 'w', encoding='utf-8') as file:
#     for sentence in ss_train_tgt:
#         file.write(sentence + '\n')


# with open(val_src, 'w', encoding='utf-8') as file:
#     for sentence in ss_val_src:
#         file.write(sentence + '\n')

# # 将句子写入文件
# with open(val_tgt, 'w', encoding='utf-8') as file:
#     for sentence in ss_val_tgt:
#         file.write(sentence + '\n')

# print(f"句子已写入文件")


SIZE_B = 240001
SIZE_E = -48000

ss_test_src = src_ss[SIZE_B:SIZE_E]
ss_test_tgt = tgt_ss[SIZE_B:SIZE_E]
# 定义输出文件的路径
test_src = './all_test_src.txt'
test_tgt = './all_test_tgt.txt'

with open(test_src, 'w', encoding='utf-8') as file:
    for sentence in ss_test_src:
        file.write(sentence + '\n')

# 将句子写入文件
with open(test_tgt, 'w', encoding='utf-8') as file:
    for sentence in ss_test_tgt:
        file.write(sentence + '\n')
