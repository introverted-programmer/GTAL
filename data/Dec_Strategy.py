import torch
import torch.nn.functional as F

# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

def greedy_decoder(model, enc_input, tgt_vocab):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    device = enc_input.device
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)  # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
    terminal = False
    start_symbol=tgt_vocab['<BOS>']
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
                              -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        next_word = prob.data[-1]  # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_symbol = next_word
        if next_symbol == tgt_vocab['<EOS>']:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict

def beam_search_decoder(model, enc_input, tgt_vocab, beam_size=4, max_len=50):
    """
    Beam Search Decoder
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol
    :param beam_size: The beam width
    :param max_len: The maximum length of the generated sequence
    :param tgt_vocab: The target vocabulary
    :return: The target sequence
    """
    
    device = enc_input.device

    # 编码器处理输入
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    start_symbol=tgt_vocab['<BOS>']
    # 初始化束
    sequences = [[(start_symbol, 0.0)]]
    
    for _ in range(max_len):
        all_candidates = []
        for seq in sequences:
            dec_input = torch.tensor([[token for token, score in seq]], dtype=enc_input.dtype).to(device)
            # 解码器生成输出
            dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
            projected = model.projection(dec_outputs)
            log_probs = F.log_softmax(projected, dim=-1)
            
            # 取最后一个时间步的 log 概率
            log_probs = log_probs[:, -1, :]
            
            # 获取 beam_size 个最高概率的下一个 token
            topk = torch.topk(log_probs, beam_size)
            
            for i in range(beam_size):
                token = topk.indices[0][i].item()
                score = topk.values[0][i].item()
                candidate = seq + [(token, seq[-1][1] + score)]
                all_candidates.append(candidate)
        
        # 按照得分排序，并保留前 beam_size 个序列
        ordered = sorted(all_candidates, key=lambda x: x[-1][1], reverse=True)
        sequences = ordered[:beam_size]

        # 检查是否达到结束符
        if any(seq[-1][0] == tgt_vocab['<EOS>'] for seq in sequences):
            break

    # 选择得分最高的序列
    best_sequence = sequences[0]
    result_sequence = [token for token, score in best_sequence if token != tgt_vocab['<EOS>']]
    
    return torch.tensor(result_sequence[1:], dtype=enc_input.dtype).unsqueeze(0)

def batch_beam_search_decoder(model, enc_inputs, tgt_vocab, beam_size=4, max_len=50):
    """
    Beam Search Decoder with batch support
    :param model: Transformer Model
    :param enc_inputs: The encoder input (batch)
    :param tgt_vocab: The target vocabulary
    :param beam_size: The beam width
    :param max_len: The maximum length of the generated sequence
    :return: The target sequences (batch)
    """
    
    device = enc_inputs.device
    batch_size = enc_inputs.size(0)

    # 编码器处理输入
    enc_outputs, enc_self_attns = model.encoder(enc_inputs)
    start_symbol = tgt_vocab['<BOS>']

    # 初始化束，每个输入都维护 beam_size 个序列
    sequences = [[[(start_symbol, 0.0)] for _ in range(beam_size)] for _ in range(batch_size)]
    
    for _ in range(max_len):
        all_candidates = [[] for _ in range(batch_size)]
        
        for b in range(batch_size):
            for seq in sequences[b]:
                dec_input = torch.tensor([[token for token, score in seq]], dtype=enc_inputs.dtype).to(device)
                # 解码器生成输出
                dec_outputs, _, _ = model.decoder(dec_input, enc_inputs[b:b+1], enc_outputs[b:b+1])
                projected = model.projection(dec_outputs)
                log_probs = F.log_softmax(projected, dim=-1)
                
                # 取最后一个时间步的 log 概率
                log_probs = log_probs[:, -1, :]
                
                # 获取 beam_size 个最高概率的下一个 token
                topk = torch.topk(log_probs, beam_size)
                
                for i in range(beam_size):
                    token = topk.indices[0][i].item()
                    score = topk.values[0][i].item()
                    candidate = seq + [(token, seq[-1][1] + score)]
                    all_candidates[b].append(candidate)
        
        # 按照得分排序，并保留前 beam_size 个序列
        for b in range(batch_size):
            ordered = sorted(all_candidates[b], key=lambda x: x[-1][1], reverse=True)
            sequences[b] = ordered[:beam_size]

        # 检查是否达到结束符
        if all(any(seq[-1][0] == tgt_vocab['<EOS>'] for seq in sequences[b]) for b in range(batch_size)):
            break

    # 选择得分最高的序列
    results = []
    for b in range(batch_size):
        best_sequence = sequences[b][0]
        result_sequence = [token for token, score in best_sequence if token != tgt_vocab['<EOS>']]
        results.append(result_sequence[1:])  # 排除起始符

    return [torch.tensor(result, dtype=enc_inputs.dtype).unsqueeze(0) for result in results]
