import torch
import torch.nn as nn
import math

# 检查是否有可用的 GPU，如果有就使用它
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # d_model: 词嵌入的维度
        # max_len: 句子的最大长度

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度
        
        # pe 的形状是 (max_len, d_model)
        # 我们增加一个 batch 维度，方便后续计算
        pe = pe.unsqueeze(0) # -> (1, max_len, d_model)
        
        # 将 pe 注册为 buffer。它不是模型的参数，但希望它能随着模型移动（例如 to(device)）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)
        # 我们将位置编码加到输入 x 上
        # self.pe[:, :x.size(1), :] 会切片出需要长度的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model       # 模型的总维度
        self.num_heads = num_heads   # 头数
        self.d_k = d_model // num_heads # 每个头的维度

        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 计算 Q 和 K 的点积
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2. 应用掩码 (mask)
        if mask is not None:
            # mask 为 0 的地方，attn_scores 设为一个非常小的负数，这样 softmax 后会趋近于 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        # 3. 应用 softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 4. 与 V 相乘
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        # 拆分 d_model -> num_heads * d_k
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 输出形状: (batch_size, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        # x 形状: (batch_size, num_heads, seq_len, d_k)
        batch_size, _, seq_len, _ = x.size()
        # 合并 head
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # 输出形状: (batch_size, seq_len, d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. 线性变换
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)
        
        # 2. 拆分多头
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        
        # 3. 缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 合并多头
        output = self.combine_heads(attn_output)
        
        # 5. 最终的线性变换
        output = self.W_o(output)
        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        # d_model: 模型的维度，也是输入和输出的维度
        # d_ff:    前馈网络中间层的维度，通常是 d_model * 4
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, d_model)
        # 流程: x -> 线性变换1 -> ReLU -> 线性变换2 -> output
        output = self.fc2(self.relu(self.fc1(x)))
        # output 的形状: (batch_size, seq_len, d_model)
        return output
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # 子模块1: 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 子模块2: 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # 定义两个层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 定义两个 dropout 层，用于正则化
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x 的形状: (batch_size, seq_len, d_model)
        # mask 用于屏蔽 padding token
        
        # --- 第一个子层: 多头注意力 ---
        # 1. 计算注意力输出
        attn_output = self.self_attn(x, x, x, mask) # Q, K, V 都是 x，所以是自注意力
        # 2. Add & Norm
        # 首先应用 dropout，然后是残差连接 (x + ...)，最后是层归一化
        x = self.norm1(x + self.dropout1(attn_output))
        
        # --- 第二个子层: 前馈网络 ---
        # 1. 计算前馈网络输出
        ff_output = self.feed_forward(x)
        # 2. Add & Norm
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # 子模块1: 带掩码的多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 子模块2: 交叉注意力 (Q from Decoder, K/V from Encoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # 子模块3: 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        # 定义三个层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 定义三个 dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x:              解码器的输入 (目标序列)，形状 (batch_size, tgt_seq_len, d_model)
        # encoder_output: 编码器的最终输出，形状 (batch_size, src_seq_len, d_model)
        # src_mask:       源序列的填充掩码 (padding mask)
        # tgt_mask:       目标序列的未来词元掩码 (look-ahead mask) 和填充掩码的组合
        
        # --- 第一个子层: 带掩码的多头自注意力 ---
        # Q, K, V 都来自解码器自身的输入 x
        # tgt_mask 用于确保我们不能看到未来的 token
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # --- 第二个子层: 交叉注意力 ---
        # Query 来自解码器 (上面的 x)
        # Key 和 Value 来自编码器的输出 (encoder_output)
        # src_mask 用于屏蔽源序列中的 padding token
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # --- 第三个子层: 前馈网络 ---
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # 词嵌入层
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        # N个编码器层
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 1. 词嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model) # 论文中提到要乘以 d_model 的平方根
        x = self.pos_encoder(self.dropout(x))
        
        # 2. 依次通过 N 个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(self.dropout(x))
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x
    
class TransformerBlock(nn.Module):
    """
    这是 Decoder-only Transformer 的核心构建块。
    它其实和我们之前写的 EncoderLayer 结构完全相同。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # 子模块1: 带掩码的多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 子模块2: 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x 的形状: (batch_size, seq_len, d_model)
        # mask: 必须是结合了 padding mask 和 look-ahead mask 的组合掩码
        
        # --- 第一个子层: 带掩码的多头自注意力 ---
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # --- 第二个子层: 前馈网络 ---
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class EDTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super(EDTransformer, self).__init__()
        
        # 实例化编码器和解码器
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        
        # 最后的线性层，将解码器的输出映射到目标词汇表大小
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 定义填充 token 的索引，方便制作掩码
        self.src_pad_idx = 0 # 假设源语言的 <pad> 索引为 0
        self.tgt_pad_idx = 0 # 假设目标语言的 <pad> 索引为 0

    def make_src_mask(self, src):
        # src 的形状: (batch_size, src_len)
        # 创建一个掩码，在 <pad> token 的位置为 0，其他位置为 1
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # 最终形状: (batch_size, 1, 1, src_len)
        # 这是为了方便和多头的 (batch_size, num_heads, seq_len, seq_len) 注意力分数进行广播
        return src_mask.to(device)

    def make_tgt_mask(self, tgt):
        # tgt 的形状: (batch_size, tgt_len)
        
        # 1. 目标语言的 <pad> token 掩码
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        # 形状: (batch_size, 1, 1, tgt_len)

        # 2. 目标语言的 "未来词元" 掩码 (look-ahead mask)
        tgt_len = tgt.shape[1]
        # torch.tril 创建一个下三角矩阵
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()
        # 形状: (tgt_len, tgt_len)
        
        # 合并两种掩码
        # & 操作符会确保一个位置只有在既不是 padding 也不是未来词元时才为 1
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        # 最终形状: (batch_size, 1, tgt_len, tgt_len)
        return tgt_mask

    def forward(self, src, tgt):
        # src: 源序列, 形状 (batch_size, src_len)
        # tgt: 目标序列, 形状 (batch_size, tgt_len)
        
        # 1. 创建掩码
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 2. 通过编码器
        enc_output = self.encoder(src, src_mask)
        
        # 3. 通过解码器
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # 4. 通过最后的线性层得到输出
        output = self.fc_out(dec_output)
        
        return output
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super(DecoderOnlyTransformer, self).__init__()
        
        self.pad_idx = 0 # 假设 padding token 的索引是 0
        self.d_model = d_model  # 关键修复
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # N 个 TransformerBlock 堆叠
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        # 在所有层之后，通常会再加一个 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        
        # 最后的线性输出层，将结果映射回词汇表
        self.fc_out = nn.Linear(d_model, vocab_size)

    def make_mask(self, src):
        # 这个方法和我们之前为 Transformer 的 target 序列创建掩码的方法完全一样
        
        # 1. <pad> token 掩码
        src_pad_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # 形状: (batch_size, 1, 1, src_len)

        # 2. "未来词元" 掩码 (look-ahead mask)
        src_len = src.shape[1]
        src_sub_mask = torch.tril(torch.ones((src_len, src_len), device=device)).bool()
        # 形状: (src_len, src_len)
        
        # 合并两种掩码
        src_mask = src_pad_mask & src_sub_mask
        # 最终形状: (batch_size, 1, src_len, src_len)
        return src_mask

    def forward(self, src):
        # src: 输入序列, 形状 (batch_size, seq_len)
        
        # 1. 创建掩码
        mask = self.make_mask(src)
        
        # 2. 词嵌入和位置编码
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(self.dropout(x))
        
        # 3. 依次通过 N 个 TransformerBlock
        for layer in self.layers:
            x = layer(x, mask)
        
        # 4. 通过最后的层归一化
        x = self.final_norm(x)
            
        # 5. 通过最后的线性层得到输出 (logits)
        output = self.fc_out(x)
        
        return output