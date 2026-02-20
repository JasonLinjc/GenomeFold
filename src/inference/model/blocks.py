import torch
import torch.nn as nn
import numpy as np
import copy

class ConvBlock(nn.Module):
    def __init__(self, size, stride=2, hidden_in=64, hidden=64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding=pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding=pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, output_size=256, filter_size=5, num_blocks=12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 32, 3, 2, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in=hi, hidden=h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class EncoderSplit(Encoder):
    def __init__(self, num_epi, output_size=256, filter_size=5, num_blocks=12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        self.conv_start_epi = nn.Sequential(
                                    nn.Conv1d(num_epi, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)
        self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        x = torch.cat([seq, epi], dim=1)
        out = self.conv_end(x)
        return out

class TransformerLayer(torch.nn.TransformerEncoderLayer):
    # Pre-LN structure
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # MHA section
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)

        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights

class TransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, record_attn=False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weight_list = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)
        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):
    def __init__(self, hidden, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttnModule(nn.Module):
    def __init__(self, hidden=128, layers=8, record_attn=False, inpu_dim=256):
        super(AttnModule, self).__init__()
        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout=0.1)
        encoder_layers = TransformerLayer(hidden,
                                          nhead=8,
                                          dropout=0.1,
                                          dim_feedforward=512,
                                          batch_first=True)
        self.module = TransformerEncoder(encoder_layers,
                                         layers,
                                         record_attn=record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output

    def inference(self, x):
        return self.module(x)

# ========== 新增：混合注意力解码器 ==========
class HybridAttentionDecoder(nn.Module):
    """
    混合注意力解码器：将 bin 特征解码为接触矩阵。
    输入: [batch, num_bins, hidden_dim]
    输出: [batch, num_bins, num_bins]
    使用注意力掩码实现混合注意力，无需 token_type_ids 输入。
    """
    def __init__(
        self,
        hidden_dim=256,
        num_bins=256,
        num_layers=6,
        num_heads=8,
        use_causal_on_queries=False,  # 查询部分是否因果
        dropout=0.1,
        max_len=512  # 2 * num_bins
    ):
        super().__init__()
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self.use_causal = use_causal_on_queries

        # 可学习查询嵌入
        self.query_embed = nn.Embedding(num_bins, hidden_dim)

        # 输出投影：每个查询的隐藏状态 -> 该行所有 bin 的接触值
        self.output_proj = nn.Linear(hidden_dim, num_bins)

        # 位置编码（使用已有的 PositionalEncoding，但需要适配 batch_first）
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max_len)

        # Transformer 编码器层（复用已有的 TransformerEncoder 和 TransformerLayer）
        encoder_layer = TransformerLayer(
            hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=4 * hidden_dim,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers, norm=None, record_attn=False)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_hybrid_mask(self, batch_size, seq_len, num_image, num_query, device, dtype):
        """
        生成混合注意力掩码，形状 [batch, seq_len, seq_len]。
        规则：
        - 图像部分内部：全 0（双向）
        - 图像 -> 查询：全 0（双向）
        - 查询 -> 图像：全 0（双向）
        - 查询内部：如果 use_causal 为 True，则为下三角掩码（包含对角线），否则全 0
        """
        mask = torch.full((batch_size, seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
        # 图像部分内部
        mask[:, :num_image, :num_image] = 0
        # 图像 -> 查询
        mask[:, :num_image, num_image:] = 0
        # 查询 -> 图像
        mask[:, num_image:, :num_image] = 0
        # 查询内部
        if self.use_causal:
            # 下三角掩码（包含对角线）
            causal_mask = torch.tril(torch.ones(num_query, num_query, device=device)).bool()
            mask[:, num_image:, num_image:] = torch.where(causal_mask, 0.0, float('-inf'))
        else:
            mask[:, num_image:, num_image:] = 0
        return mask

    def forward(self, bin_features):
        """
        bin_features: [batch, num_bins, hidden_dim]
        """
        batch_size, num_bins, _ = bin_features.shape
        device = bin_features.device
        dtype = bin_features.dtype
        assert num_bins == self.num_bins, f"Expected {self.num_bins} bins, got {num_bins}"

        # 生成查询并扩展到 batch
        queries = self.query_embed.weight                     # [num_bins, hidden]
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_bins, hidden]

        # 拼接 bin 特征和查询
        combined = torch.cat([bin_features, queries], dim=1)  # [batch, 2*num_bins, hidden]

        # 添加位置编码（PositionalEncoding 期望 [seq_len, batch, hidden]）
        combined = combined.transpose(0, 1)                    # [2*num_bins, batch, hidden]
        combined = self.pos_encoder(combined)                  # 仍为 [2*num_bins, batch, hidden]
        combined = combined.transpose(0, 1)                    # 转回 [batch, 2*num_bins, hidden]

        # 生成掩码
        mask = self._create_hybrid_mask(
            batch_size=batch_size,
            seq_len=2 * num_bins,
            num_image=num_bins,
            num_query=num_bins,
            device=device,
            dtype=dtype
        )

        # 通过 TransformerEncoder
        output = self.encoder(combined, mask=mask)             # [batch, 2*num_bins, hidden]

        # 取查询部分的输出
        query_outputs = output[:, num_bins:, :]                # [batch, num_bins, hidden]

        # 映射到接触矩阵的每一行
        contact = self.output_proj(query_outputs)               # [batch, num_bins, num_bins]
        return contact
