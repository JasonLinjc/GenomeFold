import torch
import torch.nn as nn
import corigami.model.blocks as blocks

# ========== 新增：混合注意力模型 ==========
class HybridHiCModel(nn.Module):
    """
    使用混合注意力解码器的新模型。
    保留原 Encoder 和可选的 AttnModule，替换 diagonalize + Decoder。
    """
    def __init__(
        self,
        num_genomic_features,
        mid_hidden=256,
        num_bins=256,
        use_attn_module=True,
        record_attn=False,
        decoder_layers=6,
        decoder_heads=8,
        use_causal_on_queries=False
    ):
        super().__init__()
        self.num_bins = num_bins
        self.mid_hidden = mid_hidden
        self.use_attn_module = use_attn_module
        self.record_attn = record_attn

        # 原 EncoderSplit（保持不变）
        self.encoder = blocks.EncoderSplit(
            num_genomic_features,
            output_size=mid_hidden,
            num_blocks=12
        )

        # 原 AttnModule（可选）
        if use_attn_module:
            self.attn = blocks.AttnModule(
                hidden=mid_hidden,
                record_attn=record_attn
            )

        # 新解码器
        self.decoder = blocks.HybridAttentionDecoder(
            hidden_dim=mid_hidden,
            num_bins=num_bins,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            use_causal_on_queries=use_causal_on_queries
        )

    def move_feature_forward(self, x):
        """将特征维度移到第1维（原模型中的辅助函数）"""
        return x.transpose(1, 2).contiguous()

    def forward(self, x):
        """
        x: [batch, length, feat]  例如 [batch, 256 * res, num_genomic_features]
        """
        # 原预处理
        x = self.move_feature_forward(x).float()          # [batch, feat, length]

        # Encoder
        x = self.encoder(x)                                 # [batch, mid_hidden, num_bins]

        # 转置回 [batch, num_bins, mid_hidden]
        x = x.transpose(1, 2)                               # [batch, num_bins, mid_hidden]

        # 可选 AttnModule
        if self.use_attn_module:
            if self.record_attn:
                x, attn_weights = self.attn(x)              # x 仍为 [batch, num_bins, mid_hidden]
            else:
                x = self.attn(x)

        # 新解码器直接输出接触矩阵
        contact = self.decoder(x)                            # [batch, num_bins, num_bins]

        if self.record_attn and self.use_attn_module:
            return contact, attn_weights
        else:
            return contact
