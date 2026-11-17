"""展示用变量选择与时序池化，提取自研究项目 model/Thgnn.py。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableSelectionNetwork(nn.Module):
    """
    变量选择网络：学习哪些特征对预测最重要
    可解释性强，有助于特征工程迭代
    """
    def __init__(self, in_features, hidden_dim, dropout=0.2):
        super().__init__()

        # 特征重要性网络（使用Sigmoid独立门控，不强制特征竞争）
        self.importance_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_features),
            nn.Sigmoid()  # 独立门控，每个特征独立学习重要性
        )

        # 特征变换
        self.transform = nn.Linear(in_features, in_features)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # 计算全局特征权重（基于序列均值）
        x_mean = x.mean(dim=1)  # (batch, features)
        weights = self.importance_net(x_mean)  # (batch, features)

        # 应用权重 (扩展到序列维度)
        weighted_x = x * weights.unsqueeze(1)  # (batch, seq_len, features)

        # 变换并归一化
        output = self.layer_norm(self.transform(weighted_x))

        return output, weights  # 返回权重用于可视化


class TemporalAttentionPooling(nn.Module):
    """
    时序注意力池化：学习哪些时间步对预测最重要
    添加时间位置偏置，使模型更关注近期数据（此处仅用于结构演示）
    """
    def __init__(self, hidden_dim, seq_len=180, time_bias_weight=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # 可学习的时间位置偏置（越靠后的时间步偏置越大）
        self.time_bias = nn.Parameter(torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1))
        self.time_bias_weight = time_bias_weight

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        _, seq_len, _ = lstm_output.shape

        # 内容注意力分数
        attn_scores = self.attention(lstm_output)  # (batch, seq_len, 1)

        # 添加时间位置偏置（近期数据权重更高）
        # 处理序列长度不匹配的情况
        if seq_len != self.time_bias.size(1):
            time_bias = torch.linspace(0, 1, seq_len, device=lstm_output.device).view(1, seq_len, 1)
        else:
            time_bias = self.time_bias.to(lstm_output.device)

        # 混合注意力：内容注意力 + 时间偏置
        combined_scores = attn_scores + self.time_bias_weight * time_bias
        attn_weights = F.softmax(combined_scores, dim=1)

        # 加权求和
        context = (lstm_output * attn_weights).sum(dim=1)  # (batch, hidden_dim)

        return context, attn_weights.squeeze(-1)
