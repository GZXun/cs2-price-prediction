"""用公开组件组装精简网络，展示结构及张量接口，不复刻业务推理流程。"""

import torch.nn as nn

from model.components import TemporalAttentionPooling, VariableSelectionNetwork
from model.xlstm import xLSTM


class DemoSequenceClassifier(nn.Module):
    """匿名特征序列 -> 未校准 logit，同时返回便于检查的注意力张量。"""

    def __init__(self, in_features=8, hidden_dim=16, sequence_length=24):
        super().__init__()
        if in_features < 1 or hidden_dim < 2 or sequence_length < 1:
            raise ValueError("输入维数和序列长度须为正数，隐藏维数至少为 2")
        self.in_features = in_features
        self.selection = VariableSelectionNetwork(in_features, hidden_dim)
        self.encoder = xLSTM(
            input_size=in_features, hidden_size=hidden_dim, num_layers=2,
            dropout=0.1, bidirectional=True, block_types=["mlstm", "slstm"],
        )
        self.pool = TemporalAttentionPooling(hidden_dim * 2, seq_len=sequence_length)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 1))

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != self.in_features:
            raise ValueError("输入须为 (batch, time, in_features)")
        if inputs.shape[0] < 1 or inputs.shape[1] < 1:
            raise ValueError("批次和时间维度不能为空")
        features, feature_weights = self.selection(inputs.float())
        encoded, _ = self.encoder(features)
        pooled, temporal_weights = self.pool(encoded)
        return self.head(pooled), feature_weights, temporal_weights
