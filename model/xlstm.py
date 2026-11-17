"""本项目的 xLSTM 实验实现，提取自 model/xlstm.py。

参考 Beck et al., xLSTM: Extended Long Short-Term Memory (2024)。
本实现不是论文官方代码，不声明与论文结构、性能完全一致。
所有参数均由 PyTorch 随机初始化；此文件不包含训练权重。
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def slstm_loop_jit(gates_x_seq: torch.Tensor, 
                   W_h_weight: torch.Tensor,
                   init_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    h, c, n, m = init_states
    seq_len = gates_x_seq.size(1)
    
    h_seq = torch.jit.annotate(List[torch.Tensor], [])
    
    for t in range(seq_len):
        pre_x_t = gates_x_seq[:, t, :]
        gates = pre_x_t + F.linear(h, W_h_weight)
        log_i, log_f, o_pre, z_pre = gates.chunk(4, dim=-1)
        
        m_new = torch.max(log_f + m, log_i)
        i = torch.exp(log_i - m_new)
        f = torch.exp(log_f + m - m_new)
        
        o = torch.sigmoid(o_pre)
        z = torch.tanh(z_pre)
        
        c = f * c + i * z
        n = f * n + i
        
        h = o * c / torch.clamp(n.abs(), min=1.0)
        m = m_new
        
        h_seq.append(h)
    
    return torch.stack(h_seq, dim=1), (h, c, n, m)


@torch.jit.script
def mlstm_loop_jit(k_seq: torch.Tensor,
                   v_seq: torch.Tensor,
                   q_seq: torch.Tensor,
                   log_i_seq: torch.Tensor,
                   log_f_seq: torch.Tensor,
                   x_norm_seq: torch.Tensor,
                   W_o_weight: torch.Tensor,
                   W_o_bias: torch.Tensor,
                   init_states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    h, C, n, m = init_states
    seq_len = k_seq.size(1)
    
    h_seq = torch.jit.annotate(List[torch.Tensor], [])
    
    for t in range(seq_len):
        k = k_seq[:, t, :]
        v = v_seq[:, t, :]
        q = q_seq[:, t, :]
        log_i = log_i_seq[:, t, :]
        log_f = log_f_seq[:, t, :]
        x_t = x_norm_seq[:, t, :]
        
        m_new = torch.max(log_f + m, log_i)
        i = torch.exp(log_i - m_new)
        f = torch.exp(log_f + m - m_new)
        
        o_input = torch.cat([x_t, h], dim=-1)
        o = torch.sigmoid(F.linear(o_input, W_o_weight, W_o_bias))
        
        vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        C = f.unsqueeze(-1) * C + i.unsqueeze(-1) * vk
        
        n = f * n + i * k
        
        h_tilde = torch.bmm(C, q.unsqueeze(2)).squeeze(2)
        nq = (n * q).sum(dim=-1, keepdim=True)
        h_tilde = h_tilde / torch.clamp(nq.abs(), min=1.0)
        
        h = o * h_tilde
        m = m_new
        
        h_seq.append(h)
    
    return torch.stack(h_seq, dim=1), (h, C, n, m)


class sLSTMCell(nn.Module):
    """
    sLSTM Cell: 带指数门控的标量LSTM

    与标准LSTM的区别:
    - 输入门/遗忘门使用 exp() 替代 sigmoid()，记忆选择更锐利
    - 引入归一化器状态 n_t，稳定隐藏状态计算
    - 使用对数空间稳定化技巧，防止 exp() 溢出
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 合并投影: [input_gate, forget_gate, output_gate, cell_input]
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        """初始化参数，遗忘门偏置设为正值以保持长期记忆"""
        with torch.no_grad():
            # 遗忘门偏置初始化为 1.0（鼓励记忆保持）
            self.W_x.bias[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, pre_x_t, states):
        """
        [注: 此前向方法当前仅作为逻辑参考和调试备用。
         实际的高效计算已提取至文件顶部的 @torch.jit.script 函数 slstm_loop_jit 中。]

        Args:
            pre_x_t: pre-computed self.W_x(x_t) shape: (batch, 4*hidden_size)
            states: (h, c, n, m) 状态元组
                h: 隐藏状态 (batch, hidden_size)
                c: 细胞状态 (batch, hidden_size)
                n: 归一化器 (batch, hidden_size)
                m: 对数空间稳定器 (batch, hidden_size)
        """
        h, c, n, m = states

        # 计算所有门（合并计算提高效率）
        gates = pre_x_t + self.W_h(h)
        log_i, log_f, o_pre, z_pre = gates.chunk(4, dim=-1)

        # === 指数门控 + 对数空间稳定化 ===
        # m_new = max(log_f + m_old, log_i) 防止 exp() 溢出
        m_new = torch.max(log_f + m, log_i)
        i = torch.exp(log_i - m_new)           # 稳定化的输入门
        f = torch.exp(log_f + m - m_new)        # 稳定化的遗忘门

        # 标准门控
        o = torch.sigmoid(o_pre)                # 输出门保持 sigmoid
        z = torch.tanh(z_pre)                   # 细胞输入

        # 状态更新
        c_new = f * c + i * z                   # 细胞状态
        n_new = f * n + i                       # 归一化器（追踪累积输入权重）

        # 归一化隐藏状态（防止值过大）
        h_new = o * c_new / torch.clamp(n_new.abs(), min=1.0)

        return h_new, (h_new, c_new, n_new, m_new)

    def init_states(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),   # h
            torch.zeros(batch_size, self.hidden_size, device=device),   # c
            torch.ones(batch_size, self.hidden_size, device=device),    # n (初始为1)
            torch.zeros(batch_size, self.hidden_size, device=device),   # m (稳定器)
        )


class mLSTMCell(nn.Module):
    """
    mLSTM Cell: 带矩阵记忆的LSTM

    与标准LSTM的区别:
    - 使用矩阵 C ∈ R^{d×d} 替代向量 c ∈ R^d，记忆容量大幅提升
    - 引入 Key-Value-Query 机制（类似 Transformer 的注意力）
    - 标量指数门控（仅依赖输入，不依赖隐藏状态）
    - 记忆容量从 O(d) 提升到 O(d²)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Key-Value-Query 投影（仅依赖输入）
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.W_q = nn.Linear(input_size, hidden_size)

        # 标量门控（仅依赖输入 → 可并行化）
        self.w_i = nn.Linear(input_size, 1)
        self.w_f = nn.Linear(input_size, 1)

        # 输出门（依赖输入 + 隐藏状态）
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.w_f.bias.fill_(1.0)

    def forward(self, pre_t, x_t, states):
        """
        [注: 此前向方法当前仅作为逻辑参考和调试备用。
         实际的高效计算已提取至文件顶部的 @torch.jit.script 函数 mlstm_loop_jit 中。]

        Args:
            pre_t: 元组 (k, v, q, log_i, log_f)
            x_t: 原输入用于输出门 (batch, input_size)
            states: (h, C, n, m)
                h: 隐藏状态 (batch, hidden_size)
                C: 矩阵记忆 (batch, hidden_size, hidden_size)
                n: 归一化向量 (batch, hidden_size)
                m: 稳定器标量 (batch, 1)
        """
        h, C, n, m = states
        k, v, q, log_i, log_f = pre_t

        # 对数空间稳定化
        m_new = torch.max(log_f + m, log_i)
        i = torch.exp(log_i - m_new)       # (batch, 1)
        f = torch.exp(log_f + m - m_new)   # (batch, 1)

        # 输出门
        o = torch.sigmoid(self.W_o(torch.cat([x_t, h], dim=-1)))   # (batch, d)

        # === 矩阵记忆更新 ===
        # C_new = f * C + i * (v ⊗ k^T)
        # v ⊗ k^T: 外积 (batch, d, 1) × (batch, 1, d) → (batch, d, d)
        vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))     # (batch, d, d)
        C_new = f.unsqueeze(-1) * C + i.unsqueeze(-1) * vk

        # 归一化向量更新: n_new = f * n + i * k
        n_new = f * n + i * k   # 广播: (batch,1)*(batch,d) → (batch,d)

        # === 从矩阵记忆中检索 ===
        # h_tilde = C @ q / max(|n^T @ q|, 1)
        h_tilde = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2)   # (batch, d)
        nq = (n_new * q).sum(dim=-1, keepdim=True)               # (batch, 1)
        h_tilde = h_tilde / torch.clamp(nq.abs(), min=1.0)

        # 最终隐藏状态
        h_new = o * h_tilde

        return h_new, (h_new, C_new, n_new, m_new)

    def init_states(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),                       # h
            torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device),     # C (矩阵记忆)
            torch.zeros(batch_size, self.hidden_size, device=device),                       # n
            torch.zeros(batch_size, 1, device=device),                                      # m (标量稳定器)
        )


class xLSTMLayer(nn.Module):
    """
    单层 xLSTM 块，包含 Pre-LayerNorm、Causal Conv1d 和可选残差连接

    结构: LayerNorm → Causal Conv1d → SiLU → xLSTM Cell → Dropout [+ Residual]
    """

    def __init__(self, input_size, hidden_size, cell_type='mlstm', dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        # 因果深度卷积 (Causal Depthwise Conv1d)，用于平滑局部时序上下文
        self.conv_kernel_size = 4
        self.causal_conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=input_size
        )
        self.conv_act = nn.SiLU()

        if cell_type == 'mlstm':
            self.cell = mLSTMCell(input_size, hidden_size)
        else:
            self.cell = sLSTMCell(input_size, hidden_size)

        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

        # 当输入输出维度相同时使用残差连接
        self.use_residual = (input_size == hidden_size)

        # 输入输出维度不同时需要投影
        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            self.input_proj = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # Pre-LayerNorm
        x_norm = self.layer_norm(x)

        # Causal Conv1D (局部时序平滑)
        x_conv = x_norm.transpose(1, 2)  # (batch, input_size, seq_len)
        x_conv = self.causal_conv(x_conv)  # (batch, input_size, seq_len + padding)
        x_conv = x_conv[:, :, :seq_len]    # 截断右侧多余部分，保持因果性避免泄漏未来数据
        x_norm = self.conv_act(x_conv).transpose(1, 2)  # 还原为 (batch, seq_len, input_size)

        # 初始化状态
        states = self.cell.init_states(batch_size, x.device)

        # Pre-compute all linear projections for sequence
        if self.cell_type == 'mlstm':
            k_seq = self.cell.W_k(x_norm) / math.sqrt(self.hidden_size)
            v_seq = self.cell.W_v(x_norm)
            q_seq = self.cell.W_q(x_norm)
            log_i_seq = self.cell.w_i(x_norm)
            log_f_seq = self.cell.w_f(x_norm)
            
            output, states = mlstm_loop_jit(
                k_seq, v_seq, q_seq, log_i_seq, log_f_seq, x_norm,
                self.cell.W_o.weight, self.cell.W_o.bias, states
            )
        else:
            gates_x_seq = self.cell.W_x(x_norm)
            
            output, states = slstm_loop_jit(
                gates_x_seq, self.cell.W_h.weight, states
            )

        output = self.dropout(output)

        # 残差连接
        if self.use_residual:
            output = output + x
        elif self.input_proj is not None:
            # 维度不匹配时，投影后相加（保留输入信息）
            output = output + self.input_proj(x)

        return output


class xLSTM(nn.Module):
    """
    多层 xLSTM 堆叠，nn.LSTM 的直接替换

    特性:
    - 支持双向处理 (bidirectional)
    - 可配置每层使用 sLSTM 或 mLSTM
    - 默认配置: [mLSTM, sLSTM] 交替（论文推荐的混合策略）
    - 接口兼容 nn.LSTM: forward 返回 (output, None)

    参数量对比 (hidden_dim=64, 2层, 双向):
    - nn.LSTM:    ~180K 参数
    - xLSTM:      ~200K 参数 (mLSTM的矩阵记忆不增加参数，仅增加运行时内存)
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0,
                 bidirectional=True, block_types=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if block_types is None:
            # 默认: mLSTM 和 sLSTM 交替（本项目的实验配置）
            # mLSTM: 大记忆容量，适合捕捉长期模式
            # sLSTM: 精确门控，适合短期依赖和细粒度选择
            block_types = ['mlstm' if i % 2 == 0 else 'slstm'
                           for i in range(num_layers)]

        # 前向层
        self.forward_layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            # 最后一层不加 dropout（与 nn.LSTM 行为一致）
            layer_dropout = dropout if i < num_layers - 1 else 0.0
            self.forward_layers.append(
                xLSTMLayer(in_size, hidden_size, block_types[i], layer_dropout)
            )

        # 反向层（双向模式）
        if bidirectional:
            self.backward_layers = nn.ModuleList()
            for i in range(num_layers):
                in_size = input_size if i == 0 else hidden_size
                layer_dropout = dropout if i < num_layers - 1 else 0.0
                self.backward_layers.append(
                    xLSTMLayer(in_size, hidden_size, block_types[i], layer_dropout)
                )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            output: (batch, seq_len, hidden_size * num_directions)
            None: 占位符，兼容 nn.LSTM 接口
        """
        # 前向传播
        fwd_out = x
        for layer in self.forward_layers:
            fwd_out = layer(fwd_out)

        if self.bidirectional:
            # 反向传播：翻转序列 → 处理 → 翻转回来
            bwd_out = x.flip(1)
            for layer in self.backward_layers:
                bwd_out = layer(bwd_out)
            bwd_out = bwd_out.flip(1)

            # 拼接前向和反向输出
            output = torch.cat([fwd_out, bwd_out], dim=-1)
        else:
            output = fwd_out

        return output, None
