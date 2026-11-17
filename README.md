# CS2 Price Prediction · CS2 饰品价格趋势预测研究

> 基于 PyTorch 与 xLSTM 的 CS2 饰品市场时序建模作品集。公开模型组件和合成输入演示，不包含训练好的权重、真实市场数据、生产特征配置或原项目的 Git 历史。

本项目围绕 CS2 市场序列开展建模研究。展示版重点呈现变量选择、xLSTM 时序编码和注意力池化的实现思路；演示网络是从研究代码抽取组件后组装的精简版本，不等同于完整业务模型。

研究方向是利用历史观测窗口识别价格趋势相关信号；公开示例采用二分类 logit 接口，不输出具体目标价格。仓库名称为 `cs2-price-prediction`，让读者直接了解研究领域。

## 相关项目

| 项目 | 侧重点 |
| --- | --- |
| [CSDATA](https://github.com/GZXun/csdata) | CS2 饰品市场观察：饰品异动、市场信号与大商库存监控 |
| **CS2 Price Prediction** | 模型研究：变量选择、xLSTM 时序编码、注意力池化与可运行的合成演示 |

两个项目分别展示市场观察工具与时序模型研究；本仓库不代表相关模型已经接入 CSDATA 线上服务。

## 公开范围

- 精选模型组件及其连接方式。
- 使用匿名随机特征的前向传播演示。
- 输入输出、实现边界和验证说明。

## 不公开的内容

- 已训练的模型参数、标准化器、特征筛选结果和校准产物。
- 原始行情、商品明细、训练样本、预测明细和私有实验记录。
- 业务标签规则、生产阈值、凭据、开发工具历史和原仓库提交历史。

随机初始化网络的输出没有市场预测意义。本展示版不提供实盘预测，也不以合成演示结果代表真实模型效果。

## 快速演示

在装有 PyTorch 的 Python 环境中，进入本目录运行：

```console
python demo.py
python demo.py --check-backward
python check_release.py
```

默认使用 CPU；有可用 CUDA 的环境可使用 `python demo.py --device cuda --check-backward`。不需要下载数据或模型。

`requirements.txt` 记录研究环境使用的 PyTorch 版本。新建环境时可使用 `python -m pip install -r requirements.txt`，但需匹配该旧版 PyTorch 支持的 Python 和平台；本次本地核验环境为 Python 3.8.10 / PyTorch 1.8.0+cu111。未验证其他版本的兼容性。

演示检查 `(4, 24, 8)` 输入到 `(4, 1)` 输出的前向计算、时间权重归一化及可选的梯度传播。具体实测情况见 [VERIFICATION.md](VERIFICATION.md)。

## 代码导航

| 文件 | 内容 |
| --- | --- |
| `model/components.py` | 变量选择、时序注意力池化 |
| `model/xlstm.py` | 实验版 mLSTM / sLSTM 编码器 |
| `model/demo_model.py` | 公开组件组成的精简分类网络 |
| `demo.py` | 匿名合成输入、形状和数值检查 |
| `check_release.py` | 白名单检查与可选 ZIP 打包 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 设计思路、来源与实现边界 |
| [PUBLICATION.md](PUBLICATION.md) | 独立仓库发布步骤 |

本项目通过不分发私有数据和训练工件来保留已有训练成果。公开范围限于此仓库的展示内容，未提供生产模型或完整业务流水线。本仓库目前未附带开源许可证；引用的研究来源见 [ARCHITECTURE.md](ARCHITECTURE.md)。
