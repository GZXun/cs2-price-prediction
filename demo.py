"""用随机输入验证公开网络，不读取文件、不下载权重、不保存模型。"""

import argparse
import json
import platform

import torch

from model.demo_model import DemoSequenceClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-backward", action="store_true", help="使用合成标签检查梯度")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有可用 CUDA；请使用默认 CPU 演示")

    torch.manual_seed(42)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model = DemoSequenceClassifier().to(device).eval()
    inputs = torch.randn(4, 24, 8, device=device)
    with torch.no_grad():
        logits, feature_weights, temporal_weights = model(inputs)
    assert logits.shape == (4, 1)
    assert feature_weights.shape == (4, 8)
    assert temporal_weights.shape == (4, 24)
    for tensor in (logits, feature_weights, temporal_weights):
        assert torch.isfinite(tensor).all(), "发现非有限值"
    assert ((feature_weights >= 0) & (feature_weights <= 1)).all()
    assert torch.allclose(temporal_weights.sum(dim=1), torch.ones(4, device=device), atol=1e-5)

    backward_checked = False
    if args.check_backward:
        model.train()
        targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]], device=device)
        train_logits, _, _ = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(train_logits, targets)
        assert torch.isfinite(loss), "损失不是有限值"
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        assert all(gradient is not None for gradient in gradients), "存在没有梯度的参数"
        assert all(torch.isfinite(gradient).all() for gradient in gradients), "存在非有限梯度"
        backward_checked = True

    print(json.dumps({
        "demo_only": True,
        "weights": "random initialization; no checkpoint loaded",
        "data": "synthetic random tensors; no market data loaded",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "input_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "feature_weights_shape": list(feature_weights.shape),
        "temporal_weights_shape": list(temporal_weights.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "forward_checks_passed": True,
        "backward_checks_passed": backward_checked,
        "warning": "This demo does not measure prediction performance.",
    }, indent=2))


if __name__ == "__main__":
    main()
