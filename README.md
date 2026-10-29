# MNIST with PyTorch

从零实现 MNIST 手写数字分类，作为深度学习入门的第一个完整项目。

## 环境

- Python 3.14
- PyTorch 2.15.0 (CPU)
- torchvision

## 运行

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python train.py
```

首次运行会自动下载 MNIST 到 `data/`。

## 模型

```
Flatten -> Linear(784, 256) -> ReLU -> Linear(256, 10)
```

## 结果

训练 3 个 epoch，测试集准确率：

| epoch | acc |
|---|---|
| 1 | 96.60% |
| 2 | 97.61% |
| 3 | 97.89% |
