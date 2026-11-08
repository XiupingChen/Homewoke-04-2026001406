# Homework 4 实验报告（运行后补充结果）

姓名：陈秀萍  
学号：2026001406

## 1. Python 图像低秩压缩

### 1.1 方法

对灰度图像矩阵进行奇异值分解：

$$
A=U\Sigma V^T.
$$

保留最大的 $k$ 个奇异值及其左右奇异向量：

$$
A_k=U_{[:,1:k]}\Sigma_{1:k,1:k}V_{[:,1:k]}^T.
$$

程序使用 `np.linalg.svd(A, full_matrices=False)`，重构后使用 `np.clip` 将像素截断到 $[0,1]$。

### 1.2 测试与结果

小矩阵测试使用 $\operatorname{diag}(0.9,0.5,0.2)$，取 $k=2$。与相同输入下的 RcppEigen 结果比较，最大绝对误差为 0。输出尺寸和输入一致，像素均位于 $[0,1]$。

原图为 $1365\times2048$，按比例缩小后实际矩阵尺寸为 $341\times512$，共 174592 个像素。

| $k$ | 相对 Frobenius 误差 | 低秩因子数 | 原图像素数 | 存储比例 |
|---:|---:|---:|---:|---:|
| 5 | 0.32358 | 4270 | 174592 | 2.4457% |
| 20 | 0.21570 | 17080 | 174592 | 9.7828% |
| 50 | 0.13295 | 42700 | 174592 | 24.4570% |

随着 $k$ 增大，图像的轮廓和局部细节逐渐清晰，相对误差下降；同时，保存截断 SVD 因子所需的 $k(m+n+1)$ 个数近似随 $k$ 线性增加。

插入：`results/low-rank-results.png`

## 2. MNIST 手写数字识别

### 2.1 数据和实验设置

MNIST 包含 60000 张训练图像和 10000 张测试图像，像素缩放到 $[0,1]$。全连接网络将图像展开为 784 维向量；CNN 使用形状为 $N\times1\times28\times28$ 的输入。测试集不参与参数更新。

`TwoLayerNet` 的结构为 784-50-10，隐藏层使用 Sigmoid；批大小为 100，使用 SGD，学习率为 0.1，更新 10000 次。`SimpleConvNet` 使用 30 个 $5\times5$ 卷积核、ReLU、$2\times2$ 最大池化、100 个隐藏单元和 10 类输出；批大小为 100，使用 Adam，学习率为 0.001，训练 2 个 epoch。

NumPy 与 PyTorch 对同一模型使用相同的预处理、训练量、优化器和学习率，并分别设置固定随机种子。

### 2.2 NumPy 实现

NumPy 全连接网络显式实现前向传播、Softmax 交叉熵和反向传播。每次更新先随机抽取 mini-batch，调用 `gradient()` 计算梯度，再由 SGD 更新参数。小规模网络的反向传播梯度与中心差分数值梯度比较，最大绝对差异为 $1.111\times10^{-11}$。

NumPy CNN 通过 `im2col` 将卷积窗口转化为矩阵乘法，并显式实现卷积层、ReLU、最大池化、全连接层、Softmax 损失及相应反向传播。

### 2.3 PyTorch 实现

PyTorch 版本使用 `nn.Linear`、`nn.Conv2d`、`nn.ReLU`、`nn.MaxPool2d` 和自动微分构建相同结构。模型直接输出 logits，并传入 `nn.CrossEntropyLoss`，没有在损失函数之前额外使用 Softmax。测试时调用 `model.eval()`，并在 `torch.no_grad()` 中完成预测。

### 2.4 实验结果

运行完成后从 `results/comparison.csv` 填写下表。

| 框架 | 模型 | 测试精度 | 训练时间（秒） | 更新次数 | 计算设备 |
|---|---|---:|---:|---:|---|
| NumPy | TwoLayerNet | 待填写 | 待填写 | 10000 | CPU |
| PyTorch | TwoLayerNet | 待填写 | 待填写 | 10000 | 待填写 |
| NumPy | SimpleConvNet | 待填写 | 待填写 | 待填写 | CPU |
| PyTorch | SimpleConvNet | 待填写 | 待填写 | 待填写 | 待填写 |

插入四种实现的损失曲线：`results/all-loss-curves.png`。

插入以下误分类样本图：

- `results/numpy-two-layer-errors.png`
- `results/numpy-cnn-errors.png`
- `results/pytorch-two-layer-errors.png`
- `results/pytorch-cnn-errors.png`

### 2.5 结果分析

运行后根据实际结果说明 CNN 与全连接网络的精度差异，以及 NumPy 和 PyTorch 的训练时间差异。CNN 利用局部连接、权重共享和池化保留图像空间结构，通常比直接展开像素的全连接网络更适合图像分类。PyTorch 自动完成计算图构建、梯度计算和参数更新，因此不需要手写每一层的反向传播。

## 3. 运行环境

运行 `python environment-info.py` 后填写 Python、NumPy、PyTorch、硬件和实际计算设备。

