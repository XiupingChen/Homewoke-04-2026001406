# Homework 4

姓名：陈秀萍  
学号：2026001406

本次作业包括两个部分：使用 NumPy 对图像进行低秩近似，以及分别使用 NumPy 和 PyTorch 完成 MNIST 手写数字识别。第二题实现了全连接网络 `TwoLayerNet` 和卷积神经网络 `SimpleConvNet`，并比较了四种实现的测试精度与训练时间。

## 文件说明

- `low-rank-approx.py`：图像低秩近似、小矩阵测试、误差与存储量计算。
- `mnist-numpy.py`：使用 NumPy 训练 `TwoLayerNet` 和 `SimpleConvNet`。
- `mnist-pytorch.py`：使用 PyTorch 训练相同结构的两个网络。
- `mnist_numpy_models.py`：NumPy 网络层、前向传播、反向传播和优化器。
- `mnist_data.py`：下载并读取 MNIST 原始 IDX 压缩文件。
- `summarize-results.py`：汇总四种实现的结果并绘制损失曲线。
- `environment-info.py`：查看 Python、软件包和计算设备信息。
- `assets/image.jpg`：第一题使用的原始图片。
- `results/`：实验生成的指标、损失曲线和误分类样本图。
- `report.pdf`：实验报告。

## 运行环境

本次正式实验使用的主要环境如下：

- 硬件：MacBook Air（Apple Silicon）。
- Python：3.12.14。
- NumPy：2.5.3。
- PyTorch：2.15.0。
- NumPy 计算设备：CPU。
- PyTorch 计算设备：MPS。

其他依赖包括 Pillow 和 Matplotlib，完整依赖见 `requirements.txt`。可以运行以下命令查看当前机器上的具体版本和设备信息：

```bash
python environment-info.py
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## MNIST 数据

程序使用 MNIST 的 60000 张训练图像和 10000 张测试图像。图像读取后缩放到 `[0,1]`，测试集只用于最终评价，不参与参数更新。

`mnist_data.py` 会从以下公开镜像下载 MNIST 原始 IDX 压缩文件：

```text
https://storage.googleapis.com/cvdf-datasets/mnist/
https://ossci-datasets.s3.amazonaws.com/mnist/
```

下载的数据保存在 `data/mnist/`，该目录不需要提交。如果自动下载失败，也可以手动下载四个 `.gz` 文件并放入该目录。

## 运行方法

在作业目录中运行第一题：

```bash
python low-rank-approx.py
```

在正式训练前，可以先运行 NumPy 梯度检查和 CNN 前后向传播测试：

```bash
python mnist-numpy.py --smoke-test
```

快速检查完整训练流程：

```bash
python mnist-numpy.py --quick
python mnist-pytorch.py --quick
```

正式实验：

```bash
python mnist-numpy.py
python mnist-pytorch.py --device auto
python summarize-results.py
```

正式训练中，批大小均为 100。`TwoLayerNet` 使用 SGD，学习率为 0.1，共更新 10000 次；`SimpleConvNet` 使用 Adam，学习率为 0.001，训练 2 个 epoch，共更新 1200 次。为控制纯 NumPy CNN 的运行时间，CNN 训练轮数由题目建议的 20 个 epoch 调整为 2 个 epoch，NumPy 和 PyTorch 使用相同训练轮数。

## 正式实验结果

| 框架 | 模型 | 测试精度 | 训练时间（秒） | 更新次数 | 设备 |
|---|---|---:|---:|---:|---|
| NumPy | TwoLayerNet | 94.78% | 5.689 | 10000 | CPU |
| PyTorch | TwoLayerNet | 94.65% | 8.450 | 10000 | MPS |
| NumPy | SimpleConvNet | 97.89% | 50.784 | 1200 | CPU |
| PyTorch | SimpleConvNet | 97.93% | 4.827 | 1200 | MPS |

NumPy 与 PyTorch 的同类模型取得了接近的测试精度。CNN 的测试精度高于全连接网络。训练时间来自不同计算设备，NumPy 使用 CPU，PyTorch 使用 MPS，因此时间差异不能只解释为框架差异。

## 课程代码来源

NumPy 网络的类接口、参数命名、层式组织和训练流程参考了课程讲义“神经网络的学习与卷积神经网络”及以下配套代码：

- `code/dl-02-learning-cnn/ch04/two_layer_net.py`
- `code/dl-02-learning-cnn/ch04/train_neuralnet.py`
- `code/dl-02-learning-cnn/ch07/simple_convnet.py`
- `code/dl-02-learning-cnn/ch07/train_convnet.py`

本作业在 `mnist_numpy_models.py` 中按照作业要求整理并实现相关网络。MNIST 原始 IDX 文件读取、训练计时、梯度检查、结果绘图、PyTorch 对照实现和结果汇总代码均在本作业中完成。

## AI 工具使用说明

本次作业使用 ChatGPT 辅助梳理题目要求、设计和检查代码结构、排查依赖及数据下载问题，并协助整理 README 和实验报告。本人对代码和结果进行了核对，并在本机完成全部程序运行。第一题通过小矩阵及 RcppEigen 参考结果验证；NumPy 全连接网络通过数值梯度检查；NumPy CNN 通过前后向传播和梯度形状检查；四种 MNIST 实现通过正式训练结果、损失曲线、测试精度和误分类样本进行验证。
