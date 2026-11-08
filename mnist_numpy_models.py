"""Small NumPy neural-network library used by mnist-numpy.py.

The layer organization follows the course examples for TwoLayerNet and
SimpleConvNet, while the implementation here is self-contained.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(logits, labels):
    probabilities = softmax(logits)
    return -np.mean(np.log(probabilities[np.arange(labels.size), labels] + 1e-12))


class TwoLayerNet:
    def __init__(self, input_size=784, hidden_size=50, output_size=10,
                 weight_init_std=0.01, seed=20261104, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0.0, weight_init_std, (input_size, hidden_size)).astype(dtype),
            "b1": np.zeros(hidden_size, dtype=dtype),
            "W2": rng.normal(0.0, weight_init_std, (hidden_size, output_size)).astype(dtype),
            "b2": np.zeros(output_size, dtype=dtype),
        }

    def predict(self, x):
        hidden = sigmoid(x @ self.params["W1"] + self.params["b1"])
        return hidden @ self.params["W2"] + self.params["b2"]

    def loss(self, x, labels):
        return cross_entropy(self.predict(x), labels)

    def accuracy(self, x, labels, batch_size=1000):
        correct = 0
        for start in range(0, len(x), batch_size):
            prediction = np.argmax(self.predict(x[start:start + batch_size]), axis=1)
            correct += int(np.sum(prediction == labels[start:start + batch_size]))
        return correct / len(x)

    def gradient(self, x, labels, return_loss=False):
        W1, W2 = self.params["W1"], self.params["W2"]
        a1 = x @ W1 + self.params["b1"]
        z1 = sigmoid(a1)
        logits = z1 @ W2 + self.params["b2"]
        probabilities = softmax(logits)
        loss_value = -np.mean(
            np.log(probabilities[np.arange(labels.size), labels] + 1e-12)
        )

        batch_size = x.shape[0]
        dlogits = probabilities
        dlogits[np.arange(batch_size), labels] -= 1.0
        dlogits /= batch_size
        gradients = {
            "W2": z1.T @ dlogits,
            "b2": np.sum(dlogits, axis=0),
        }
        dz1 = dlogits @ W2.T
        da1 = dz1 * z1 * (1.0 - z1)
        gradients["W1"] = x.T @ da1
        gradients["b1"] = np.sum(da1, axis=0)
        if return_loss:
            return gradients, float(loss_value)
        return gradients

    def numerical_gradient(self, x, labels, h=1e-5):
        gradients = {}
        for name, parameter in self.params.items():
            gradient = np.zeros_like(parameter)
            iterator = np.nditer(parameter, flags=["multi_index"], op_flags=["readwrite"])
            while not iterator.finished:
                index = iterator.multi_index
                original = parameter[index]
                parameter[index] = original + h
                positive_loss = self.loss(x, labels)
                parameter[index] = original - h
                negative_loss = self.loss(x, labels)
                parameter[index] = original
                gradient[index] = (positive_loss - negative_loss) / (2.0 * h)
                iterator.iternext()
            gradients[name] = gradient
        return gradients


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    padded = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    col = np.empty((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = padded[:, :, y:y_max:stride, x:x_max:stride]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    reshaped = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    reshaped = reshaped.transpose(0, 3, 4, 5, 1, 2)
    image = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1),
                     dtype=col.dtype)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            image[:, :, y:y_max:stride, x:x_max:stride] += reshaped[:, :, y, x, :, :]
    return image[:, :, pad:H + pad, pad:W + pad]


class Relu:
    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout = dout.copy()
        dout[self.mask] = 0
        return dout


class Affine:
    def __init__(self, W, b):
        self.W, self.b = W, b

    def forward(self, x):
        self.original_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return self.x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return (dout @ self.W.T).reshape(self.original_shape)


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W, self.b, self.stride, self.pad = W, b, stride, pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, _, H, W = x.shape
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1
        self.x = x
        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T
        out = self.col @ self.col_W + self.b
        return out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

    def backward(self, dout):
        FN, _, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = (self.col.T @ dout).T.reshape(self.W.shape)
        dcol = dout @ self.col_W.T
        return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)


class Pooling:
    def __init__(self, pool_h=2, pool_w=2, stride=2, pad=0):
        self.pool_h, self.pool_w, self.stride, self.pad = pool_h, pool_w, stride, pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        self.arg_max = np.argmax(col, axis=1)
        self.x_shape = x.shape
        out = np.max(col, axis=1)
        return out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size), dtype=dout.dtype)
        dmax[np.arange(self.arg_max.size), self.arg_max] = dout.ravel()
        dcol = dmax.reshape(-1, pool_size)
        return col2im(dcol, self.x_shape, self.pool_h, self.pool_w,
                      self.stride, self.pad)


class SoftmaxWithLoss:
    def forward(self, logits, labels):
        self.labels = labels
        self.probabilities = softmax(logits)
        return -np.mean(np.log(self.probabilities[np.arange(labels.size), labels] + 1e-12))

    def backward(self, dout=1.0):
        dx = self.probabilities.copy()
        dx[np.arange(self.labels.size), self.labels] -= 1.0
        return dx * (dout / self.labels.size)


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_filters=30, filter_size=5,
                 hidden_size=100, output_size=10, weight_init_std=0.01,
                 seed=20261104):
        C, H, W = input_dim
        conv_h = H - filter_size + 1
        conv_w = W - filter_size + 1
        pool_h, pool_w = conv_h // 2, conv_w // 2
        pool_output_size = conv_filters * pool_h * pool_w
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0.0, weight_init_std,
                             (conv_filters, C, filter_size, filter_size)).astype(np.float32),
            "b1": np.zeros(conv_filters, dtype=np.float32),
            "W2": rng.normal(0.0, weight_init_std,
                             (pool_output_size, hidden_size)).astype(np.float32),
            "b2": np.zeros(hidden_size, dtype=np.float32),
            "W3": rng.normal(0.0, weight_init_std,
                             (hidden_size, output_size)).astype(np.float32),
            "b3": np.zeros(output_size, dtype=np.float32),
        }
        self.layers = OrderedDict([
            ("Conv1", Convolution(self.params["W1"], self.params["b1"])),
            ("Relu1", Relu()),
            ("Pool1", Pooling(2, 2, 2)),
            ("Affine1", Affine(self.params["W2"], self.params["b2"])),
            ("Relu2", Relu()),
            ("Affine2", Affine(self.params["W3"], self.params["b3"])),
        ])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, labels):
        return self.last_layer.forward(self.predict(x), labels)

    def accuracy(self, x, labels, batch_size=100):
        correct = 0
        for start in range(0, len(x), batch_size):
            prediction = np.argmax(self.predict(x[start:start + batch_size]), axis=1)
            correct += int(np.sum(prediction == labels[start:start + batch_size]))
        return correct / len(x)

    def gradient(self, x, labels, return_loss=False):
        loss_value = self.loss(x, labels)
        dout = self.last_layer.backward()
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)
        gradients = {
            "W1": self.layers["Conv1"].dW,
            "b1": self.layers["Conv1"].db,
            "W2": self.layers["Affine1"].dW,
            "b2": self.layers["Affine1"].db,
            "W3": self.layers["Affine2"].dW,
            "b3": self.layers["Affine2"].db,
        }
        if return_loss:
            return gradients, float(loss_value)
        return gradients


class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        for name in params:
            params[name] -= self.learning_rate * gradients[name]


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = beta1, beta2
        self.iteration = 0
        self.m, self.v = {}, {}

    def update(self, params, gradients):
        if not self.m:
            self.m = {name: np.zeros_like(value) for name, value in params.items()}
            self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.iteration += 1
        adjusted = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.iteration)
        adjusted /= 1.0 - self.beta1 ** self.iteration
        for name in params:
            self.m[name] += (1.0 - self.beta1) * (gradients[name] - self.m[name])
            self.v[name] += (1.0 - self.beta2) * (gradients[name] ** 2 - self.v[name])
            params[name] -= adjusted * self.m[name] / (np.sqrt(self.v[name]) + 1e-7)

