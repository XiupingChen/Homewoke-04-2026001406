"""Print version and hardware information for README/report."""

import platform
import sys

import matplotlib
import numpy
import PIL

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("Processor:", platform.processor() or "not reported")
print("NumPy:", numpy.__version__)
print("Pillow:", PIL.__version__)
print("Matplotlib:", matplotlib.__version__)

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
except ImportError:
    print("PyTorch: not installed")

