import numpy as np
import os
import sys

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None

sys.path.insert(0, os.path.dirname(__file__))
from AcoshGrad import impl

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)


# --- Case 0 ---
np.random.seed(42 + 0)
os.makedirs("input/case0", exist_ok=True)
y = np.random.uniform(low=1.01, high=10, size=(1, 8)).astype(np.float32)
y.tofile("input/case0/y.bin")
dy = np.random.uniform(low=-2, high=2, size=(1, 8)).astype(np.float32)
dy.tofile("input/case0/dy.bin")



os.makedirs("output/golden_case0", exist_ok=True)
golden = impl(y, dy)
if golden is not None:
    golden.tofile("output/golden_case0/golden_z.bin")

print(f"Generated test data and golden output for 1 cases.")
