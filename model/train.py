import os
import sys
# Root path added for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils.preprocessing import load_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

train_data, test_data = load_mnist()