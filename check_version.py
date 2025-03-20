
# 检查版本
import torch

print(torch.__version__)
print('device:', "cuda:0" if torch.cuda.is_available() else "cpu")
