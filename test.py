import torch

print("Torch version:", torch.__version__)
print("XPU available:", torch.xpu.is_available())

if torch.xpu.is_available():
    print("XPU device count:", torch.xpu.device_count())
    print("Device name:", torch.xpu.get_device_name(0))
    print("Current device:", torch.xpu.current_device())
else:
    print("xpu not available")