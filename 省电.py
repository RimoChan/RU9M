import os
import sys
import glob
import ctypes
import time
import torch


def 开启省电模式():
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    lib_path = glob.glob(os.path.join(torch_lib, "*cudart*"))[0]
    cudart = ctypes.CDLL(lib_path)
    return cudart.cudaSetDeviceFlags(0x04) == 0


def 测试CPU占用():
    x = torch.randn(8000, 8000, device="cuda")
    y = torch.randn(8000, 8000, device="cuda")
    for _ in range(3):
        torch.matmul(x, y)
    torch.cuda.synchronize()

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    for _ in range(25):
        x = torch.matmul(x, y)
    torch.cuda.synchronize()

    t_wall = time.perf_counter() - t0_wall
    t_cpu = time.process_time() - t0_cpu
    cpu_rate = (t_cpu / t_wall) * 100
    print(f"耗时: {t_wall:.2f}s | CPU占用: {t_cpu:.2f}s ({cpu_rate:.1f}%)")


if __name__ == "__main__":
    if sys.argv[-1] == "save":
        开启省电模式()
        测试CPU占用()
    else:
        测试CPU占用()
