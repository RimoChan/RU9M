from pathlib import Path

import fire
import torch
from safetensors.torch import load_file, save_file


def 转换权重(输入路径, alpha=None):
    原始权重 = load_file(str(输入路径))
    rank = next(v.shape[0] for k, v in 原始权重.items() if k.endswith('.lora_A.weight'))
    if alpha is None:
        alpha = rank // 2

    bf16权重 = {}
    fp8权重 = {}
    模块集合 = set()

    for k, v in 原始权重.items():
        新key = k.replace('transformer.', 'diffusion_model.', 1) if k.startswith('transformer.') else k
        if 新key.endswith('.lora_A.weight'):
            模块集合.add(新key[:-len('.lora_A.weight')])
        bf16权重[新key] = v.to(torch.bfloat16)
        fp8权重[新key] = v.to(torch.float8_e4m3fn)

    alpha张量 = torch.tensor(float(alpha), dtype=torch.bfloat16)
    for 模块 in sorted(模块集合):
        alpha_key = f'{模块}.alpha'
        bf16权重[alpha_key] = alpha张量.clone()
        fp8权重[alpha_key] = alpha张量.clone()

    metadata = {
        'format': 'pt',
        'modelspec.architecture': 'moss/lora',
        'ss_network_dim': str(rank),
        'ss_network_alpha': str(alpha),
    }
    return bf16权重, fp8权重, metadata


def main(输入='ru9moss_128_98000.safetensors', alpha=None):
    输入路径 = Path(输入)
    bf16权重, fp8权重, metadata = 转换权重(输入路径, alpha)

    基础名 = 输入路径.with_suffix('')
    bf16路径 = Path(f'{基础名}_comfy_bf16.safetensors')
    fp8路径 = Path(f'{基础名}_comfy_fp8.safetensors')

    save_file(bf16权重, str(bf16路径), metadata=metadata)
    save_file(fp8权重, str(fp8路径), metadata=metadata)
    print(f'完成: {bf16路径} ({bf16路径.stat().st_size / 1024 / 1024:.2f} MB)')
    print(f'完成: {fp8路径} ({fp8路径.stat().st_size / 1024 / 1024:.2f} MB)')


if __name__ == '__main__':
    fire.Fire(main)
