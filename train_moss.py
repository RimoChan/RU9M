import os
os.environ['PYTORCH_ALLOC_CONF'] = os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'

import csv
import sys
import math
import time
import json
import pickle
import random
import logging
import argparse
import itertools
import importlib.util
from pathlib import Path
from functools import lru_cache

import fire
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from transformers import AutoModel, AutoTokenizer
from diffusers import AutoencoderKLFlux2, Krea2Pipeline
from diffusers.loaders import PeftAdapterMixin
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module

from common import clean, 生成optimizer, 哈, 计时, buffered_iterator, add_image_jpeg, cosine_with_restart_scheduler改


@lru_cache()
def _Kaloscope_2_model():
    from timm.models import create_model
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    sys.path.append('S:/lsnet-test')
    from model import lsnet_artist

    checkpoint_path = hf_hub_download(repo_id='heathcliff01/Kaloscope2.0', filename='448-90.13/best_checkpoint.pth')
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    state = checkpoint['model']

    num_classes, feature_dim = state["head.l.weight"].shape
    model = create_model('lsnet_xl_artist_448', pretrained=False, num_classes=num_classes, feature_dim=feature_dim)
    model.load_state_dict(state, strict=True)
    model = model.to('cuda').eval()

    data_config = resolve_data_config({"input_size": (3, 448, 448)}, model=model)
    transform = create_transform(**data_config, is_training=False)

    csv_path = hf_hub_download(repo_id='heathcliff01/Kaloscope2.0', filename='class_mapping.csv')
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        画师映射 = {int(row['class_id']): row['class_name'] for row in csv.DictReader(f)}
    return model, 画师映射, transform


def 识别画师(img: Image.Image, top_k: int = 5) -> list[tuple[str, float]]:
    model, 画师映射, 预处理 = _Kaloscope_2_model()
    input_tensor = 预处理(img.convert('RGB')).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    probs, indices = torch.topk(probabilities, top_k)
    return [(画师映射.get(i, str(i)), float(p)) for p, i in zip(probs.tolist(), indices.tolist())]


def 标签源():
    while True:
        for a in ['1girl']:
            for b in ['white hair', 'black hair', 'blonde hair']:
                for c in ['twintails', 'long hair']:
                    for d in ['indoors', 'outdoors', None]:
                        yield [i for i in (a, b, c, d) if i]


def 评测pipeline画风(pipe, validation_infer_steps, n_iter, guidance_scale):
    要测的画师 = json.load(open('赠品/好画师.json', encoding='utf8'))[:n_iter]

    pipe.transformer.to('cpu')
    clean()
    pipe.text_encoder.to('cuda')
    prompts_embeds = []
    neg_embeds = [i.to('cpu') for i in pipe.encode_prompt("")] if guidance_scale != 1.0 else (None, None)
    for index, tags, 画师 in zip(range(n_iter), 标签源(), 要测的画师):
        prompt = ', '.join([i.replace('_', ' ') for i in (tags + ['@' + 画师])])
        print('看看', prompt)
        prompts_embeds.append([i.to('cpu') for i in pipe.encode_prompt(prompt)])
    pipe.text_encoder.to('cpu')
    clean()
    pipe.transformer.to('cuda')

    总分 = 0
    for index, (画师, (pe, pm)) in enumerate(tqdm([*zip(要测的画师, prompts_embeds, strict=True)], ncols=70, desc='评测画风')):
        image = pipe(
            prompt_embeds=pe.to('cuda'),
            prompt_attention_mask=pm.to('cuda'),
            negative_prompt_embeds=neg_embeds[0].to('cuda') if neg_embeds[0] is not None else None,
            negative_prompt_attention_mask=neg_embeds[1].to('cuda') if neg_embeds[1] is not None else None,
            generator=torch.Generator(device='cpu').manual_seed(index),
            num_inference_steps=validation_infer_steps,
            guidance_scale=guidance_scale,
            uncond_skip_layers=[22],
            width=768,
            height=1024,
        ).images[0]
        预测 = 识别画师(image)
        for i, (预测画师, 概率) in enumerate(预测):
            if 预测画师 == 画师:
                总分 += ((len(预测) - i) + 1) / (len(预测) + 1)
    return 总分 / len(要测的画师)


def validation(global_step, accelerator, vae, text_encoder, tokenizer, transformer, scheduler, trackers, resume_global_step, validation_prompt_list, validation_infer_steps, validation_infer_cfg, validation_n_iter=50):
    if accelerator.is_main_process:
        with torch.no_grad():
            transformer.eval()
            transformer.to('cpu')
            clean()
            text_encoder.to(accelerator.device)
            pipeline = transformer._pipeline_cls(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                vae=vae,
                scheduler=type(scheduler).from_config(scheduler.config),
            )
            pipeline.set_progress_bar_config(disable=True)
            prompts_embeds = [[i.to('cpu') for i in pipeline.encode_prompt(prompt)] for prompt in validation_prompt_list[:validation_n_iter]]
            neg_embeds = [i.to('cpu') for i in pipeline.encode_prompt("")] if validation_infer_cfg != 1.0 else (None, None)
            text_encoder.to('cpu')
            clean()
            transformer.to(accelerator.device)
            images = [
                pipeline(
                    prompt_embeds=pe.to(accelerator.device),
                    prompt_attention_mask=pm.to(accelerator.device),
                    negative_prompt_embeds=neg_embeds[0].to(accelerator.device) if neg_embeds[0] is not None else None,
                    negative_prompt_attention_mask=neg_embeds[1].to(accelerator.device) if neg_embeds[1] is not None else None,
                    generator=torch.Generator(device='cpu').manual_seed(100 + i),
                    num_inference_steps=validation_infer_steps,
                    guidance_scale=validation_infer_cfg,
                    uncond_skip_layers=[22],
                    width=768,
                    height=1024,
                ).images[0] for i, (pe, pm) in enumerate(prompts_embeds)
            ]
            del prompts_embeds
            clean()
            for tracker in trackers:
                if tracker.name == "tensorboard":
                    add_image_jpeg(tracker.writer, f"validation-{validation_infer_steps}step-{validation_infer_cfg}cfg", np.concatenate([np.asarray(img) for img in images], axis=1), global_step)
            if global_step > resume_global_step:
                分数 = 评测pipeline画风(pipeline, validation_infer_steps=validation_infer_steps, n_iter=validation_n_iter, guidance_scale=validation_infer_cfg)
                accelerator.log({f"画风分数-{validation_n_iter}-step{validation_infer_steps}-cfg{validation_infer_cfg}": 分数}, step=global_step)
                clean()
            transformer.train()


def 加载模型(path) -> tuple:
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", Path(path) / "pipeline_moss_image.py")
    moss_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moss_mod)

    class TrainableMOSSImageTransformer(moss_mod.MOSSImageTransformer, PeftAdapterMixin):
        pass

    class TrainableMOSSImagePipeline(moss_mod.MOSSImagePipeline):
        @torch.no_grad()
        def encode_prompt(self, prompt, max_sequence_length=4096):
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            rows = []
            for value in prompts:
                formatted = self.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': value}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True)
                inputs = self.tokenizer(formatted, return_tensors='pt', truncation=True,
                                        max_length=max_sequence_length).to(self.text_encoder.device)
                output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
                rows.append(output.last_hidden_state[0])
            length = max(row.shape[0] for row in rows)
            embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
            mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
            for index, row in enumerate(rows):
                embeds[index, :row.shape[0]] = row
                mask[index, :row.shape[0]] = True
            return embeds, mask

    tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    transformer = TrainableMOSSImageTransformer.from_pretrained(path, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer._pipeline_cls = TrainableMOSSImagePipeline
    vae = AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16)
    noise_scheduler = moss_mod.MOSSImageScheduler.from_pretrained(path, subfolder="scheduler")

    for m in [text_encoder, transformer, vae]:
        m.requires_grad_(False)

    return tokenizer, text_encoder, transformer, vae, noise_scheduler


def conditional_loss(model_pred: Tensor, target: Tensor, reduction: str = "mean", loss_type: str = "l2", huber_c: float = 0.1):
    if loss_type == "l2":
        loss = F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type in ("huber", "huber_scheduled"):
        loss = huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
    return loss


@torch.no_grad()
def vae_encode(vae, pixel_values: Tensor) -> Tensor:
    z = vae.encode(pixel_values.to(vae.device, dtype=vae.dtype)).latent_dist.mode().float()
    b, c, hh, ww = z.shape
    z = z.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, hh // 2, ww // 2)
    mean = vae.bn.running_mean.float().view(1, -1, 1, 1)
    std = (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1)
    return (z - mean) / std


def 读取配对数据集(data_dir: str, 幅度: str, seed: int = 1):
    data_dir = Path(data_dir)
    r = random.Random(seed)
    transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    pairs = []
    for pkl_file in data_dir.glob('*.pkl'):
        sdxl_file = pkl_file.with_name(f'{pkl_file.stem}_sdxl_{幅度}.png')
        if sdxl_file.is_file():
            pairs.append((pkl_file, sdxl_file))
    print(f'成功匹配到{len(pairs)}个样本对！')
    while True:
        r.shuffle(pairs)
        for pkl_file, sdxl_file in pairs:
            try:
                with open(pkl_file, 'rb') as f:
                    pkl_data = pickle.load(f)
                img = Image.open(sdxl_file).convert('RGB')
                pixel_values = transform(transforms.ToTensor()(img)).unsqueeze(0)
                yield {
                    'pixel_values': pixel_values,
                    'noise': pkl_data['noise'],
                    'prompt_embeds': pkl_data['prompt_embeds'],
                    'prompt_embeds_mask': pkl_data['prompt_embeds_mask'],
                }
            except Exception:
                continue


def main(
    pretrained_model_name_or_path: str = "C:/Users/Administrator/Desktop/MOSS-Image-SFT",
    train_data_dir: str = "X:/RU9M_moss_小喙5.0",
    validation_prompt_list: str | list[str] = '1girl',
    validation_infer_steps: int = 20,
    validation_infer_cfg: float = 5.0,
    validation_interval: int = 100,
    validation_n_iter: int = 50,
    output_dir: str = "超rum_moss",
    logging_dir: str = None,
    seed: int = None,
    max_train_steps: int = 10000,
    cycle_steps: int = 3000,
    checkpointing_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    lr: float = 1e-4,
    lr_scheduler: str = "cosine_with_restarts",
    lr_warmup_steps: int = 50,
    lr_cosine_min: float = 0.1,
    optimizer: str = "adam",
    muon_lr_scale: float = 10,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    mixed_precision: str = "bf16",
    loss_type: str = "l2",
    huber_c: float = 0.1,
    rank: int = 32,
    alpha: int = None,
    time_min: int = 0,
    time_max: int = 1000,
    resume_from_checkpoint: str = 'latest',
    shift: float = None,
    幅度: str = '0.8',
    reset_noise_rate: float = 0,
):
    alpha = alpha or rank // 2
    if seed is None:
        seed = int(time.time()) % 100
    metadata = {k: v for k, v in locals().items() if isinstance(v, (float, int, str, list, tuple)) and not k.startswith('_')}
    metadata['torch_version'] = torch.__version__
    metadata['device_name'] = torch.cuda.get_device_name()
    if logging_dir is None:
        logging_dir = os.path.join(output_dir, 'logs')
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with='tensorboard',
        project_config=ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(seed)
    if isinstance(validation_prompt_list, str):
        validation_prompt_list = validation_prompt_list.split(';')
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    特 = [哈(train_data_dir), 哈(pretrained_model_name_or_path), optimizer, str(muon_lr_scale) * (optimizer == 'muon'), f'lr{lr}', mixed_precision, lr_scheduler, f'{round(lr_cosine_min, 3)}_{cycle_steps}' * (lr_scheduler == 'cosine_with_restarts'), f'lora{rank}_{alpha}', f'time{time_min}_{time_max}', f'decay{adam_weight_decay}', f'shift{shift}', f'幅{幅度}', seed, f'rn{reset_noise_rate}' * (reset_noise_rate > 0)]
    特征 = '-'.join([str(i) for i in 特 if i != ''])

    if os.path.isdir(os.path.join(output_dir, 特征, f'checkpoint-{max_train_steps}')):
        print('跳过', 特征)
        return

    checkpoint_dir = os.path.join(output_dir, 特征)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(Path(checkpoint_dir) / 'metadata.json', 'w', encoding='utf8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print('开始训练: ', str((Path(checkpoint_dir) / 'metadata.json').resolve()))

    tokenizer, text_encoder, transformer, vae, noise_scheduler = 加载模型(pretrained_model_name_or_path)
    vae.to(accelerator.device)
    transformer.to(accelerator.device)

    transformer.add_adapter(LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"],
    ))

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    def save_model_hook(models, weights, out_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(unwrap_model(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                if weights:
                    weights.pop()
            Krea2Pipeline.save_lora_weights(out_dir, transformer_lora_layers=transformer_lora_layers_to_save)

    def load_model_hook(models, in_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                transformer_ = unwrap_model(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")
        lora_state_dict = Krea2Pipeline.lora_state_dict(in_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        assert not incompatible_keys or not incompatible_keys.unexpected_keys, f'不对，{incompatible_keys.unexpected_keys}'
        cast_training_params([transformer_], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if gradient_checkpointing:
        for layer in transformer.layers:
            orig_forward = layer.forward
            def make_ckpt_forward(f):
                def ckpt_forward(x, rope, modulation):
                    if torch.is_grad_enabled():
                        return torch.utils.checkpoint.checkpoint(f, x, rope, modulation, use_reentrant=False)
                    return f(x, rope, modulation)
                return ckpt_forward
            layer.forward = make_ckpt_forward(orig_forward)

    cast_training_params([transformer], dtype=torch.float32)

    optimizer_obj = 生成optimizer(optimizer, transformer, adam_beta1, adam_beta2, adam_weight_decay, adam_epsilon, lr, lr * muon_lr_scale)

    scheduler参数 = dict(
        optimizer=optimizer_obj,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps * 1000,
        num_cycles=round(max_train_steps * gradient_accumulation_steps * 1000 / cycle_steps),
        cosine_min=lr_cosine_min,
    )
    assert lr_scheduler == 'cosine_with_restarts'
    scheduler_obj = cosine_with_restart_scheduler改(**scheduler参数)

    transformer, optimizer_obj, scheduler_obj = accelerator.prepare(
        transformer, optimizer_obj, scheduler_obj
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(特征)

    resume_global_step = global_step = 0
    if resume_from_checkpoint == 'latest':
        if 候选checkpoint := [*Path(checkpoint_dir).glob('checkpoint-*')]:
            resume_from_checkpoint = str(max(候选checkpoint, key=lambda i: int(i.stem.split('-')[-1])))
        else:
            resume_from_checkpoint = None
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)
        resume_global_step = global_step = int(resume_from_checkpoint.split("-")[-1])
        if True:
            for group in optimizer_obj.param_groups:
                if not group.get("use_muon", False):
                    group["lr"] = group["initial_lr"] = lr
                else:
                    group["lr"] = group["initial_lr"] = lr * muon_lr_scale
            scheduler_obj = cosine_with_restart_scheduler改(**scheduler参数)
            scheduler_obj = accelerator.prepare(scheduler_obj)

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    数据源 = buffered_iterator(读取配对数据集(train_data_dir, 幅度=幅度, seed=seed + 1), maxsize=20)
    transformer.train()

    while global_step < max_train_steps:
        with 计时(accelerator, global_step, '取数据'):
            batch = next(数据源)

        if random.random() < reset_noise_rate:
            batch['noise'] = torch.randn_like(batch['noise'])
        try:
            with 计时(accelerator, global_step, 'VAE', sync=True):
                x0 = vae_encode(vae, batch["pixel_values"]).to(accelerator.device, dtype=torch.bfloat16)
            prompt_embeds = batch['prompt_embeds'].to(accelerator.device, dtype=torch.bfloat16)
            prompt_embeds_mask = batch['prompt_embeds_mask'].to(accelerator.device)
            noise = batch['noise'].to(accelerator.device, dtype=torch.bfloat16)
        except Exception:
            logging.exception('准备输入时出了问题:')
            continue

        with accelerator.accumulate(transformer), 计时(accelerator, global_step, '全', sync=True):
            bsz, c, latent_h, latent_w = x0.shape
            t = torch.randint(time_min, time_max, (bsz,), device=x0.device).float() / 1000.0

            if shift is not None:
                current_alpha = shift
            else:
                image_seq_len = latent_h * latent_w
                mu = calculate_shift(
                    image_seq_len,
                    base_seq_len=noise_scheduler.config.get("base_image_seq_len", 256),
                    max_seq_len=noise_scheduler.config.get("max_image_seq_len", 4096),
                    base_shift=noise_scheduler.config.get("base_shift", 0.5),
                    max_shift=noise_scheduler.config.get("max_shift", 1.15),
                )
                current_alpha = math.exp(mu)

            sigmas = (current_alpha * t) / (1 + (current_alpha - 1) * t)
            timesteps = sigmas * 1000.0
            sigmas = sigmas.view(bsz, 1, 1, 1)

            noisy_model_input = (1.0 - sigmas) * x0 + sigmas * noise
            target = noise - x0

            with 计时(accelerator, global_step, '正向', sync=True):
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_embeds_mask,
                    return_dict=False,
                )[0]
            loss = conditional_loss(model_pred.float(), target.float(), reduction="mean", loss_type=loss_type, huber_c=huber_c)
            del noisy_model_input, target, model_pred
            del x0, prompt_embeds, prompt_embeds_mask, noise

            if global_step < 100 and accelerator.is_main_process:
                with open(f'{checkpoint_dir}/loss_log.txt', 'a', encoding='utf8') as f:
                    print(f'{global_step}: {loss.detach().item():.8f}', file=f)
            with 计时(accelerator, global_step, '反向', sync=True):
                accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_([*filter(lambda p: p.requires_grad, transformer.parameters())], max_grad_norm)
            with 计时(accelerator, global_step, 'optimizer', sync=True):
                optimizer_obj.step()
                scheduler_obj.step()
                optimizer_obj.zero_grad()

        if accelerator.sync_gradients:
            if validation_interval and (global_step % validation_interval == 0 or global_step in [validation_interval // 2, validation_interval // 4]):
                with 计时(accelerator, global_step, 'validation', sync=True):
                    for validation_infer_cfg, validation_infer_steps in [(5.0, 20)]:
                        validation(
                            global_step=global_step,
                            accelerator=accelerator,
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            transformer=unwrap_model(transformer),
                            scheduler=noise_scheduler,
                            trackers=accelerator.trackers,
                            resume_global_step=resume_global_step,
                            validation_prompt_list=validation_prompt_list,
                            validation_infer_steps=validation_infer_steps,
                            validation_infer_cfg=validation_infer_cfg,
                            validation_n_iter=validation_n_iter,
                        )
            logs = {"loss": loss.detach().item(), "lr": scheduler_obj.get_last_lr()[0], "grad_norm": grad_norm.item(), "t": timesteps[0]}
            for a, b in itertools.pairwise([1000, 900, 800, 0]):
                if a >= timesteps[0] >= b:
                    logs = logs | {
                        f'grad_norm_{a}到{b}': grad_norm.item(),
                        f'loss_{a}到{b}': loss.item(),
                    }
                    accelerator.log(logs, step=global_step)
                    break
            del loss
            progress_bar.update(1)
            global_step += 1
            if accelerator.is_main_process and global_step % checkpointing_steps == 0:
                save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logging.info(f"Saved state to {save_path}")


def ember(config={}, **kwargs):
    if isinstance(config, str):
        with open(config, 'r', encoding='utf8') as f:
            config = json.load(f)
    return main(**{k: v for k, v in (config | kwargs).items() if k not in ('torch_version', 'device_name', 'quantization')})


if __name__ == "__main__":
    fire.Fire(ember)


# python .\train_moss.py --config=moss默认参数.json
# python .\train_moss.py --config=moss默认参数.json --pretrained_model_name_or_path=R:/models/MOSS-Image-SFT --reset_noise_rate=0.4
