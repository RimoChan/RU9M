import os
os.environ['PYTORCH_ALLOC_CONF'] = os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'

import math
import time
import json
import pickle
import random
import logging
import itertools
from pathlib import Path

import fire
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, Qwen3VLModel, BitsAndBytesConfig
from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline, Krea2Transformer2DModel, AutoencoderKLQwenImage
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module

from common import clean, 生成optimizer, 哈, 哈哈, 计时, buffered_iterator, add_image_jpeg, cosine_with_restart_scheduler改, batched


def 评测pipeline人(pipe, validation_infer_steps, n_iter, guidance_scale, 人数=1):
    from imgutils.tagging import get_wd14_tags
    人频率1 = {k: v for k, v in json.loads(open('./赠品/人频率6000000~7000000.json').read()).items() if v > 24*1.3}
    人频率2 = {k: v for k, v in json.loads(open('./赠品/人频率1~6400000.json').read()).items() if v > 64*1.5}
    要测的人 = sorted(set(人频率1) | set(人频率2))
    记录 = []
    要测的人batched = [*batched(要测的人, 人数)]

    pipe.transformer.to('cpu')
    clean()
    pipe.text_encoder.to('cuda')
    prompts_embeds = []
    neg_embeds = pipe.encode_prompt("", device="cuda") if guidance_scale > 0 else (None, None)
    for index, 人组 in enumerate(要测的人batched[:n_iter]):
        人组 = [i.strip().replace('_', ' ') for i in 人组]
        if 人数 == 1:
            tags = ['1girl'] + 人组
        else:
            tags = [f'{人数}girls'] + 人组
        prompt=', '.join(tags)
        prompts_embeds.append([i.to('cpu') for i in pipe.encode_prompt(prompt, device='cuda')])
    pipe.text_encoder.to('cpu')
    clean()
    pipe.transformer.to('cuda')

    for index, (人组, (pe, pm)) in enumerate(tqdm([*zip(要测的人batched[:n_iter], prompts_embeds, strict=True)], ncols=70, desc='评测人')):
        seed = index*100
        image = pipe(
            prompt_embeds=pe.to('cuda'),
            prompt_embeds_mask=pm.to('cuda'),
            negative_prompt_embeds=neg_embeds[0],
            negative_prompt_embeds_mask=neg_embeds[1],
            generator=torch.Generator(device='cpu').manual_seed(seed),
            num_inference_steps=validation_infer_steps,
            guidance_scale=guidance_scale,
            width=512 + 128*人数,
            height=768,
        ).images[0]
        预测 = get_wd14_tags(image, character_threshold=0.3)[2]
        记录.append({
            '预测': 预测,
            '人组': 人组,
            'seed': seed,
        })
    好 = 0
    for d in 记录:
        for 人 in d['人组']:
            t = 人.replace(' ', '_')
            好 += d['预测'].get(t, 0)
            # print('对吗', d['人组'], d['预测'])
    return 好 / len(记录) / 人数


def validation(global_step, accelerator, vae, text_encoder, tokenizer, transformer, scheduler, trackers, resume_global_step, validation_prompt_list, validation_infer_steps, validation_infer_cfg, validation_n_iter=50):
    if accelerator.is_main_process:
        with torch.no_grad():
            transformer.to('cpu')
            clean()
            text_encoder.to(accelerator.device)
            pipeline = Krea2Pipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                vae=vae,
                scheduler=scheduler,
            )
            pipeline.set_progress_bar_config(disable=True)
            prompts_embeds = [pipeline.encode_prompt(prompt, device=accelerator.device) for prompt in validation_prompt_list[:validation_n_iter]]
            neg_embeds = pipeline.encode_prompt("", device=accelerator.device) if validation_infer_cfg > 0 else (None, None)
            text_encoder.to('cpu')
            clean()
            transformer.to(accelerator.device)
            images = [
                pipeline(
                    prompt_embeds=pe,
                    prompt_embeds_mask=pm,
                    negative_prompt_embeds=neg_embeds[0],
                    negative_prompt_embeds_mask=neg_embeds[1],
                    generator=torch.Generator(device=accelerator.device).manual_seed(100 + i),
                    num_inference_steps=validation_infer_steps,
                    guidance_scale=validation_infer_cfg,
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
                分数 = 评测pipeline人(pipeline, validation_infer_steps=validation_infer_steps, n_iter=validation_n_iter, guidance_scale=validation_infer_cfg)
                accelerator.log({f"人分数-{validation_n_iter}-step{validation_infer_steps}-cfg{validation_infer_cfg}": 分数}, step=global_step)
                clean()
            transformer.train()


def 加载模型(path, quantization="4bit") -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
    text_encoder = Qwen3VLModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    bnb_config = None
    device_map = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device_map = {"": torch.cuda.current_device()}

    transformer = Krea2Transformer2DModel.from_pretrained(
        path,
        subfolder="transformer",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    vae = AutoencoderKLQwenImage.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

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


def 打包latents(latents: Tensor) -> Tensor:
    bsz, c, _, h, w = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4)
    packed = latents.view(bsz, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(bsz, (h // 2) * (w // 2), c * 4)


def 解包latents(packed: Tensor, h: int, w: int, c: int = 16) -> Tensor:
    bsz = packed.shape[0]
    unpacked = packed.view(bsz, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(bsz, c, 1, h, w)


@torch.no_grad()
def vae_encode(vae, pixel_values: Tensor) -> Tensor:
    latents = vae.encode(pixel_values.unsqueeze(2).to(vae.device, dtype=vae.dtype)).latent_dist.sample()
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, 16, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, 16, 1, 1, 1)
    return (latents - latents_mean) * latents_std


def 读取配对数据集(data_dir: str, 幅度: str, seed: int = 1):
    data_dir = Path(data_dir)
    r = random.Random(seed)
    transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    pairs = []
    for pkl_file in data_dir.glob('*.pkl'):
        sdxl_file = pkl_file.with_name(f'{pkl_file.stem}_sdxl_{幅度}.png')
        krea_file = pkl_file.with_name(f'{pkl_file.stem}_krea.png')
        if sdxl_file.is_file() and krea_file.is_file():
            pairs.append((pkl_file, sdxl_file, krea_file))
    print(f'成功匹配到{len(pairs)}个样本对！')
    while True:
        r.shuffle(pairs)
        for pkl_file, sdxl_file, krea_file in pairs:
            try:
                with open(pkl_file, 'rb') as f:
                    pkl_data = pickle.load(f)
                img = Image.open(sdxl_file).convert('RGB')
                img_l = Image.open(krea_file).convert('RGB')
                pixel_values = transform(transforms.ToTensor()(img)).unsqueeze(0)
                pixel_values_l = transform(transforms.ToTensor()(img_l)).unsqueeze(0)
                noise = pkl_data['noise']
                if noise.ndim == 4:
                    noise = noise.unsqueeze(2)
                yield {
                    'pixel_values': pixel_values,
                    'pixel_values_l': pixel_values_l,
                    'noise': noise,
                    'prompt_embeds': pkl_data['prompt_embeds'],
                    'prompt_embeds_mask': pkl_data['prompt_embeds_mask'],
                }
            except Exception:
                continue


def main(
    pretrained_model_name_or_path: str = r"R:\models\Krea-2-Turbo",
    train_data_dir: str = "X:/RU9M_渴望",
    validation_prompt_list: str | list[str] = '1girl',
    validation_infer_steps: int = 8,
    validation_infer_cfg: float = 0.0,
    validation_interval: int = 100,
    output_dir: str = "超rum_krea2",
    logging_dir: str = None,
    seed: int = None,
    max_train_steps: int = 10000,
    cycle_steps: int = 3000,
    checkpointing_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    lr: float = 1e-4,
    lr_scheduler: str = "constant_with_warmup",
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
    quantization: str = "4bit",
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
    dpo_beta: float = 0,
    dpo_w_rate: float = 0.5,
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

    特 = [哈(train_data_dir), 哈(pretrained_model_name_or_path), optimizer, str(muon_lr_scale) * (optimizer == 'muon'), f'lr{lr}', mixed_precision, f'q_{quantization}', lr_scheduler, f'{round(lr_cosine_min, 3)}_{cycle_steps}' * (lr_scheduler == 'cosine_with_restarts'), f'lora{rank}_{alpha}', f'time{time_min}_{time_max}', f'decay{adam_weight_decay}', f'shift{shift}', f'幅{幅度}', seed, f'rn{reset_noise_rate}' * (reset_noise_rate > 0), f'dpo{dpo_beta}_{dpo_w_rate}' * (dpo_beta > 0)]
    特征 = '-'.join([str(i) for i in 特 if i != ''])

    if os.path.isdir(os.path.join(output_dir, 特征, f'checkpoint-{max_train_steps}')):
        print('跳过', 特征)
        return

    checkpoint_dir = os.path.join(output_dir, 特征)
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(Path(checkpoint_dir) / 'metadata.json', 'w', encoding='utf8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print('开始训练: ', str((Path(checkpoint_dir) / 'metadata.json').resolve()))

    tokenizer, text_encoder, transformer, vae, noise_scheduler = 加载模型(pretrained_model_name_or_path, quantization=quantization)
    vae.to(accelerator.device)

    transformer.add_adapter(LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "to_gate"],
    ))

    if quantization in ("fp8", "8bit"):
        for name, module in transformer.named_modules():
            if isinstance(module, torch.nn.Linear) and not any(p in name for p in ['lora_', 'text_fusion']):
                module.weight.data = module.weight.data.to(torch.float8_e4m3fn)
                def make_f(mod):
                    orig_f = mod.forward
                    def forward(input):
                        orig_weight = mod.weight
                        mod.weight = torch.nn.Parameter(orig_weight.data.to(input.dtype), requires_grad=False)
                        res = orig_f(input)
                        mod.weight = orig_weight
                        return res
                    return forward
                module.forward = make_f(module)
        transformer.to(accelerator.device)

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
        transformer.enable_gradient_checkpointing()

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
                if not group["use_muon"]:
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
                if dpo_beta > 0:
                    x0_l = vae_encode(vae, batch["pixel_values_l"]).to(accelerator.device, dtype=torch.bfloat16)
            prompt_embeds = batch['prompt_embeds'].to(accelerator.device, dtype=torch.bfloat16)
            prompt_embeds_mask = batch['prompt_embeds_mask'].to(accelerator.device)
            noise = batch['noise'].to(accelerator.device, dtype=torch.bfloat16)
        except Exception:
            logging.exception('准备输入时出了问题:')
            continue

        with accelerator.accumulate(transformer), 计时(accelerator, global_step, '全', sync=True):
            bsz, c, _, latent_h, latent_w = x0.shape
            t = torch.randint(time_min, time_max, (bsz,), device=x0.device).float() / 1000.0

            if shift is not None:
                current_alpha = shift
            else:
                image_seq_len = (latent_h // 2) * (latent_w // 2)
                mu = calculate_shift(
                    image_seq_len,
                    base_seq_len=noise_scheduler.config.get("base_image_seq_len", 256),
                    max_seq_len=noise_scheduler.config.get("max_image_seq_len", 6400),
                    base_shift=noise_scheduler.config.get("base_shift", 0.5),
                    max_shift=noise_scheduler.config.get("max_shift", 1.15),
                )
                current_alpha = math.exp(mu)

            sigmas = (current_alpha * t) / (1 + (current_alpha - 1) * t)
            timesteps = sigmas * 1000.0
            sigmas = sigmas.view(bsz, 1, 1, 1, 1)

            position_ids = Krea2Pipeline.prepare_position_ids(prompt_embeds.shape[1], latent_h // 2, latent_w // 2, accelerator.device)

            if dpo_beta > 0:
                noisy_w = (1.0 - sigmas) * x0 + sigmas * noise
                noisy_l = (1.0 - sigmas) * x0_l + sigmas * noise
                target_w = noise - x0
                target_l = noise - x0_l

                packed_input = 打包latents(torch.cat([noisy_w, noisy_l], dim=0))
                prompt_concat = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                mask_concat = torch.cat([prompt_embeds_mask, prompt_embeds_mask], dim=0)
                t_concat = torch.cat([timesteps, timesteps], dim=0) / 1000.0

                with 计时(accelerator, global_step, '正向', sync=True):
                    with torch.no_grad():
                        unwrap_model(transformer).disable_adapters()
                        ref_pred_packed = transformer(
                            hidden_states=packed_input,
                            encoder_hidden_states=prompt_concat,
                            timestep=t_concat,
                            position_ids=position_ids,
                            encoder_attention_mask=mask_concat,
                            return_dict=False,
                        )[0]
                        unwrap_model(transformer).enable_adapters()
                    model_pred_packed = transformer(
                        hidden_states=packed_input,
                        encoder_hidden_states=prompt_concat,
                        timestep=t_concat,
                        position_ids=position_ids,
                        encoder_attention_mask=mask_concat,
                        return_dict=False,
                    )[0]

                v_theta_w, v_theta_l = torch.chunk(解包latents(model_pred_packed, latent_h, latent_w).float(), 2, dim=0)
                v_ref_w, v_ref_l = torch.chunk(解包latents(ref_pred_packed, latent_h, latent_w).float(), 2, dim=0)

                loss_theta_w = conditional_loss(v_theta_w, target_w.float(), reduction="none", loss_type=loss_type, huber_c=huber_c).mean(dim=[1, 2, 3, 4])
                loss_theta_l = conditional_loss(v_theta_l, target_l.float(), reduction="none", loss_type=loss_type, huber_c=huber_c).mean(dim=[1, 2, 3, 4])
                loss_ref_w = conditional_loss(v_ref_w, target_w.float(), reduction="none", loss_type=loss_type, huber_c=huber_c).mean(dim=[1, 2, 3, 4])
                loss_ref_l = conditional_loss(v_ref_l, target_l.float(), reduction="none", loss_type=loss_type, huber_c=huber_c).mean(dim=[1, 2, 3, 4])

                delta_w = loss_theta_w - loss_ref_w
                delta_l = loss_theta_l - loss_ref_l
                loss_dpo = -F.logsigmoid(-0.5 * dpo_beta * (delta_w - delta_l)).mean()
                loss = loss_dpo / (0.5 * dpo_beta) + dpo_w_rate * loss_theta_w.mean()
                delta_w, delta_l, loss_dpo = delta_w.detach(), delta_l.detach(), loss_dpo.detach()
                del x0_l, noisy_w, noisy_l, target_w, target_l, packed_input, prompt_concat, mask_concat, t_concat, ref_pred_packed, model_pred_packed, v_theta_w, v_theta_l, v_ref_w, v_ref_l, loss_theta_w, loss_theta_l, loss_ref_w, loss_ref_l
            else:
                noisy_model_input = (1.0 - sigmas) * x0 + sigmas * noise
                target = noise - x0

                packed_input = 打包latents(noisy_model_input)

                with 计时(accelerator, global_step, '正向', sync=True):
                    model_pred_packed = transformer(
                        hidden_states=packed_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps / 1000.0,
                        position_ids=position_ids,
                        encoder_attention_mask=prompt_embeds_mask,
                        return_dict=False,
                    )[0]
                model_pred = 解包latents(model_pred_packed, latent_h, latent_w)
                loss = conditional_loss(model_pred.float(), target.float(), reduction="mean", loss_type=loss_type, huber_c=huber_c)
                del noisy_model_input, target, packed_input, model_pred_packed, model_pred
            del x0, prompt_embeds, prompt_embeds_mask, noise, position_ids

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
                    for validation_infer_cfg, validation_infer_steps in [(1.6, 8), (3, 16)]:
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
                        )
            logs = {"loss": loss.detach().item(), "lr": scheduler_obj.get_last_lr()[0], "grad_norm": grad_norm.item(), "t": timesteps[0]}
            if dpo_beta > 0:
                logs |= {
                    "loss_dpo": loss_dpo.detach().item(),
                    "implicit_acc": (delta_w < delta_l).float().mean().item(),
                    "delta_w": delta_w.mean().item(),
                    "delta_l": delta_l.mean().item(),
                }
                del loss_dpo, delta_w, delta_l
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
    return main(**{k: v for k, v in (config | kwargs).items() if k not in ('torch_version', 'device_name')})


if __name__ == "__main__":
    fire.Fire(ember)


# accelerate launch train_krea2.py --config=1.json --pretrained_model_name_or_path="R:\models\Krea-2-Raw"
# accelerate launch train_krea2.py --config=1.json --pretrained_model_name_or_path="R:\models\Krea-2-Raw" --rank=128 --adam_weight_decay=0.01
# accelerate launch train_krea2.py --config=1.json --pretrained_model_name_or_path="C:\Users\Administrator\Desktop\Krea-2-Raw" --rank=128 --adam_weight_decay=0.01 --optimizer=muon
# accelerate launch train_krea2.py --config=1.json --pretrained_model_name_or_path="R:\models\Krea-2-Raw" --lr=0.00003 --dpo_beta=1000
