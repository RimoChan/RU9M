import os
import pickle
import random
from pathlib import Path

import fire
from tqdm import tqdm

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, Krea2Pipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
from imgutils.tagging import get_wd14_tags

from data import 读取数据集, 分离人数标签
import 省电

省电.开启省电模式()


def prompt_pre_process(d):
    a, b = 分离人数标签(d['tag_string_general'].split(' '))
    if random.random() < 0.4:
        保留标签数 = min(len(b), random.randint(3, 5))
    else:
        保留标签数 = len(b)
        if 保留标签数 > 5:
            保留标签数 = (5 * 2 + 保留标签数) / 3
    新b = random.sample(b, min(30, int(保留标签数)))
    for 要t in ('indoors', 'outdoors'):
        if 要t in b and 要t not in 新b:
            新b.append(要t)
    print(len(b), len(新b))
    random.shuffle(新b)
    d['tag_string_general'] = ' '.join([*a, *新b])
    return d


坏标签 = {'1boy', '2boys', 'transparent background', 'oekaki', 'monochrome', 'greyscale', 'greyscale with colored background', 'artist name', 'signature', 'twitter username', 'patreon username', 'speech bubble', 'thought bubble', 'limited palette', 'sketch', 'typo', 'character name', 'english text', 'dated', 'copyright name', 'signature', 'comic', 'qr code', 'watermark', 'sample watermark'}


def krea(
    data_dir='X:/image_balance大出_2024',
    out_dir='output',
    krea_model='R:/models/Krea-2-Turbo',
    moss_model='C:/Users/Administrator/Desktop/MOSS-Image-SFT',
    model_type='krea',
    num_samples=10**8,
    seed=9,
):
    random.seed(seed)
    妙蛙 = random.randint(0, 10**7)
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir)

    源 = 读取数据集(data_dir, size_min=896, size_max=1152, 替换画师概率=1, prompt_pre_process=prompt_pre_process, seed=seed, ep=1)

    if model_type == 'moss':
        torch.set_num_threads(12)
        pipe = DiffusionPipeline.from_pretrained(
            moss_model,
            custom_pipeline="pipeline_moss_image",
            trust_remote_code=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        )
        pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.text_encoder.to("cpu")

        def custom_encode_prompt(prompt, max_sequence_length=4096):
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            rows = []
            for value in prompts:
                formatted = pipe.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': value}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True)
                inputs = pipe.tokenizer(formatted, return_tensors='pt', truncation=True,
                                        max_length=max_sequence_length).to(pipe.text_encoder.device)
                with torch.inference_mode():
                    output = pipe.text_encoder(**inputs, use_cache=False, return_dict=True)
                rows.append(output.last_hidden_state[0])
            length = max(row.shape[0] for row in rows)
            embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
            mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
            for index, row in enumerate(rows):
                embeds[index, :row.shape[0]] = row
                mask[index, :row.shape[0]] = True
            return embeds, mask

        pipe.encode_prompt = custom_encode_prompt
        pipe.set_progress_bar_config(disable=True)
    else:
        pipe = Krea2Pipeline.from_pretrained(krea_model, torch_dtype=torch.bfloat16)
        pipe.enable_group_offload(
            onload_device=torch.device('cuda'),
            offload_type='leaf_level',
            use_stream=True,
            exclude_modules=['transformer', 'vae']
        )
        pipe.set_progress_bar_config(disable=True)

    已刷 = 0
    with torch.inference_mode():
        for i, data in enumerate(tqdm(源, desc=model_type, ncols=60, total=num_samples)):
            if 已刷 >= num_samples:
                break
            prompt = data['prompts'][0]
            if {*prompt.split(', ')} & 坏标签:
                continue
            if '1girl' not in prompt:
                continue
            _, _, h, w = data['pixel_values'].shape
            img_hash = data['image_hash']

            png_path = out_path / f'{i}_{img_hash}_{model_type}.png'
            pkl_path = out_path / f'{i}_{img_hash}.pkl'
            txt_path = out_path / f'{i}_{img_hash}.txt'
            if pkl_path.is_file():
                已刷 += 1
                continue

            if model_type == 'moss':
                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt)
                noise = torch.randn((1, pipe.transformer.config.latent_channels, h // 16, w // 16), dtype=torch.float32, generator=torch.Generator("cpu").manual_seed(妙蛙 + i))
                img = pipe(
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_embeds_mask,
                    latents=noise,
                    height=h,
                    width=w,
                    num_inference_steps=20,
                    guidance_scale=5.0,
                    uncond_skip_layers=[22],
                ).images[0]
            else:
                prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt)
                noise = torch.randn((1, 16, h // 8, w // 8), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(妙蛙 + i))
                packed_noise = pipe._pack_latents(noise, 1, 16, h // 8, w // 8)

                img = pipe(
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    latents=packed_noise,
                    height=h,
                    width=w,
                    num_inference_steps=8,
                    guidance_scale=0.0
                ).images[0]
            pkl_data = {
                'noise': noise.cpu(),
                'prompt_embeds': prompt_embeds.cpu(),
                'prompt_embeds_mask': prompt_embeds_mask.cpu(),
                'prompt': prompt,
                'height': h,
                'width': w,
            }

            img.save(png_path)
            with open(pkl_path, 'wb') as f:
                pickle.dump(pkl_data, f)
            with open(txt_path, 'w') as f:
                f.write(prompt)
            已刷 += 1

def sdxl(
    out_dir='output',
    sdxl_model='R:/stable-diffusion-webui-master/models/Stable-diffusion/waiNSFWIllustrious_v140.safetensors',
    strength=0.8,
    seed=9,
    注入发色=False,
    后缀='krea',
):
    random.seed(seed)
    out_path = Path(out_dir)

    pipe_sdxl = StableDiffusionXLImg2ImgPipeline.from_single_file(
        sdxl_model,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to('cuda')
    pipe_sdxl.set_progress_bar_config(disable=True)

    compel = Compel(truncate_long_prompts=False, tokenizer=[pipe_sdxl.tokenizer, pipe_sdxl.tokenizer_2], text_encoder=[pipe_sdxl.text_encoder, pipe_sdxl.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    if 注入发色:
        人物特征标签表, 发色 = _色()
    with torch.inference_mode():
        for i, pkl_path in tqdm(enumerate(sorted(out_path.glob('*.pkl'))), desc='sdxl', ncols=60):
            新模型png_path = pkl_path.with_name(f'{pkl_path.stem}_{后缀}.png')
            sdxl_png_path: Path = pkl_path.with_name(f'{pkl_path.stem}_sdxl_{strength}.png')
            if sdxl_png_path.is_file():
                continue
            with open(pkl_path, 'rb') as f:
                prompt = pickle.load(f)['prompt']
            prompt = prompt.replace(', @', ', ')
            if 注入发色:
                tags = prompt.split(', ')
                if 人s := {i.replace(' ', '_') for i in tags} & {*人物特征标签表}:
                    人 = [*人s][0]
                    if 人的发色 := {*人物特征标签表[人]} & 发色:
                        tags[-1:-1] = [[*人的发色][0].replace('_', ' ')]
                        # print('假面变化', prompt, ', '.join(tags))
                        prompt = ', '.join(tags)
                        # exit()
            img新 = Image.open(新模型png_path)
            conditioning, pooled = compel(prompt)
            img_sdxl = pipe_sdxl(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                image=img新,
                strength=strength,
                generator=torch.Generator(device='cuda').manual_seed(seed*100 + i*7)
            ).images[0]
            img_sdxl.save(sdxl_png_path)


def 清洗(out_dir, cuda_dir=None):
    if cuda_dir:
        import onnxruntime
        onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=cuda_dir)
    了 = 0
    out_path = Path(out_dir)
    images = [p for p in out_path.iterdir() if p.suffix.lower() == '.png']
    for img_path in tqdm(sorted(images), desc='清洗', ncols=60):
        _, tags, _ = get_wd14_tags(img_path, character_threshold=0.9, general_threshold=0.3, model_name='EVA02_Large')
        if set(tags) & {'multiple_views', '2girls', '3girls', '4girls', '1boy', '2boys', '3boys', '4boys'}:
            pkl_path = img_path.with_name(img_path.stem.rsplit('_krea', 1)[0].rsplit('_moss', 1)[0].rsplit('_sdxl', 1)[0] + '.pkl')
            img_path.unlink()
            if pkl_path.is_file():
                pkl_path.unlink()
            了 += 1
            print(f'清洗了{了}个！')


def _色():
    from waifu_sensor.v3 import _人阵, _人均值阵, 人标签
    人物特征标签表: dict[str, list[str]] = {}
    for 人, 均 in zip(_人阵, _人均值阵, strict=True):
        人物特征标签表[人.replace(' ', '_')] = [k.replace(' ', '_') for k, v in zip(人标签, 均) if v > 0.5 and ('background' not in k)]
    发色 = {'blue_hair', 'red_hair', 'light_blue_hair', 'white_hair', 'grey_hair', 'green_hair', 'purple_hair', 'black_hair', 'pink_hair', 'blonde_hair', 'aqua_hair', 'silver_hair', 'brown_hair', 'orange_hair', 'light_brown_hair', 'platinum_blonde_hair'}
    return 人物特征标签表, 发色


def 清洗发色(out_dir, cuda_dir=None):
    if cuda_dir:
        import onnxruntime
        onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=cuda_dir)
    人物特征标签表, 发色 = _色()
    out_path = Path(out_dir)
    images = [p for p in out_path.iterdir() if p.suffix.lower() == '.png' and 'sdxl' in p.stem]
    坏 = 0
    for img_path in tqdm(sorted(images), desc='清洗发色', ncols=60):
        prompt = open(out_path / (img_path.stem.split('_sdxl')[0]+'.txt')).read()
        tags = {i.replace(' ', '_') for i in prompt.split(', ')}
        if 人s := tags & {*人物特征标签表}:
            人 = [*人s][0]
            if 人的发色 := {*人物特征标签表[人]} & 发色:
                _, tags, _ = get_wd14_tags(img_path, general_threshold=0.3, model_name='EVA02_Large')
                if not (人的发色 & set(tags)):
                    坏 += 1
                    print('坏', img_path, 坏)
                    img_path.unlink()


if __name__ == '__main__':
    fire.Fire({
        'krea': krea,
        'sdxl': sdxl,
        '清洗': 清洗,
        '清洗发色': 清洗发色,
    })
