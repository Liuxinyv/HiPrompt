import argparse
import json

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", type=str,
                        default="Lin-Chen/ShareCaptioner")
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--images-file", type=str, default="images_to_describe.json",
                        help="a list, each element is a string for image path")
    parser.add_argument("--save-path", type=str, default="captions.json")
    args = parser.parse_args()
    return args


def auto_configure_device_map(num_gpus):
    # visual_encoder 算4层
    # internlm_model.model.embed_tokens 占用1层
    # norm 和 lm_head 占用1层
    # transformer.layers 占用 32 层
    # 总共34层分配到num_gpus张卡上
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'visual_encoder': 0,
        'ln_vision': 0,
        'Qformer': 0,
        'internlm_model.model.embed_tokens': 0,
        'internlm_model.model.norm': 0,
        'internlm_model.lm_head': 0,
        'query_tokens': 0,
        'flag_image_start': 0,
        'flag_image_end': 0,
        'internlm_proj': 0,
    }

    used = 6
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'internlm_model.model.layers.{i}'] = gpu_target
        used += 1

    return device_map


def eval_model_share(model,imgs,device):
    num_gpus=1
    if num_gpus > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(num_gpus)
        model = dispatch_model(model, device_map=device_map)
    else:
        model.cuda()
    batch_size=8
    seg1 = '<|User|>:'
    seg2 = f'Analyze the image in a comprehensive and detailed manner.{model.eoh}\n<|Bot|>:'
    seg_emb1 = model.encode_text(seg1, add_special_tokens=True)
    seg_emb2 = model.encode_text(seg2, add_special_tokens=False)
    part_len = len(imgs)
    captions = []
    chunk_size = part_len//batch_size
    if part_len % batch_size != 0:
        chunk_size += 1

    for i in range(chunk_size):
        print(f'{i}/{chunk_size}')
        subs = []
        for j in range(batch_size):
            if i*batch_size+j < len(imgs):
                img_path = imgs[i*batch_size+j]
                image=img_path
                # image = Image.open(img_path).convert("RGB")
                subs.append(model.vis_processor(image).unsqueeze(0))
        print(len(subs))
        if len(subs) == 0:
            break
        subs = torch.cat(subs, dim=0).cuda()
        tmp_bs = subs.shape[0]
        tmp_seg_emb1 = seg_emb1.repeat(tmp_bs, 1, 1)
        tmp_seg_emb2 = seg_emb2.repeat(tmp_bs, 1, 1)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                subs = model.encode_img(subs)
                input_emb = torch.cat(
                    [tmp_seg_emb1, subs, tmp_seg_emb2], dim=1).half()
                input_emb=input_emb.to(dtype=torch.float16)
                out_embeds = model.internlm_model.generate(inputs_embeds=input_emb,
                                                           max_length=500,
                                                           num_beams=3,
                                                           min_length=1,
                                                           do_sample=True,
                                                           repetition_penalty=1.5,
                                                           length_penalty=1.0,
                                                           temperature=1.,
                                                           eos_token_id=model.tokenizer.eos_token_id,
                                                           num_return_sequences=1,
                                                           )
        for j, out in enumerate(out_embeds):
            out[out == -1] = 2
            response = model.decode_text([out])
            captions.append(response)
    return captions