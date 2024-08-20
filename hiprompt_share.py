


import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from hiprompt_sdxl_share import HiPromptSDXLPipeline
import glob 
import logging
import time
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import ControlNetModel, AutoencoderKL
from torchvision import transforms
import open_clip
import random
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--share", type=bool, default=False)
    parser.add_argument("--noise_decom", type=bool, default=False)
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--ngram", type=bool, default=False)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=bool, default=False)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--cosine_scale_3", type=float, default=1)
    parser.add_argument("--dataset_root", type=str, default='eval_texts')
    parser.add_argument("--view_args", default=None, type=str, nargs='+', help='Args to pass to views')
    parser.add_argument("--views_type", required=False, type=str, nargs='+', help='Name of views to use. See `get_views` in `views.py`.')
    parser.add_argument("--guidance_scale_fact", type=float, default=12.0)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default='./outputs_4k_llava/',
    )
    parser.add_argument(
        "--model_name_share",
        type=str,
        default='Lin-Chen/ShareCaptioner',
    )
    args = parser.parse_args()

    return args
args = parse_args()
def main():   
    args = parse_args()
    model_ckpt = args.model_ckpt
    pipe = HiPromptSDXLPipeline.from_pretrained(model_ckpt,torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
    seed=args.seed
    noise_decom=args.noise_decom
    reduction=args.reduction                                                                                                                                                    
    beta=args.beta          
    view_args=args.view_args
    views_type=args.views_type
    share = args.share
    height = args.height
    width = args.width
    steps = args.steps
    scale = args.scale
    guidance_scale=args.guidance_scale
    logging_dir = args.logging_dir
    cosine_scale_3=args.cosine_scale_3
    ngram=args.ngram
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_share, trust_remote_code=True)
    share_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_share, device_map="cuda", trust_remote_code=True).eval().half()    
    share_model.cuda()
    share_model.tokenizer = tokenizer
    if ngram:
        image_model, _, clip_image_processor, = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai') 
        image_model_2, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
        image_model = image_model.to(dtype=torch.float16, device="cuda") 
        image_model_2 = image_model_2.to(dtype=torch.float16, device="cuda") 
        clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
    else:
        image_model,clip_image_processor = None, None
        image_model_2 = None
        image_model = None
        image_model_2 = None
        clip_tokenizer = None
    pipe.enable_model_cpu_offload()
    prompt=args.prompt
    images = pipe(prompt, negative_prompt=negative_prompt, generator=generator,
                height=height, width=width, view_batch_size=16, stride=64,
                num_inference_steps=steps, guidance_scale = guidance_scale,
                cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=cosine_scale_3, sigma=0.8, 
                multi_decoder=True, show_image=True,
                share=share,
                share_model=share_model,
                image_lr = None,
                scale=scale,
                beta=beta,
                sample_path=prompt[:15],
                image_enc=image_model,
                clip_image_processor=clip_image_processor,
                clip_tokenizer=clip_tokenizer,
                image_enc_2=image_model_2,
                noise_decom=noise_decom,
                reduction=reduction,
                view_args=view_args,
                views_type=views_type,
                logging_dir=logging_dir,
                seed=seed,
                ngram=ngram
                )

if __name__ == "__main__":
    main()



