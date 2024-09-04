# HiPrompt: Tuning-free Higher-Resolution Generation with Hierarchical MLLM Prompts
<p align="center">
  <a href="https://github.com/Liuxinyv/HiPrompt">Xinyu Liu</a><sup>1</sup>,
  <a href="https://yingqinghe.github.io/">Yingqing He</a><sup>1</sup>, 
  <a href="https://guolanqing.github.io/">Lanqing Guo</a><sup>2</sup>, 
  <a href="https://github.com/Liuxinyv/HiPrompt">Xiang Li</a><sup>3</sup>, 
  <a href="https://jxbbb.github.io/">Bu Jin</a><sup>4</sup>, 
  <a href="https://github.com/Liuxinyv/HiPrompt">Peng Li</a><sup>1</sup>,
   <a href="https://github.com/Liuxinyv/HiPrompt">Yan Li</a><sup>1</sup>,
   <a href="https://github.com/Liuxinyv/HiPrompt">Chi-Min Chan</a><sup>1</sup>, 
  <a href="https://cqf.io/">Qifeng Chen</a><sup>1</sup>,
   <a href="https://github.com/Liuxinyv/HiPrompt">Wei Xue</a><sup>1</sup>,
    <a href="https://github.com/Liuxinyv/HiPrompt">Wenhan Luo</a><sup>1</sup>,
   <a href="https://github.com/Liuxinyv/HiPrompt">Qingfeng Liu</a><sup>1</sup>,
    <a href="https://github.com/Liuxinyv/HiPrompt">QiYike Guo</a><sup>1</sup>
  <br><br>
  <sup>1</sup>Hong Kong University of Science and Technology<br>
  <sup>2</sup>Nanyang Technological University<br>
  <sup>3</sup>Tsinghua University<br>
 <sup>4</sup>University of Chinese Academy of Sciences</span> <br>
</p>
<div align="center">
  <a href="https://liuxinyv.github.io/HiPrompt/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://github.com/Liuxinyv/HiPrompt"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;

</div>

<img src="figures/illustration.jpg" width="800"/>

## 🔆 Abstract
 > The potential for higher-resolution image generation using pretrained diffusion models is immense, yet these models often struggle with issues of object repetition and structural artifacts especially when scaling to 4K resolution and higher. We figure out that the problem is caused by that, a single prompt for the generation of multiple scales provides insufficient efficacy. In response, we propose HiPrompt, a new tuning-free solution that tackles the above problems by introducing hierarchical prompts. The hierarchical prompts offer both global and local guidance. Specifically, the global guidance comes from the user input that describes the overall content, while the local guidance utilizes patch-wise descriptions from  MLLMs to elaborately guide the regional structure and texture generation. Furthermore, during the inverse denoising process, the generated noise is decomposed into low- and high-frequency spatial components. These components are conditioned on multiple prompt levels, including detailed patch-wise descriptions and broader image-level prompts, facilitating prompt-guided denoising under hierarchical semantic guidance. It further allows the generation to focus more on local spatial regions and ensures the generated images maintain coherent local and global semantics, structures, and textures with high definition. Extensive experiments demonstrate that HiPrompt outperforms state-of-the-art works in higher-resolution image generation, significantly reducing object repetition and enhancing structural quality.
 > 
## 📝 Changelog
- __[2024.09.04]__: 🔥 Release paper.

## 🌰 More Examples
- __[2024.09.04]__: 🔥 Release paper.

## ⚙️ Setup:
```
conda create -n HiPrompt python=3.9
conda activate HiPrompt 
pip install -r requirements.txt
```
---

## 💫 Inference
```
python hiprompt_llava.py \
	--model_ckpt="stabilityai/stable-diffusion-xl-base-1.0" 、
    --height 4096 \
    --width 4096 \
    --logging_dir ${your-logging-dir} \
    --validation_prompt "a professional photograph of an astronaut riding a horse" \
    --llava true \
    --scale true \
    --cosine_scale_3 0.8 \
    --nd true \
    --reduction sum \
    --view_args 1.0 1.0 \
    --views_type low_pass high_pass \
    --guidance_scale_fact 10 \
    --beta 0.95 \
    --ngram true
```
by setting `--validation_prompt` to a prompt string or a path to your custom `.txt` file.
