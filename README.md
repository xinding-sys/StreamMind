<!-- <p align="center">
    <img src="https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/e7bc34e0e9a96d77947a75b54399d9f96ccf209d/assets/logo.png" width="150" style="margin-bottom: 0.2;"/>
<p> -->

<h3 align="center"><a href="https://arxiv.org/html/2503.06220" style="color:#9C276A">
StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition</a></h3>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2503.06220-AD1C18.svg?logo=arXiv)](https://arxiv.org/html/2503.06220v1) <br>

<!-- </h5> -->
<!-- 
<details open><summary>💡 Some other multimodal-LLM projects from our team may interest you ✨. </summary><p>
 may

<!-- > [**Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**](https://github.com/DAMO-NLP-SG/Video-LLaMA) <br>
> Hang Zhang, Xin Li, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/Video-LLaMA)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/Video-LLaMA.svg?style=social)](https://github.com/DAMO-NLP-SG/Video-LLaMA) [![arXiv](https://img.shields.io/badge/Arxiv-2306.02858-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2306.02858) <br>

> [**VCD: Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding**](https://arxiv.org/abs/2311.16922) <br>
> Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/DAMO-NLP-SG/VCD)  [![github](https://img.shields.io/github/stars/DAMO-NLP-SG/VCD.svg?style=social)](https://github.com/DAMO-NLP-SG/VCD)  [![arXiv](https://img.shields.io/badge/Arxiv-2311.16922-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.16922) <br> -->

<!-- </p></details> -->

<!-- <div align="center"><video src="https://github.com/DAMO-NLP-SG/VideoLLaMA2/assets/18526640/e0e7951c-f392-42ed-afad-b2c7984d3e38" width="800"></div>
 -->

## 📰 News
* **[2025.03.18]**  Release training, evaluation, and serving codes of StreamMind.
<div align="center">
    <img src="./assets/framework_v2.png" alt="overview">
</div>

## 🛠️ Requirements and Installation
Basic Dependencies:
* Python >= 3.10
* Pytorch >= 2.5.1
* CUDA Version >= 11.8
* transformers >= 4.44.2 (for mistral tokenizer)
* tokenizers >= 0.19.1 (for mistral tokenizer)

**[Online Mode]** Install required packages (better for development):
```bash
git clone https://github.com/xinding-sys/StreamMind
cd StreamMind
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation
```

## 🚀 Main Results

### Streaming Dialogue
<div align="center">
    <img src="./assets/result1.png" alt="overview">
</div>
<div align="center">
    <img src="./assets/result2.png" alt="overview">
</div>

### Offline benchmark
<div align="center">
    <img src="./assets/result3.png" alt="overview">
</div>
<div align="center">
    <img src="./assets/result4.png" alt="overview">
</div>


## 🗝️ Training & Evaluation

### Quick Start

<!-- <!-- <!-- 1. Training Data Structure: -->
```bash
StreamMind
├── Online_datasets
│   ├── ego4d
|   |   ├── v2 
|   |   |   ├── annotations 
|   |   |   ├── full_scale
│   ├── MatchTime
|   |   ├── SN-caption 
|   |   ├── Video
├── Offline_datasets
│   ├── videollava_pt
|   |   ├── llava_image/ 
|   |   ├── valley/      
|   |   └── valley_llavaimage.json 
│   ├── videollava_sft
|   |   ├── llava_image_tune/  
|   |   ├── videochatgpt_tune/ 
|   |   └── videochatgpt_llavaimage_tune.json 
``` -->
1. Command:
```bash
# Streammind train stage 1
bash scripts/custom/finetune_stage1.sh
# Streammind train stage 2
bash scripts/custom/finetune_stage2.sh
# Streammind evaluate
bash scripts/custom/eval/evaluate.sh
```

## 📑 Citation

If you find StreamMind useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{ding2025streammind,
  title={StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition},
  author={Ding, Xin and Wu, Hao and Yang, Yifan and Jiang, Shiqi and Bai, Donglin and Chen, Zhibo and Cao, Ting},
  journal={arXiv preprint arXiv:2503.06220},
  year={2025}
}
```

## 👍 Acknowledgement
The codebase of StreamMind is adapted from [**VideoLLaMA 2**](https://github.com/DAMO-NLP-SG/VideoLLaMA2), We are also grateful for the following projects our StreamMind arise from:
* [**LLaMA 2**](https://github.com/meta-llama/llama), [**Mistral-7B**](https://mistral.ai/news/announcing-mistral-7b/), [**OpenAI CLIP**](https://openai.com/index/clip/), [**Honeybee**](https://github.com/kakaobrain/honeybee).
* [**Video-ChatGPT**](https://github.com/mbzuai-oryx/Video-ChatGPT), [**Video-LLaVA**](https://github.com/PKU-YuanGroup/Video-LLaVA). 
* [**WebVid**](https://github.com/m-bain/webvid), [**Panda-70M**](https://github.com/snap-research/Panda-70M), [**LanguageBind**](https://github.com/PKU-YuanGroup/LanguageBind), [**InternVid**](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid).
* [**VideoChat2**](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), [**Valley**](https://github.com/RupertLuo/Valley), [**VTimeLLM**](https://github.com/huangb23/VTimeLLM), [**ShareGPT4V**](https://sharegpt4v.github.io/).


## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file.
The service is a research preview intended for **non-commercial use ONLY**, subject to the model Licenses of LLaMA and Mistral, Terms of Use of the data generated by OpenAI, and Privacy Practices of ShareGPT. Please get in touch with us if you find any potential violations.
