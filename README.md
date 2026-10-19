<div align="center">
  <a href="https://youtu.be/MuwWxCOufQ8">
    <img src="https://img.youtube.com/vi/MuwWxCOufQ8/maxresdefault.jpg" alt="I Tested MOSS Image AI: The New King of Local Product Ads?">
  </a>
  <h3>📺 <a href="https://youtu.be/MuwWxCOufQ8">Watch the full video tutorial on YouTube</a></h3>
</div>

# 🎨 MOSS Image: 8-Step Local AI Ad Studio

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Model: OpenMOSS-Team/MOSS-Image-SFT](https://img.shields.io/badge/Model-MOSS--Image--SFT-00e676.svg)](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
[![Acceleration: 8--Step DMD](https://img.shields.io/badge/DMD-8--Step_Distillation-38bdf8.svg)](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)

An end-to-end commercial product advertising studio powered by **OpenMOSS's 5B Diffusion Transformer (DiT)** and **8-step DMD (Distribution Matching Distillation)**. Takes minimal user input for a single product and generates 5 diverse, photorealistic production ad environments with clean label typography and spatial bounding boxes.

---

## ⚡ 3-Step Visual Workflow

| Step | Action | Output |
|---|---|---|
| **1. Define Product** | Input product name, bottle type, and brand tagline | Minimal structured specification |
| **2. Layout Planning** | Auto-generate JSON spatial bounding boxes for 5 scenes | Centered typography coordinates |
| **3. 8-Step DMD DiT** | Fast 8-step flow-matching inference (5s on 4070S) | 5 commercial 4K campaign ads |

---

## 💻 Quickstart (Windows PowerShell)

```powershell
git clone https://github.com/RayCodes/moss-image-ads.git; cd moss-image-ads; python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt; python main.py
```

---

## 📥 Model Weights Download

Download the official OpenMOSS-Image-SFT weights and the 8-step DMD LoRA using Hugging Face CLI:

```bash
# 1. Download official MOSS-Image-SFT model weights into models/
hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir models/MOSS-Image-SFT

# 2. Download 8-step DMD acceleration LoRA (Lodestones)
hf download lodestones/MOSS-Image-SFT-8-steps-dmd --local-dir models/MOSS-Image-SFT-8-steps-dmd
```

---

## 🛠️ Tech Stack & Requirements

* **Python 3.10+**
* **Diffusers >= 0.38**, **Transformers >= 5.3**, **Accelerate >= 1.10**
* **PyTorch 2.5+ with CUDA**
* **Hardware:** 8GB-16GB VRAM (or low-VRAM CPU offload / ComfyUI GGUF)

---

## 🚀 How to Run

### 1. Direct Python Execution
```bash
python main.py
```
This ingests your product specifications and compiles all 5 ad environments into `outputs/campaign_manifest.json` and outputs image files into `outputs/`.

### 2. Standalone Diffusers Pipeline with 8-Step DMD LoRA
```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "models/MOSS-Image-SFT", # or "OpenMOSS-Team/MOSS-Image-SFT"
    custom_pipeline="pipeline_moss_image",
    trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
)
pipe.enable_model_cpu_offload()

# Load 8-step DMD LoRA for 6x acceleration (guidance_scale=1.0)
pipe.load_lora_weights("models/MOSS-Image-SFT-8-steps-dmd", weight_name="lora.safetensors")

image = pipe(
    "Commercial luxury cosmetic shot of AURA GLOW serum on travertine stone",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=1.0,
).images[0]
image.save("outputs/ad_variant_1_studio.jpg")
```

---

## 🎯 5 Real-World Use Cases

1. **Brand Ad Campaign Matrix**: Generate 5 diverse photo environments (studio, macro splash, sunset vanity, obsidian, beach) for one product in minutes.
2. **E-Commerce Hero Banners**: Create high-conversion Shopify and Amazon carousel images with crisp typography.
3. **Packaging Typography Verification**: Pre-visualize printed labels and brand slogans before ordering physical bottles.
4. **Rapid A/B Ad Variant Testing**: Export distinct aesthetic angles for TikTok and Meta ad spend optimization.
5. **Hyper-Personalized Seasonal Content**: Seamlessly switch background lighting from dark obsidian studio to warm beach dunes.

---

## 🔮 5 Future Planned Features

1. **Native ComfyUI Custom Node Integration**: Direct node for 8-step DMD with automated batch queues.
2. **Interactive Web UI**: Gradio / Streamlit interface for live drag-and-drop bounding-box adjustments.
3. **LoRA Fine-Tuning Script**: One-click scripts to train custom bottle shapes on 8GB VRAM.
4. **Automated Video Zoom Motion**: Export high-res latents into animated MP4 social reels.
5. **Multi-Brand Workspace Manager**: Switch between enterprise brand guidelines in JSON configs.

---

## 🏷️ Keywords & Tags

`moss image` `comfyui moss image` `openmoss` `8-step dmd` `dit diffusion transformer` `local ai product ads` `text to image` `ai advertising studio` `flux vae` `qwen text encoder` `bbox spatial layout` `rtx 4070 local ai`
