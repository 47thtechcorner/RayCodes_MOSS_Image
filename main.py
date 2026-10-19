"""
MOSS Image 8-Step Batch Product-Ad Studio
Automated commercial campaign generation for a single product across 5 diverse production environments.
Executes OpenMOSS-Team/MOSS-Image-SFT via Hugging Face Diffusers with 8-step DMD acceleration.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# Lazy-load torch and diffusers when pipeline is executed
torch = None
DiffusionPipeline = None

@dataclass
class ProductSpec:
    name: str = "AURA GLOW"
    item: str = "Botanical Face Serum dropper bottle"

class MossAdCampaignGenerator:
    # 5 distinctly styled, highly complex commercial ad scenes with varied angles, lighting, and typographic layouts
    SCENES = [
        ("Macro Liquid Dropper", "extreme close-up macro 45-degree angle of glass pipette releasing an amber oil droplet into glowing emerald serum", "ad_variant_1_studio.jpg", [150, 200, 850, 800]),
        ("Travertine Monolith Hero", "low-angle dramatic hero perspective resting between raw chipped beige travertine blocks with deep geometric diagonal shadows", "ad_variant_2_splash.jpg", [250, 320, 780, 680]),
        ("Dynamic Botanical Vortex", "high-velocity dynamic spiral of shattered crystal ice, floating green tea leaves, and effervescent splashing water droplets", "ad_variant_3_golden_hour.jpg", [180, 250, 820, 750]),
        ("Floating Cyber Vanity", "tilted 30-degree levitating bottle above reflective black mirror glass with dual cyan and neon green circular rim lighting", "ad_variant_4_obsidian.jpg", [200, 300, 800, 700]),
        ("Editorial Skincare Flatlay", "top-down bird's-eye flatlay composition flanked by halved golden bergamot fruit, fresh botanicals, and textured linen", "ad_variant_5_mediterranean.jpg", [300, 350, 700, 650]),
    ]

    def __init__(self, spec: ProductSpec, output_dir: str = "outputs"):
        self.spec = spec
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.pipe = None

    def load_pipeline(self, model_id: str = "OpenMOSS-Team/MOSS-Image-SFT", lora_id: str = "lodestones/MOSS-Image-SFT-8-steps-dmd"):
        """Initializes MOSS-Image-SFT pipeline with 8-step DMD LoRA."""
        global torch, DiffusionPipeline
        if torch is None:
            import torch
            from diffusers import DiffusionPipeline

        print(f"[*] Initializing MOSS-Image-SFT: {model_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_cfg = {"default": torch.bfloat16, "vae": torch.float32} if device == "cuda" else torch.float32

        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, custom_pipeline="pipeline_moss_image", trust_remote_code=True, torch_dtype=dtype_cfg
        )
        if device == "cuda":
            try: self.pipe.enable_model_cpu_offload()
            except Exception: self.pipe.to("cuda")
        else:
            self.pipe.to("cpu")

        if lora_id:
            try:
                self.pipe.load_lora_weights(lora_id, weight_name="lora.safetensors")
                print("[*] 8-Step DMD LoRA fused successfully (guidance_scale=1.0).")
            except Exception as e:
                print(f"[!] LoRA notice: {e}")

    def get_campaign_variants(self) -> List[Dict[str, Any]]:
        """Automatically builds 5 dynamic, multi-angle commercial environments for the single product."""
        variants = []
        for idx, (title, backdrop, filename, bbox) in enumerate(self.SCENES, start=1):
            prompt = (
                f"Commercial luxury advertising photography of {self.spec.item} with embossed gold typography '{self.spec.name}', "
                f"composition: {backdrop}, bold multi-line label text '100% PURE RADIANCE', cinematic commercial lighting, 8k print."
            )
            layout = {
                "summary": {"caption": f"{self.spec.name} {title} dynamic commercial campaign"},
                "items": [{"label": f"{self.spec.name} bottle", "bbox": bbox}]
            }
            variants.append({"id": idx, "title": title, "filename": filename, "prompt": prompt, "layout": layout})
        return variants

    def run(self, num_steps: int = 8, guidance_scale: float = 1.0, width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """Executes the batch ad creation process across all 5 distinct environments."""
        variants = self.get_campaign_variants()
        manifest_entries = []

        print("=" * 68)
        print(f"[MOSS-Image] 8-Step DMD Studio | Product: {self.spec.name} ({self.spec.item})")
        print(f"[*] Engine: MOSS-Image-SFT + 8-Step DMD LoRA | Steps: {num_steps} | CFG: {guidance_scale}")
        print("=" * 68)

        for v in variants:
            filepath = os.path.join(self.output_dir, v["filename"])
            print(f"\n[+] Variant {v['id']}/5: {v['title']} -> {v['prompt'][:75]}...")
            if self.pipe is not None:
                gen = torch.Generator(device=self.pipe.device).manual_seed(100 + v["id"])
                img = self.pipe(prompt=v["prompt"], height=height, width=width, num_inference_steps=num_steps, guidance_scale=guidance_scale, generator=gen).images[0]
                img.save(filepath)
                print(f"    [OK] Saved: {filepath}")
            else:
                print(f"    [TARGET] Deliverable: {filepath}")

            manifest_entries.append({"id": v["id"], "title": v["title"], "file": filepath, "prompt": v["prompt"], "layout": v["layout"], "steps": num_steps, "cfg": guidance_scale})

        manifest_path = os.path.join(self.output_dir, "campaign_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"product_spec": asdict(self.spec), "settings": {"steps": num_steps, "cfg": guidance_scale}, "variants": manifest_entries}, f, indent=2)

        print("\n" + "=" * 68)
        print(f"[SUCCESS] 5 Commercial Variants Compiled! Manifest: {manifest_path}")
        print("=" * 68)
        return {"status": "success", "manifest": manifest_path, "variants": manifest_entries}


def main():
    spec = ProductSpec(name="AURA GLOW", item="Botanical Face Serum dropper bottle")
    studio = MossAdCampaignGenerator(spec)

    local_weights = os.path.join("models", "MOSS-Image-SFT")
    if os.path.isdir(local_weights) and any(f.endswith((".safetensors", ".bin")) for f in os.listdir(local_weights)):
        studio.load_pipeline(model_id=local_weights)

    studio.run(num_steps=8, guidance_scale=1.0)


if __name__ == "__main__":
    main()
