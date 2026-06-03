"""
3.3  Image Generation Agent  
========================================
Task : Generate a new image based on the refined prompt from the
       Prompt Engineering Agent.

Requirements:
  - Use a text-to-image model (Stable Diffusion 1.5 via diffusers).
  - Generate at least one output image.
  - Support one simple generation mode: stylization, variation, or enhancement.
  - Record basic configuration: prompt, model, inference steps, seed.
  - Auto-detects CUDA GPU; falls back to CPU gracefully.
"""

import time
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from PIL import Image


# Lazy import to avoid slow load at import time
_pipeline_cache: dict = {}


class ImageGenerationAgent:
    """
    Image Generation Agent — wraps Stable Diffusion 1.5 (via diffusers)
    to produce images from text prompts.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier. Defaults to SD 1.5.
    output_dir : str | Path
        Directory where generated images are saved.
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    num_inference_steps : int
        Diffusion steps (20 is fast; 50 is higher quality).
    guidance_scale : float
        Classifier-free guidance scale (7.5 is standard).
    """

    DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        output_dir: Union[str, Path] = "outputs",
        device: Union[str, None] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Device selection
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[GenerationAgent] Device: {self.device.upper()}")
        if self.device == "cpu":
            print(
                "  ⚠  No CUDA GPU detected — generation will run on CPU.\n"
                "     Expect ~1–3 minutes per image with 25 steps."
            )

        self._pipe = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        refined_prompt: str,
        mode: str = "stylization",
        negative_prompt: str = (
            "blurry, low quality, watermark, text, ugly, disfigured, "
            "bad anatomy, extra limbs"
        ),
        seed: Union[int, None] = None,
        num_images: int = 1,
    ) -> dict:
        """
        Generate image(s) from *refined_prompt*.

        Parameters
        ----------
        refined_prompt : str
            The enriched prompt from the Prompt Engineering Agent.
        mode : str
            Generation mode label (``stylization`` / ``variation`` /
            ``enhancement``).  Recorded in metadata only.
        negative_prompt : str
            What to avoid in the generated image.
        seed : int | None
            Random seed for reproducibility.  Random if *None*.
        num_images : int
            Number of images to generate (default 1).

        Returns
        -------
        dict with keys:
            output_paths (list), metadata, generation_time_s, mode
        """
        self._load_pipeline()

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print(
            f"[GenerationAgent] Generating {num_images} image(s) "
            f"[mode={mode}, steps={self.num_inference_steps}, seed={seed}] …"
        )
        t0 = time.time()

        # Run diffusion
        result = self._pipe(
            prompt=refined_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )
        elapsed = round(time.time() - t0, 1)

        # Save images
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_paths = []
        for idx, img in enumerate(result.images):
            fname = self.output_dir / f"gen_{mode}_{ts}_{idx}.png"
            img.save(fname)
            output_paths.append(str(fname))
            print(f"  Saved: {fname}")

        metadata = {
            "prompt": refined_prompt,
            "negative_prompt": negative_prompt,
            "model_id": self.model_id,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": seed,
            "device": self.device,
            "mode": mode,
            "num_images": num_images,
            "generation_time_s": elapsed,
        }

        # Save metadata JSON alongside images
        meta_path = self.output_dir / f"gen_{mode}_{ts}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[GenerationAgent] Done in {elapsed}s. Metadata → {meta_path}")

        return {
            "output_paths": output_paths,
            "metadata": metadata,
            "generation_time_s": elapsed,
            "mode": mode,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load (or retrieve cached) Stable Diffusion pipeline."""
        global _pipeline_cache
        cache_key = (self.model_id, self.device)

        if cache_key in _pipeline_cache:
            self._pipe = _pipeline_cache[cache_key]
            return

        from diffusers import StableDiffusionPipeline

        print(f"[GenerationAgent] Loading model '{self.model_id}' …")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,          # disabled for research use
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.device)

        if self.device == "cuda":
            pipe.enable_attention_slicing()  # save VRAM

        _pipeline_cache[cache_key] = pipe
        self._pipe = pipe
        print("[GenerationAgent] Model loaded and ready.")
