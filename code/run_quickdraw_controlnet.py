import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


# -----------------------------
# Config
# -----------------------------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-scribble"  # scribble ControlNet
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# -----------------------------
# QuickDraw loader
# -----------------------------
def load_quickdraw_bitmap_npy(npy_path: str, idx: int = 0) -> np.ndarray:
    """
    Loads a QuickDraw numpy_bitmap .npy and returns one sample as 28x28 uint8.
    The .npy contains N x 784 flattened grayscale bitmaps.
    """
    arr = np.load(npy_path)               # shape: (N, 784)
    if arr.ndim != 2 or arr.shape[1] != 784:
        raise ValueError(f"Unexpected shape {arr.shape}. Expected (N, 784).")
    if idx < 0 or idx >= arr.shape[0]:
        raise IndexError(f"idx {idx} out of range. N={arr.shape[0]}")
    img28 = arr[idx].reshape(28, 28).astype(np.uint8)
    return img28


def quickdraw_to_scribble_image(img28: np.ndarray, size: int = 512) -> Image.Image:
    """
    Converts 28x28 grayscale bitmap to a ControlNet scribble conditioning image (RGB).
    - Upscales to (size, size)
    - Ensures black lines on white background
    - Optionally thickens lines a bit for better conditioning
    """
    # QuickDraw bitmap often has white strokes on black background depending on source.
    # We normalize to black strokes on white background.
    # Heuristic: if mean is low, likely black background -> invert.
    if img28.mean() < 127:
        img28 = 255 - img28

    # Upscale (nearest keeps the doodle look)
    up = cv2.resize(img28, (size, size), interpolation=cv2.INTER_NEAREST)

    # Binarize to make it more "scribble-like"
    _, bw = cv2.threshold(up, 200, 255, cv2.THRESH_BINARY)

    # Make strokes thicker (optional but helps)
    kernel = np.ones((3, 3), np.uint8)
    bw_thick = cv2.erode(bw, kernel, iterations=1)

    # Convert to RGB PIL
    rgb = np.stack([bw_thick, bw_thick, bw_thick], axis=-1)
    return Image.fromarray(rgb)


# -----------------------------
# Pipeline
# -----------------------------
def build_pipe():
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,   # demo 목적이면 꺼도 됨 (환경 정책은 본인 환경에 맞게)
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)

    # Optional speed/memory tweaks
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def make_grid(images, cols: int = 3) -> Image.Image:
    """
    Simple grid maker for PIL images.
    """
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(im, (c * w, r * h))
    return grid


# -----------------------------
# Main experiment: CFG sweep
# -----------------------------
def run_cfg_sweep(
    npy_path: str,
    sample_idx: int,
    prompt: str,
    negative_prompt: str = "low quality, blurry, deformed",
    cfg_list=(3.0, 7.0, 12.0),
    steps: int = 30,
    seed: int = 42,
):
    pipe = build_pipe()

    img28 = load_quickdraw_bitmap_npy(npy_path, sample_idx)
    control_img = quickdraw_to_scribble_image(img28, size=512)

    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    results = []
    for cfg in cfg_list:
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_img,                # ControlNet conditioning image
            guidance_scale=float(cfg),        # CFG scale
            num_inference_steps=int(steps),
            generator=gen,
        )
        results.append(out.images[0])

    # Save conditioning + grid
    control_img.save(OUT_DIR / f"control_idx{sample_idx}.png")
    grid = make_grid(results, cols=len(results))
    grid.save(OUT_DIR / f"cfg_sweep_idx{sample_idx}.png")

    print(f"[OK] saved: {OUT_DIR / f'cfg_sweep_idx{sample_idx}.png'}")


if __name__ == "__main__":
    # ---- Edit these ----
    NPY_PATH = "house.npy"  # 예: numpy_bitmap/house.npy
    SAMPLE_IDX = 0

    # prompt는 "의사소통 장면"으로 가게끔 구체화하는 게 포인트
    PROMPT = "a small cozy house in a park, clear composition, simple illustration, friendly, high detail"
    NEG = "text, watermark, blurry, low quality, distorted, extra limbs"

    run_cfg_sweep(
        npy_path=NPY_PATH,
        sample_idx=SAMPLE_IDX,
        prompt=PROMPT,
        negative_prompt=NEG,
        cfg_list=(3.0, 7.0, 12.0),
        steps=30,
        seed=123,
    )
