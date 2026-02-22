"""
Inference script for LP-Diff — works on tracks with ONLY LR images (no HR).

For each track folder it:
  1. Loads up to 3 LR frames (lr-001.png, lr-002.png, …)
  2. Fuses them with the MTA module
  3. Runs the reverse diffusion to produce a super-resolved image
  4. Saves the SR output to --output_dir/<track_name>/sr-001.png, …

It also generates one SR per HR-slot (matching the number of LR frames),
because downstream evaluation may expect one SR per LR frame.

Usage
-----
# Basic inference on test-public (LR-only tracks)
python infer_new.py \
    --checkpoint experiments/train_new/checkpoint/best_gen.pth \
    --dataroot dataset/test-public

# Specify output directory
python infer_new.py \
    --checkpoint experiments/train_new/checkpoint/best_gen.pth \
    --dataroot dataset/test-public \
    --output_dir my_sr_results

# Limit to first 10 tracks (for quick testing)
python infer_new.py \
    --checkpoint experiments/train_new/checkpoint/best_gen.pth \
    --dataroot dataset/test-public \
    --max_tracks 10
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.metrics as Metrics
from model.LPDiff_modules.diffusion import GaussianDiffusion
from model.LPDiff_modules.unet import UNet
from model.networks import init_weights


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════
def load_model(args, device):
    """Build the diffusion model and load checkpoint weights."""
    unet = UNet(
        in_channel=args.unet_in_ch,
        out_channel=args.unet_out_ch,
        norm_groups=32,
        inner_channel=args.inner_channel,
        channel_mults=args.channel_mults,
        attn_res=args.attn_res,
        res_blocks=args.res_blocks,
        dropout=0.0,   # no dropout at inference
        image_size=args.image_size,
    )
    net = GaussianDiffusion(
        denoise_fn=unet,
        image_size=args.image_size,
        channels=3,
        loss_type="l1",
        conditional=True,
        schedule_opt=None,
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(state, strict=False)
    print(f"[✓] Loaded checkpoint: {args.checkpoint}")

    net = net.to(device)
    net.eval()

    net.set_new_noise_schedule(
        {
            "schedule": args.schedule,
            "n_timestep": args.n_timestep,
            "linear_start": args.linear_start,
            "linear_end": args.linear_end,
        },
        device,
    )
    return net


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE I/O
# ═══════════════════════════════════════════════════════════════════════════
def make_transform(img_height, img_width):
    """Return (resize, to_tensor, normalize) transforms."""
    resize = transforms.Resize(
        (img_height, img_width),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    return resize, to_tensor, normalize


def load_image(path, resize, to_tensor, normalize, device):
    """Load → resize → normalise → [1,3,H,W] tensor on device."""
    img = Image.open(path).convert("RGB")
    img = resize(img)
    t = normalize(to_tensor(img))
    return t.unsqueeze(0).to(device)


def tensor_to_pil(tensor, target_size=None):
    """
    Convert a [-1,1] tensor [1,3,H,W] back to a PIL Image.
    Optionally resize to target_size=(W,H) to match original LR dimensions.
    """
    t = tensor.squeeze(0).cpu().clamp(-1, 1)
    t = (t + 1) / 2          # → [0, 1]
    t = (t * 255).byte()
    arr = t.permute(1, 2, 0).numpy()
    pil = Image.fromarray(arr)
    if target_size is not None:
        pil = pil.resize(target_size, Image.BILINEAR)
    return pil


# ═══════════════════════════════════════════════════════════════════════════
#  TRACK DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════
def discover_tracks(roots):
    """
    Find all track directories that contain at least one lr-*.png.
    First tries flat scan, then falls back to recursive scan.
    Returns list of dicts: {name, dir, lr_files}.
    """
    tracks = []
    print(f"[discover] Roots to scan: {roots}")

    for root in roots:
        if not os.path.exists(root):
            print(f"[Warn] Root directory {root} does not exist. Skipping.")
            continue

        root_tracks = []
        # Try flat scan first
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                d = os.path.join(root, name)
                if os.path.isdir(d):
                    lr_files = sorted(glob(os.path.join(d, "lr-*.png")))
                    if lr_files:
                        root_tracks.append({"name": name, "dir": d, "lr": lr_files})
        
        # If flat scan failed, try recursive walk (fallback)
        if not root_tracks:
            for dirpath, dirnames, filenames in os.walk(root):
                lr_files = sorted([os.path.join(dirpath, f) for f in filenames if f.startswith("lr-") and f.endswith(".png")])
                if lr_files:
                    name = os.path.basename(dirpath)
                    root_tracks.append({"name": name, "dir": dirpath, "lr": lr_files})

        count = len(root_tracks)
        if count == 0:
            print(f"  [Warn] No valid tracks found in {root}")
        else:
            print(f"  {root}: Found {count} tracks")
            tracks.extend(root_tracks)
            
    return tracks


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net = load_model(args, device)
    resize, to_tensor, normalize = make_transform(args.img_height, args.img_width)
    tracks = discover_tracks(args.dataroot)

    if args.max_tracks and args.max_tracks < len(tracks):
        tracks = tracks[: args.max_tracks]

    print(f"Processing {len(tracks)} tracks → {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    for track in tqdm(tracks, desc="Inference"):
        lr_files = track["lr"]
        n_lr = len(lr_files)
        track_out = os.path.join(args.output_dir, track["name"])
        os.makedirs(track_out, exist_ok=True)

        # For each LR frame, produce one SR image.
        # We use the target frame + its neighbours as the 3 MTA inputs.
        for i in range(n_lr):
            # Build the 3-frame window centred on frame i
            indices = _pick_3_indices(i, n_lr)

            lr1 = load_image(lr_files[indices[0]], resize, to_tensor, normalize, device)
            lr2 = load_image(lr_files[indices[1]], resize, to_tensor, normalize, device)
            lr3 = load_image(lr_files[indices[2]], resize, to_tensor, normalize, device)

            with torch.no_grad():
                condition = net.MTA(lr1, lr2, lr3)
                sr = net.super_resolution(condition, continous=False)

            # Get original LR size so we can optionally resize SR to match
            orig_lr = Image.open(lr_files[i])
            # SR output name mirrors the LR name:  lr-001.png → sr-001.png
            sr_name = os.path.basename(lr_files[i]).replace("lr-", "sr-")
            sr_pil = tensor_to_pil(sr)
            sr_pil.save(os.path.join(track_out, sr_name))

            # Also save a copy at the original LR resolution (if requested)
            if args.save_original_size:
                sr_orig = tensor_to_pil(sr, target_size=orig_lr.size)
                orig_name = sr_name.replace("sr-", "sr-orig-")
                sr_orig.save(os.path.join(track_out, orig_name))

    print(f"\nDone. SR images saved to: {args.output_dir}")
    print(f"  Total tracks:  {len(tracks)}")
    print(f"  Output format: <track_name>/sr-NNN.png")


def _pick_3_indices(centre: int, total: int) -> list[int]:
    """
    Pick 3 frame indices for MTA input, centred on `centre`.
    Always includes `centre`; tries to pick one before and one after.
    """
    if total == 1:
        return [0, 0, 0]
    if total == 2:
        other = 1 - centre
        return sorted([centre, centre, other])

    # total >= 3 — pick centre, one before, one after
    before = centre - 1 if centre > 0 else centre + 1
    after = centre + 1 if centre < total - 1 else centre - 1
    # Ensure 3 unique if possible
    indices = list({before, centre, after})
    while len(indices) < 3:
        indices.append(centre)
    return sorted(indices)


# ═══════════════════════════════════════════════════════════════════════════
#  ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Run LP-Diff inference on LR-only tracks")

    # ── required ─────────────────────────────────────────────────────────
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to *_gen.pth checkpoint")
    p.add_argument("--dataroot", nargs="+", required=True,
                   help="Root dir(s) containing track_XXXXX folders with lr-*.png")
    p.add_argument("--output_dir", type=str, default="sr_outputs",
                   help="Where to write SR images (default: sr_outputs)")

    # ── image size (must match training) ─────────────────────────────────
    p.add_argument("--img_height", type=int, default=32)
    p.add_argument("--img_width", type=int, default=64)

    # ── inference options ────────────────────────────────────────────────
    p.add_argument("--max_tracks", type=int, default=None,
                   help="Process only the first N tracks (for quick testing)")
    p.add_argument("--save_original_size", action="store_true",
                   help="Also save SR resized back to original LR dimensions")

    # ── diffusion schedule (must match training) ─────────────────────────
    p.add_argument("--schedule", type=str, default="linear")
    p.add_argument("--n_timestep", type=int, default=1000)
    p.add_argument("--linear_start", type=float, default=1e-6)
    p.add_argument("--linear_end", type=float, default=1e-2)

    # ── UNet architecture (must match training) ──────────────────────────
    p.add_argument("--unet_in_ch", type=int, default=6)
    p.add_argument("--unet_out_ch", type=int, default=3)
    p.add_argument("--inner_channel", type=int, default=64)
    p.add_argument("--channel_mults", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--attn_res", nargs="+", type=int, default=[16])
    p.add_argument("--res_blocks", type=int, default=2)
    p.add_argument("--image_size", type=int, default=32)

    # ── GPU ──────────────────────────────────────────────────────────────
    p.add_argument("--gpu", type=str, default="0")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_inference(args)
