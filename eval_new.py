"""
Inference / evaluation script for models trained with train_new.py.

Takes a trained checkpoint and runs super-resolution on a set of track folders,
computing PSNR and SSIM against the HR ground truth.

Usage
-----
# Evaluate on test tracks
python eval_new.py \
    --checkpoint experiments/train_new/checkpoint/best_gen.pth \
    --dataroot dataset/test-public \
    --output_dir experiments/train_new/eval_results

# Evaluate on a specific subset
python eval_new.py \
    --checkpoint experiments/train_new/checkpoint/best_gen.pth \
    --dataroot dataset/train/Scenario-A/Brazilian \
    --output_dir experiments/train_new/eval_results \
    --max_samples 50

# Inference only (no GT / no metrics — just produce SR images)
python eval_new.py \
    --checkpoint experiments/train_new/checkpoint/best_gen.pth \
    --dataroot dataset/test-public \
    --output_dir sr_outputs \
    --no_metrics
"""

import argparse
import os
import sys
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.metrics as Metrics
from model.LPDiff_modules.diffusion import GaussianDiffusion
from model.LPDiff_modules.unet import UNet
from model.networks import init_weights


def load_model(args, device):
    """Build model and load checkpoint weights."""
    unet = UNet(
        in_channel=args.unet_in_ch,
        out_channel=args.unet_out_ch,
        norm_groups=32,
        inner_channel=args.inner_channel,
        channel_mults=args.channel_mults,
        attn_res=args.attn_res,
        res_blocks=args.res_blocks,
        dropout=0.0,  # no dropout at inference
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

    # Load weights
    state = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(state, strict=False)
    net = net.to(device)
    net.eval()

    # Set noise schedule for inference
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


def load_image(path, img_height, img_width, device):
    """Load a single image, resize, normalise to [-1,1], return tensor [1,3,H,W]."""
    resize = transforms.Resize((img_height, img_width),
                                interpolation=transforms.InterpolationMode.BILINEAR)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    img = Image.open(path).convert("RGB")
    img = resize(img)
    tensor = normalize(to_tensor(img))
    return tensor.unsqueeze(0).to(device)


def discover_tracks(roots):
    """Find all track directories containing lr-*.png files."""
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
                    lr_files = sorted(glob(os.path.join(d, "lr-*.png")) + glob(os.path.join(d, "lr-*.jpg")))
                    hr_files = sorted(glob(os.path.join(d, "hr-*.png")) + glob(os.path.join(d, "hr-*.jpg")))
                    if lr_files:
                        root_tracks.append({"name": name, "dir": d,
                                            "lr": lr_files, "hr": hr_files})
        
        # If flat scan failed, try recursive walk (fallback)
        if not root_tracks:
            for dirpath, dirnames, filenames in os.walk(root):
                lr_files = sorted([os.path.join(dirpath, f) for f in filenames if f.startswith("lr-") and (f.endswith(".png") or f.endswith(".jpg"))])
                hr_files = sorted([os.path.join(dirpath, f) for f in filenames if f.startswith("hr-") and (f.endswith(".png") or f.endswith(".jpg"))])
                if lr_files:
                    name = os.path.basename(dirpath)
                    root_tracks.append({"name": name, "dir": dirpath, 
                                        "lr": lr_files, "hr": hr_files})

        count = len(root_tracks)
        if count == 0:
            print(f"  [Warn] No valid tracks found in {root}")
        else:
            print(f"  {root}: Found {count} tracks")
            tracks.extend(root_tracks)
            
    return tracks


def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    net = load_model(args, device)
    tracks = discover_tracks(args.dataroot)

    if args.max_samples and args.max_samples < len(tracks):
        tracks = tracks[:args.max_samples]

    print(f"Evaluating {len(tracks)} tracks...")

    all_psnr = []
    all_ssim = []

    for track in tqdm(tracks, desc="Inference"):
        lr_files = track["lr"]
        hr_files = track["hr"]

        # Pick 3 LR images (first 3, or repeat if fewer)
        if len(lr_files) >= 3:
            chosen_lr = lr_files[:3]
        else:
            chosen_lr = (lr_files * 3)[:3]

        lr1 = load_image(chosen_lr[0], args.img_height, args.img_width, device)
        lr2 = load_image(chosen_lr[1], args.img_height, args.img_width, device)
        lr3 = load_image(chosen_lr[2], args.img_height, args.img_width, device)

        # Run MTA fusion + diffusion reverse process
        with torch.no_grad():
            condition = net.MTA(lr1, lr2, lr3)
            sr = net.super_resolution(condition, continous=False)

        sr_img = Metrics.tensor2img(sr.cpu())

        # Save SR output
        out_path = os.path.join(args.output_dir, f"{track['name']}_sr.png")
        Metrics.save_img(sr_img, out_path)

        # Save input LR for reference
        lr_ref = Metrics.tensor2img(lr1.cpu())
        Metrics.save_img(lr_ref,
                         os.path.join(args.output_dir, f"{track['name']}_lr.png"))

        # Compute metrics against HR if available and not disabled
        if hr_files and not args.no_metrics:
            # Use first HR as ground truth
            hr_tensor = load_image(hr_files[0], args.img_height, args.img_width, device)
            hr_img = Metrics.tensor2img(hr_tensor.cpu())

            psnr = Metrics.calculate_psnr(sr_img, hr_img)
            ssim = Metrics.calculate_ssim(sr_img, hr_img)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            # Save HR for comparison
            Metrics.save_img(hr_img,
                             os.path.join(args.output_dir, f"{track['name']}_hr.png"))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Total tracks processed: {len(tracks)}")

    if all_psnr:
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        print(f"Average PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")

        # Save metrics to file
        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            f.write(f"Tracks evaluated: {len(all_psnr)}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"\nPer-track PSNR: {all_psnr}\n")
            f.write(f"Per-track SSIM: {all_ssim}\n")
    else:
        print("No metrics computed (no HR ground truth or --no_metrics used)")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate / infer with a trained LP-Diff model")

    # ── required ─────────────────────────────────────────────────────────
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best_gen.pth or similar checkpoint")
    p.add_argument("--dataroot", nargs="+", required=True,
                   help="Root dir(s) with track_XXXXX folders")
    p.add_argument("--output_dir", type=str, default="eval_results")

    # ── image size (must match training) ─────────────────────────────────
    p.add_argument("--img_height", type=int, default=32)
    p.add_argument("--img_width", type=int, default=64)

    # ── eval options ─────────────────────────────────────────────────────
    p.add_argument("--no_metrics", action="store_true",
                   help="Skip PSNR/SSIM computation (use when no GT available)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of tracks to evaluate")

    # ── diffusion schedule (must match training) ─────────────────────────
    p.add_argument("--schedule", type=str, default="linear")
    p.add_argument("--n_timestep", type=int, default=1000)
    p.add_argument("--linear_start", type=float, default=1e-6)
    p.add_argument("--linear_end", type=float, default=1e-2)

    # ── UNet arch (must match training) ──────────────────────────────────
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
    run_eval(args)
