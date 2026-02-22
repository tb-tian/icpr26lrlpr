"""
Training script for LP-Diff on the new dataset structure.

New dataset layout (e.g. dataset/train/Scenario-A/Brazilian/track_XXXXX/):
  - lr-001.png … lr-005.png   (low-resolution frames)
  - hr-001.png … hr-005.png   (high-resolution ground truths)
  - annotations.json           (plate text, corners, etc.)

The diffusion model takes 3 LR images → MTA fusion → condition for the
denoising U-Net, which learns to predict the residual between HR and the
fused condition.  We pick an anchor frame so the HR and LR are from the
same timestamp, plus 2 neighbouring LR frames for temporal fusion.

Usage examples
--------------
# Train on one subset (Brazilian plates only)
python train_new.py \
    --dataroot dataset/train/Scenario-A/Brazilian \
    --val_split 0.1 --epochs 300 --batch_size 8

# Train on everything under Scenario-A
python train_new.py \
    --dataroot dataset/train/Scenario-A/Brazilian \
               dataset/train/Scenario-A/Mercosur \
    --val_split 0.1

# Resume from a checkpoint
python train_new.py \
    --dataroot dataset/train/Scenario-A/Brazilian \
    --resume_gen  checkpoint/best_gen.pth \
    --resume_opt  checkpoint/best_opt.pth
"""

import argparse
import logging
import os
import random
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# ─── make sure repo modules are importable ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.metrics as Metrics
from model.LPDiff_modules import diffusion as diff_module, unet as unet_module
from model.LPDiff_modules.diffusion import GaussianDiffusion
from model.LPDiff_modules.unet import UNet
from model.networks import init_weights

# ─── logging helpers ────────────────────────────────────────────────────────
def get_logger(log_dir: str, name: str = "train"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════
class NewLPDataset(Dataset):
    """
    Dataset for the new track-based layout.

    Each track folder contains lr-{NNN}.png and hr-{NNN}.png files.
    For every sample we:
      1. Pick the track at the given index.
      2. Pick a random anchor frame — the HR comes from this frame.
      3. Choose 3 LR frames (anchor + 2 neighbours) for MTA fusion.
      4. Resize all images to a common (H, W) and normalise to [-1, 1].
    """

    def __init__(self, roots: list[str], img_height: int, img_width: int,
                 augment: bool = True):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment

        # Discover track folders across all roots
        self.tracks: list[dict] = []
        for root in roots:
            for track_name in sorted(os.listdir(root)):
                track_dir = os.path.join(root, track_name)
                if not os.path.isdir(track_dir):
                    continue
                lr_files = sorted(glob(os.path.join(track_dir, "lr-*.png")))
                hr_files = sorted(glob(os.path.join(track_dir, "hr-*.png")))
                if len(lr_files) == 0 or len(hr_files) == 0:
                    continue
                self.tracks.append({"lr": lr_files, "hr": hr_files})

        # Shared transforms
        self.resize = transforms.Resize((img_height, img_width),
                                        interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.ToTensor()  # [0, 1]
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])  # → [-1, 1]

    # ── helpers ──────────────────────────────────────────────────────────
    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = self.resize(img)
        tensor = self.to_tensor(img)       # [0, 1]
        tensor = self.normalize(tensor)     # [-1, 1]
        return tensor

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        lr_files = track["lr"]
        hr_files = track["hr"]
        n_frames = min(len(lr_files), len(hr_files))

        # Pick a random anchor frame index — the HR will come from this frame
        anchor = random.randint(0, n_frames - 1)
        chosen_hr = hr_files[anchor]

        # Pick 3 LR frames: always include the anchor, plus 2 others
        lr_indices = list(range(len(lr_files)))
        if len(lr_indices) >= 3:
            others = [i for i in lr_indices if i != anchor]
            chosen_idx = [anchor] + random.sample(others, min(2, len(others)))
            # pad if fewer than 2 others
            while len(chosen_idx) < 3:
                chosen_idx.append(anchor)
        else:
            chosen_idx = (lr_indices * 3)[:3]
        random.shuffle(chosen_idx)  # shuffle so anchor isn't always first

        lr1 = self._load(lr_files[chosen_idx[0]])
        lr2 = self._load(lr_files[chosen_idx[1]])
        lr3 = self._load(lr_files[chosen_idx[2]])
        hr  = self._load(chosen_hr)

        # NOTE: No horizontal flip — flipping destroys license plate text
        # Augmentation: small brightness/contrast jitter on LR only
        if self.augment and random.random() > 0.5:
            jitter = random.uniform(-0.05, 0.05)
            lr1 = (lr1 + jitter).clamp(-1, 1)
            lr2 = (lr2 + jitter).clamp(-1, 1)
            lr3 = (lr3 + jitter).clamp(-1, 1)

        return {"LR1": lr1, "LR2": lr2, "LR3": lr3, "HR": hr,
                "path": chosen_hr}


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════
def build_model(args, device):
    """Instantiate the UNet + GaussianDiffusion and optionally load weights."""

    unet = UNet(
        in_channel=args.unet_in_ch,
        out_channel=args.unet_out_ch,
        norm_groups=32,
        inner_channel=args.inner_channel,
        channel_mults=args.channel_mults,
        attn_res=args.attn_res,
        res_blocks=args.res_blocks,
        dropout=args.dropout,
        image_size=args.image_size,
    )

    net = GaussianDiffusion(
        denoise_fn=unet,
        image_size=args.image_size,
        channels=3,
        loss_type="l1",
        conditional=True,
        schedule_opt={
            "schedule": args.schedule,
            "n_timestep": args.n_timestep,
            "linear_start": args.linear_start,
            "linear_end": args.linear_end,
        },
    )

    init_weights(net, init_type="orthogonal")

    # Load pretrained weights if provided
    if args.resume_gen and os.path.isfile(args.resume_gen):
        state = torch.load(args.resume_gen, map_location="cpu")
        net.load_state_dict(state, strict=False)
        print(f"[✓] Loaded generator weights from {args.resume_gen}")

    net = net.to(device)
    return net


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def train(args):
    # ── device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── output dirs ──────────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    logger = get_logger(args.log_dir)

    # ── dataset & loaders ────────────────────────────────────────────────
    full_dataset = NewLPDataset(
        roots=args.dataroot,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=True,
    )
    total = len(full_dataset)
    val_size = max(1, int(total * args.val_split))
    train_size = total - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Disable augmentation for validation subset (wrap approach)
    val_set.dataset_augment_backup = full_dataset.augment  # keep reference
    # (augment flag is shared; we toggle it in the val loop)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(f"Dataset: {total} tracks  (train={train_size}, val={val_size})")
    logger.info(f"Image size: {args.img_height}x{args.img_width}")

    # ── model ────────────────────────────────────────────────────────────
    net = build_model(args, device)
    net.set_loss(device)
    net.set_new_noise_schedule(
        {
            "schedule": args.schedule,
            "n_timestep": args.n_timestep,
            "linear_start": args.linear_start,
            "linear_end": args.linear_end,
        },
        device,
    )
    logger.info("Model initialised.")

    # ── optimiser ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0

    if args.resume_opt and os.path.isfile(args.resume_opt):
        opt_state = torch.load(args.resume_opt, map_location="cpu")
        optimizer.load_state_dict(opt_state["optimizer"])
        start_epoch = opt_state.get("epoch", 0)
        global_step = opt_state.get("iter", 0)
        logger.info(f"Resumed optimiser from epoch {start_epoch}, step {global_step}")

    # ── scheduler (cosine annealing) ─────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for _ in range(start_epoch):
        scheduler.step()  # fast-forward if resuming

    # ── tracking ─────────────────────────────────────────────────────────
    best_val_psnr = -float("inf")
    best_val_loss = float("inf")

    # ════════════════════════════  EPOCH LOOP  ════════════════════════════
    for epoch in range(start_epoch, args.epochs):
        net.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Move data to device
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            optimizer.zero_grad()
            loss = net(data)                    # GaussianDiffusion.forward → p_losses
            b, c, h, w = data["HR"].shape
            loss = loss.sum() / (b * c * h * w)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # ── periodic console log ─────────────────────────────────────
            if global_step % args.print_freq == 0:
                logger.info(
                    f"<epoch:{epoch:4d}  step:{global_step:8d}>  "
                    f"l_pix: {loss.item():.4e}  lr: {scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            f"Epoch {epoch:4d} done  |  avg_train_loss: {avg_train_loss:.4e}  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()

        # ── validation ───────────────────────────────────────────────────
        if (epoch + 1) % args.val_freq == 0:
            net.eval()
            full_dataset.augment = False          # disable augment for val

            val_psnr_sum = 0.0
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for vbatch in val_loader:
                    vdata = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in vbatch.items()}
                    # Fuse LR images with MTA, then run reverse diffusion
                    condition = net.MTA(vdata["LR1"], vdata["LR2"], vdata["LR3"])
                    sr = net.super_resolution(condition, continous=False)

                    mse = nn.MSELoss()(sr, vdata["HR"])
                    val_loss_sum += mse.item()

                    sr_img = Metrics.tensor2img(sr.cpu())
                    hr_img = Metrics.tensor2img(vdata["HR"].cpu())
                    psnr = Metrics.calculate_psnr(sr_img, hr_img)
                    val_psnr_sum += psnr
                    val_count += 1

                    # Save first sample for visual check
                    if val_count == 1:
                        result_path = os.path.join(args.result_dir, f"epoch_{epoch+1}")
                        os.makedirs(result_path, exist_ok=True)
                        Metrics.save_img(sr_img,
                                         os.path.join(result_path, "sr.png"))
                        Metrics.save_img(hr_img,
                                         os.path.join(result_path, "hr.png"))
                        lr1_img = Metrics.tensor2img(vdata["LR1"].cpu())
                        Metrics.save_img(lr1_img,
                                         os.path.join(result_path, "lr1.png"))

            full_dataset.augment = True           # re-enable augment

            avg_val_loss = val_loss_sum / max(val_count, 1)
            avg_val_psnr = val_psnr_sum / max(val_count, 1)

            logger.info(
                f"  ── VAL  epoch {epoch+1:4d} ──  "
                f"PSNR: {avg_val_psnr:.4f}  loss: {avg_val_loss:.4e}"
            )

            # ── save best model ──────────────────────────────────────────
            improved = False
            if avg_val_psnr > best_val_psnr:
                best_val_psnr = avg_val_psnr
                improved = True
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                improved = True
            if improved:
                _save_checkpoint(net, optimizer, epoch + 1, global_step,
                                 args.checkpoint_dir, tag="best")
                logger.info(
                    f"  ★ New best  PSNR={best_val_psnr:.4f}  "
                    f"loss={best_val_loss:.4e}  (saved)"
                )

            net.train()

        # ── periodic checkpoint ──────────────────────────────────────────
        if (epoch + 1) % args.save_freq == 0:
            _save_checkpoint(net, optimizer, epoch + 1, global_step,
                             args.checkpoint_dir, tag=f"epoch{epoch+1}")
            logger.info(f"  Checkpoint saved (epoch {epoch+1})")

    # ── final save ───────────────────────────────────────────────────────
    _save_checkpoint(net, optimizer, args.epochs, global_step,
                     args.checkpoint_dir, tag="final")
    logger.info("Training complete.")


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _save_checkpoint(net, optimizer, epoch, step, ckpt_dir, tag="latest"):
    net_state = net.state_dict()
    # Move everything to CPU for portability
    for k in net_state:
        net_state[k] = net_state[k].cpu()
    gen_path = os.path.join(ckpt_dir, f"{tag}_gen.pth")
    opt_path = os.path.join(ckpt_dir, f"{tag}_opt.pth")
    torch.save(net_state, gen_path)
    torch.save({"epoch": epoch, "iter": step,
                "optimizer": optimizer.state_dict()}, opt_path)


# ═══════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Train LP-Diff on the new track-based dataset")

    # ── data ─────────────────────────────────────────────────────────────
    p.add_argument("--dataroot", nargs="+", required=True,
                   help="One or more root directories containing track_XXXXX folders")
    p.add_argument("--img_height", type=int, default=32,
                   help="Resize height for all images (default: 32)")
    p.add_argument("--img_width", type=int, default=64,
                   help="Resize width for all images  (default: 64)")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="Fraction of tracks used for validation (default: 0.1)")

    # ── training ─────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Initial learning rate (default: 1e-4)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--print_freq", type=int, default=50,
                   help="Log every N global steps")
    p.add_argument("--val_freq", type=int, default=5,
                   help="Validate every N epochs")
    p.add_argument("--save_freq", type=int, default=20,
                   help="Save checkpoint every N epochs")

    # ── diffusion schedule ───────────────────────────────────────────────
    p.add_argument("--schedule", type=str, default="linear",
                   choices=["linear", "cosine", "quad"])
    p.add_argument("--n_timestep", type=int, default=1000)
    p.add_argument("--linear_start", type=float, default=1e-6)
    p.add_argument("--linear_end", type=float, default=1e-2)

    # ── UNet architecture ────────────────────────────────────────────────
    p.add_argument("--unet_in_ch", type=int, default=6,
                   help="UNet input channels (condition + noisy = 3+3)")
    p.add_argument("--unet_out_ch", type=int, default=3)
    p.add_argument("--inner_channel", type=int, default=64)
    p.add_argument("--channel_mults", nargs="+", type=int,
                   default=[1, 2, 4, 8])
    p.add_argument("--attn_res", nargs="+", type=int, default=[16])
    p.add_argument("--res_blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--image_size", type=int, default=32,
                   help="Internal image_size reference for UNet attention layers (should match img_height)")

    # ── checkpoints / resume ─────────────────────────────────────────────
    p.add_argument("--resume_gen", type=str, default=None,
                   help="Path to generator .pth to resume from")
    p.add_argument("--resume_opt", type=str, default=None,
                   help="Path to optimiser .pth to resume from")

    # ── output paths ─────────────────────────────────────────────────────
    p.add_argument("--log_dir", type=str, default="experiments/train_new/logs")
    p.add_argument("--checkpoint_dir", type=str,
                   default="experiments/train_new/checkpoint")
    p.add_argument("--result_dir", type=str,
                   default="experiments/train_new/results")

    # ── GPU ──────────────────────────────────────────────────────────────
    p.add_argument("--gpu", type=str, default="0",
                   help="Comma-separated GPU ids (default: '0')")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(args)
