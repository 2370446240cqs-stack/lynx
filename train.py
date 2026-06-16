# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from diffusers import FlowMatchEulerDiscreteScheduler

from modules.common.inference_utils import dtype_mapping
from modules.lite.attention_processor import register_ip_adapter_wan
from modules.lite.lynx_lite_pipeline import LynxLiteWanPipeline


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    data_dir: str = str(SCRIPT_DIR / "data" / "train")
    base_model_path: str = str(SCRIPT_DIR / "Wan2.1-T2V-14B-Diffusers")
    vggt_omega_path: str = str(SCRIPT_DIR / "vggt-omega")
    output_dir: str = str(SCRIPT_DIR / "checkpoints" / "vggt_omega_ip_adapter")
    npz_key: str = ""
    height: int = 480
    width: int = 832
    num_frames: int = 81
    max_sequence_length: int = 512
    train_batch_size: int = 1
    num_workers: int = 0
    max_train_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    checkpointing_steps: int = 1000
    logging_steps: int = 10
    seed: int = 42
    torch_dtype: str = "bf16"
    device: str = "cuda:0"
    ip_scale: float = 1.0
    ip_layers: int = 2
    init_method: str = "zero"
    resume_ip_layers: str = ""


class VideoVGGTFeatureDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        height: int,
        width: int,
        num_frames: int,
        npz_key: str = "",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.npz_key = npz_key

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Training data directory not found: {self.data_dir}")

        self.samples = []
        for video_path in sorted(self.data_dir.glob("*.mp4")):
            stem = video_path.stem
            prompt_path = self.data_dir / f"{stem}.txt"
            feature_path = self.data_dir / f"{stem}.npz"
            if not prompt_path.is_file() or not feature_path.is_file():
                raise FileNotFoundError(
                    f"Sample {stem} is incomplete. Expected {video_path.name}, {prompt_path.name}, and {feature_path.name}."
                )
            self.samples.append((video_path, prompt_path, feature_path))

        if not self.samples:
            raise ValueError(f"No .mp4 samples found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        video_path, prompt_path, feature_path = self.samples[index]
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        video = load_video_tensor(video_path, self.height, self.width, self.num_frames)
        vggt_tokens = load_npz_tensor(feature_path, self.npz_key)

        return {
            "video": video,
            "prompt": prompt,
            "vggt_tokens": vggt_tokens,
            "name": video_path.stem,
        }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a VGGT-Omega-conditioned Lynx/Wan IP-adapter.")
    parser.add_argument("--data_dir", default=str(SCRIPT_DIR / "data" / "train"))
    parser.add_argument("--base_model_path", default=str(SCRIPT_DIR / "Wan2.1-T2V-14B-Diffusers"))
    parser.add_argument("--vggt_omega_path", default=str(SCRIPT_DIR / "vggt-omega"))
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "checkpoints" / "vggt_omega_ip_adapter"))
    parser.add_argument("--npz_key", default="", help="Feature key inside each .npz. Defaults to the first array.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ip_scale", type=float, default=1.0)
    parser.add_argument(
        "--ip_layers",
        type=int,
        default=2,
        help="Register IP-adapter every N transformer blocks, matching Lynx-lite default.",
    )
    parser.add_argument("--init_method", choices=["zero", "clone"], default="zero")
    parser.add_argument("--resume_ip_layers", default="", help="Optional ip_layers.safetensors to resume from.")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    validate_config(cfg)
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device(cfg.device)
    weight_dtype = dtype_mapping[cfg.torch_dtype]

    dataset = VideoVGGTFeatureDataset(
        data_dir=cfg.data_dir,
        height=cfg.height,
        width=cfg.width,
        num_frames=cfg.num_frames,
        npz_key=cfg.npz_key,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_samples,
    )

    pipe = LynxLiteWanPipeline.from_pretrained(cfg.base_model_path, torch_dtype=weight_dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(cfg.base_model_path, subfolder="scheduler")

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.transformer.eval()

    feature_dim = infer_feature_dim(dataset)
    hidden_size = pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim
    if cfg.init_method == "clone" and feature_dim != hidden_size:
        raise ValueError(
            "--init_method clone requires VGGT feature dim to equal Wan hidden size "
            f"({hidden_size}), but got {feature_dim}. Use --init_method zero for VGGT-Omega tokens."
        )

    pipe.transformer, ip_layers = register_ip_adapter_wan(
        pipe.transformer,
        hidden_size=hidden_size,
        cross_attention_dim=feature_dim,
        dtype=weight_dtype,
        init_method=cfg.init_method,
        layers=cfg.ip_layers,
    )
    if cfg.resume_ip_layers:
        from safetensors.torch import load_file

        ip_layers.load_state_dict(load_file(cfg.resume_ip_layers, device="cpu"))

    pipe.to(device)
    ip_layers.train()

    optimizer = torch.optim.AdamW(ip_layers.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    global_step = 0
    accumulation_step = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < cfg.max_train_steps:
        for batch in dataloader:
            loss = training_step(pipe, scheduler, batch, cfg, device, weight_dtype)
            (loss / cfg.gradient_accumulation_steps).backward()
            accumulation_step += 1

            if accumulation_step % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(ip_layers.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    print(f"step={global_step} loss={loss.item():.6f}")

                if global_step % cfg.checkpointing_steps == 0:
                    save_checkpoint(ip_layers, cfg.output_dir, global_step)

                if global_step >= cfg.max_train_steps:
                    break

    save_checkpoint(ip_layers, cfg.output_dir, global_step)


def validate_config(cfg: TrainConfig) -> None:
    if cfg.num_frames < 1:
        raise ValueError("--num_frames must be positive")
    if (cfg.num_frames - 1) % 4 != 0:
        raise ValueError("--num_frames must satisfy (num_frames - 1) % 4 == 0 for Wan VAE temporal scaling")
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if cfg.train_batch_size < 1:
        raise ValueError("--train_batch_size must be positive")
    if not os.path.isdir(cfg.base_model_path):
        raise FileNotFoundError(f"Wan2.1 model directory not found: {cfg.base_model_path}")
    if not os.path.isdir(cfg.vggt_omega_path):
        print(f"[WARN] VGGT-Omega path does not exist: {cfg.vggt_omega_path}. Precomputed .npz features will still be used.")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def collate_samples(samples: list[dict]) -> dict:
    return {
        "video": torch.stack([sample["video"] for sample in samples]),
        "prompt": [sample["prompt"] for sample in samples],
        "vggt_tokens": torch.stack([sample["vggt_tokens"] for sample in samples]),
        "name": [sample["name"] for sample in samples],
    }


def load_npz_tensor(path: Path, key: str = "") -> torch.Tensor:
    with np.load(path) as data:
        if key:
            if key not in data:
                raise KeyError(f"{path} does not contain key {key!r}. Available keys: {list(data.keys())}")
            array = data[key]
        else:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"{path} does not contain any arrays")
            array = data[keys[0]]

    tensor = torch.from_numpy(np.asarray(array)).float()
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected VGGT feature tokens with shape [tokens, dim], got {tuple(tensor.shape)} from {path}")
    return tensor


def load_video_tensor(path: Path, height: int, width: int, num_frames: int) -> torch.Tensor:
    reader = imageio.get_reader(path)
    try:
        total_frames = reader.count_frames()
    except Exception:
        total_frames = len(reader)

    if total_frames <= 0:
        raise ValueError(f"No frames found in video: {path}")

    frame_indices = sample_frame_indices(total_frames, num_frames)
    frames = []
    for frame_idx in frame_indices:
        frame = reader.get_data(int(frame_idx))
        image = Image.fromarray(frame).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        frames.append(np.asarray(image))
    reader.close()

    video = torch.from_numpy(np.stack(frames)).float()
    video = video.permute(0, 3, 1, 2) / 127.5 - 1.0
    return video


def sample_frame_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)
    last = np.full(num_frames - total_frames, total_frames - 1, dtype=np.int64)
    return np.concatenate([np.arange(total_frames, dtype=np.int64), last], axis=0)


def infer_feature_dim(dataset: VideoVGGTFeatureDataset) -> int:
    _, _, feature_path = dataset.samples[0]
    return int(load_npz_tensor(feature_path, dataset.npz_key).shape[-1])


def training_step(
    pipe: LynxLiteWanPipeline,
    scheduler: FlowMatchEulerDiscreteScheduler,
    batch: dict,
    cfg: TrainConfig,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    video = batch["video"].to(device=device, dtype=pipe.vae.dtype)
    prompts = batch["prompt"]
    vggt_tokens = batch["vggt_tokens"].to(device=device, dtype=weight_dtype)

    with torch.no_grad():
        latents = encode_video_latents(pipe.vae, video)
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=cfg.max_sequence_length,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

    noise = torch.randn_like(latents)
    timesteps, sigmas = sample_flow_timesteps(scheduler, latents.shape[0], device, latents.dtype)
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    target = noise - latents

    model_pred = pipe.transformer(
        hidden_states=noisy_latents.to(dtype=weight_dtype),
        timestep=timesteps,
        image_embed=vggt_tokens,
        ip_scale=cfg.ip_scale,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]

    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def encode_video_latents(vae, video: torch.Tensor) -> torch.Tensor:
    video = video.permute(0, 2, 1, 3, 4)
    latent_dist = vae.encode(video).latent_dist
    latents = latent_dist.sample()

    mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    return (latents - mean) * std


def sample_flow_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
    indices = torch.randint(0, num_train_timesteps, (batch_size,), device=device)
    timesteps = indices.to(dtype=torch.float32)
    sigmas = indices.to(dtype=dtype) / float(num_train_timesteps)
    sigmas = sigmas.clamp(min=1.0 / num_train_timesteps, max=1.0)
    sigmas = sigmas.view(batch_size, 1, 1, 1, 1)
    return timesteps, sigmas


def save_checkpoint(ip_layers: torch.nn.Module, output_dir: str, step: int) -> None:
    checkpoint_dir = os.path.join(output_dir, f"step-{step:06d}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_file(ip_layers.state_dict(), os.path.join(checkpoint_dir, "ip_layers.safetensors"))
    save_file(ip_layers.state_dict(), os.path.join(output_dir, "ip_layers.safetensors"))
    print(f"saved checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
