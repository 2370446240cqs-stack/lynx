# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

import os
import argparse

import torch
import numpy as np
from PIL import Image

from modules.common.face_encoder import FaceEncoderArcFace, get_landmarks_from_image
from modules.common.inference_utils import SubjectInfo, VideoStyleInfo, dtype_mapping
from modules.common.vggt_omega_encoder import extract_vggt_omega_tokens

from modules.full.lynx_infer import LynxWanInfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple single-GPU inference for Lynx (Wan + IPA + Ref)"
    )

    # Required-ish (with defaults matching README layout)
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="models/Wan2.1-T2V-14B-Diffusers",
        help="Path to Wan2.1 base model directory",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/lynx_full",
        help="Path to Lynx adapter directory (resampler/ip/ref layers)",
    )

    # Minimal inputs
    parser.add_argument(
        "--subject_image",
        type=str,
        required=True,
        help="Path to the subject image",
    )
    parser.add_argument(
        "--feature_source",
        type=str,
        default="face",
        choices=["face", "vggt_omega"],
        help="Conditioning feature source",
    )
    parser.add_argument(
        "--vggt_omega_repo_path",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vggt-omega")),
        help="Path to the local VGGT-Omega repository",
    )
    parser.add_argument(
        "--vggt_omega_checkpoint",
        type=str,
        default="",
        help="Path to a local VGGT-Omega checkpoint",
    )
    parser.add_argument(
        "--vggt_omega_resolution",
        type=int,
        default=512,
        help="VGGT-Omega preprocessing resolution",
    )
    parser.add_argument(
        "--vggt_omega_preprocess_mode",
        type=str,
        default="balanced",
        choices=["balanced", "max_size"],
        help="VGGT-Omega preprocessing mode",
    )
    parser.add_argument(
        "--vggt_omega_feature_kind",
        type=str,
        default="registers",
        choices=["registers", "camera_register", "camera", "text_alignment_embedding", "text_alignment_token"],
        help="Which VGGT-Omega output to inject as conditioning tokens",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, blurred background, static, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Optional negative prompt",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for generated video",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        choices=["mp4", "webp"],
        help="Output format",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="demo",
        help="Style name used in output filename",
    )

    # Generation parameters (defaults mirror repo examples)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--guidance_scale_i", type=float, default=2.0)
    parser.add_argument("--ip_scale", type=float, default=1.0)
    parser.add_argument("--ref_scale", type=float, default=1.0)

    # Runtime
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device (single GPU)",
    )

    return parser.parse_args()


def build_subject_info(args: argparse.Namespace) -> SubjectInfo:
    '''
    将输入的人脸图像特征提取出来
    '''
    image_path = args.subject_image
    image_pil = Image.open(image_path).convert("RGB")
    name = os.path.splitext(os.path.basename(image_path))[0]

    if args.feature_source == "vggt_omega":
        feature_device = "cuda" if args.device.startswith("cuda") else args.device
        feature_tokens = extract_vggt_omega_tokens(
            image_path=image_path,
            checkpoint_path=args.vggt_omega_checkpoint,
            repo_path=args.vggt_omega_repo_path,
            image_resolution=args.vggt_omega_resolution,
            preprocess_mode=args.vggt_omega_preprocess_mode,
            feature_kind=args.vggt_omega_feature_kind,
            device=feature_device,
        )

        return SubjectInfo(
            name=name,
            image_pil=image_pil,
            feature_tokens=feature_tokens.squeeze(0).numpy(),
            feature_source="vggt_omega",
        )

    # Landmarks
    landmarks = get_landmarks_from_image(image_pil)

    # Face embedding via ArcFace
    face_encoder = FaceEncoderArcFace()
    face_encoder.init_encoder_model("cuda" if args.device.startswith("cuda") else args.device)
    embeds = face_encoder(image_pil, need_proc=True, landmarks=landmarks)
    embeds = np.array(embeds.squeeze(0).cpu())

    return SubjectInfo(
        name=name,
        image_pil=image_pil,
        landmarks=landmarks,
        face_embeds=embeds,
        feature_source="face",
    )


def build_style_info(args: argparse.Namespace) -> VideoStyleInfo:
    '''
    规定生成的视频的参数
    '''
    if os.path.isfile(args.prompt):
        with open(args.prompt, 'r') as f:
            args.prompt = f.read().strip()

    return VideoStyleInfo(
        style_name=args.name,
        num_frames=args.num_frames,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        guidance_scale_i=args.guidance_scale_i,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
    )


def main():
    args = parse_args()

    # Prepare subject/style
    subject = build_subject_info(args)#将输入的人脸图片提取为特征向量
    '''
    return SubjectInfo(
        name=name,
        image_pil=image_pil,
        landmarks=landmarks,
        face_embeds=embeds,
    )
    '''
    style = build_style_info(args)#规定生成的视频参数（直接通过命令行指定或者通过参数文本输入）
    '''
    return VideoStyleInfo(
        style_name=args.name,
        num_frames=args.num_frames,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        guidance_scale_i=args.guidance_scale_i,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
    )
    '''

    # Init pipeline
    '''
    初始化LynxWanInfer类（核心类），定义在modules/full/lynx_infer中
    adapter_path:存放lynx-adapter的路径，默认为lynx_full（包含了ip_layers,ref_layers和resampler）
    base_model_path:基座模型路径，默认为Wan2.1-T2V-14B-Diffusers
    dtype:设置数据精度
    device:用哪个设备来推理，默认cuda:0，没有cuda就用CPU
    '''
    infer = LynxWanInfer(
        adapter_path=args.adapter_path,
        base_model_path=args.base_model_path,
        dtype=dtype_mapping[args.torch_dtype],
        device=args.device,
    )

    # Generate
    infer.generate_t2v(
        subject_info=subject,
        style_info=style,
        output_dir=args.output_dir,
        ext=args.ext,
        fps=args.fps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_i=args.guidance_scale_i,
        ip_scale=args.ip_scale,
        ref_scale=args.ref_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
