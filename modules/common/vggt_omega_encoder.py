# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional

import torch


VGGTFeatureKind = Literal[
    "registers",
    "camera_register",
    "camera",
    "text_alignment_embedding",
    "text_alignment_token",
]


@dataclass
class VGGTOmegaConfig:
    repo_path: str = "../vggt-omega"
    checkpoint_path: str = ""
    image_resolution: int = 512
    preprocess_mode: str = "balanced"
    feature_kind: VGGTFeatureKind = "registers"


class VGGTOmegaFeatureExtractor:
    """Extract token features from a local VGGT-Omega checkout."""

    def __init__(self, config: VGGTOmegaConfig, device: str = "cuda") -> None:
        if not config.checkpoint_path:
            raise ValueError("--vggt_omega_checkpoint is required when --feature_source vggt_omega")

        repo_path = os.path.abspath(config.repo_path)
        checkpoint_path = os.path.abspath(config.checkpoint_path)
        if not os.path.isdir(repo_path):
            raise FileNotFoundError(f"VGGT-Omega repo not found: {repo_path}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from vggt_omega.models import VGGTOmega

        enable_alignment = config.feature_kind in {"text_alignment_embedding", "text_alignment_token"}
        self.model = VGGTOmega(enable_alignment=enable_alignment).eval()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        self.config = config
        self.device = device

    @torch.inference_mode()
    def __call__(self, image_path: str) -> torch.Tensor:
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        images = load_and_preprocess_images(
            [image_path],
            mode=self.config.preprocess_mode,
            image_resolution=self.config.image_resolution,
        ).to(self.device)

        predictions = self.model(images)
        feature_kind = self.config.feature_kind

        if feature_kind in {"registers", "camera_register", "camera"}:
            tokens = predictions["camera_and_register_tokens"][:, 0]
            if feature_kind == "registers":
                tokens = tokens[:, 1:]
            elif feature_kind == "camera":
                tokens = tokens[:, :1]
        elif feature_kind in {"text_alignment_embedding", "text_alignment_token"}:
            tokens = predictions[feature_kind].unsqueeze(1)
        else:
            raise ValueError(f"Unsupported VGGT-Omega feature kind: {feature_kind}")

        return tokens.detach().float().cpu()


def extract_vggt_omega_tokens(
    image_path: str,
    checkpoint_path: str,
    repo_path: str = "../vggt-omega",
    image_resolution: int = 512,
    preprocess_mode: str = "balanced",
    feature_kind: VGGTFeatureKind = "registers",
    device: str = "cuda",
) -> torch.Tensor:
    extractor = VGGTOmegaFeatureExtractor(
        VGGTOmegaConfig(
            repo_path=repo_path,
            checkpoint_path=checkpoint_path,
            image_resolution=image_resolution,
            preprocess_mode=preprocess_mode,
            feature_kind=feature_kind,
        ),
        device=device,
    )
    return extractor(image_path)
