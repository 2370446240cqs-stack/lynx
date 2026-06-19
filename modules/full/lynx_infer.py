# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import os
import torch
import numpy as np

from PIL import Image
from diffusers import UniPCMultistepScheduler

from modules.common.inference_utils import SubjectInfo, VideoStyleInfo, dtype_mapping
from utix import Logger, save_numpy_to_mp4, save_numpy_to_webp
from .lynx_pipeline import LynxWanPipeline

logger = Logger(__name__)


class LynxWanInfer():
    def __init__(
        self,
        adapter_path: str = None,
        base_model_path: str = None,
        pipe: LynxWanPipeline = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "bf16",
        enable_nsfw_check: bool = False,
    ) -> None:
        logger.info("Initializing pipeline")
        #初始化LynxWanPipeline对象，核心的视频生成流水线
        if adapter_path is not None:
            assert pipe is None, "Model path is already provided!"
            dtype = dtype_mapping[dtype] if isinstance(dtype, str) else dtype

            loaded = LynxWanInfer.load_pipeline_and_models(
                adapter_path, base_model_path, device=device, dtype=dtype
            )
            self.pipe = loaded["pipe"]

        else:
            assert pipe, "Should provide model path or a pipe object!"
            self.pipe = pipe

        assert self.pipe, "Init pipeline failed!"

        #安全性检查模型NSFW(nsfw_classifier)
        self.enable_nsfw_check = enable_nsfw_check
        if self.enable_nsfw_check:
            logger.info("Initializing NSFW classifier")
            from transformers import pipeline
            self.nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
        else:
            logger.warning("NSFW check is disabled")
            self.nsfw_classifier = None

    def generate_t2v(
        self,
        subject_info: SubjectInfo,
        style_info: VideoStyleInfo,
        output_dir: str,
        ext: str = "mp4",
        fps: int = 16,
        **override_kwargs
    ) -> None:
        logger.info(f"Generating video for style: {style_info.style_name}")

        # Override the style info args
        #临时修改生成参数
        for k in override_kwargs:
            setattr(style_info, k, override_kwargs[k])

        if getattr(subject_info, "feature_tokens", None) is not None:
            feature_tokens = torch.as_tensor(subject_info.feature_tokens)
            if feature_tokens.ndim == 2:
                feature_tokens = feature_tokens.unsqueeze(0)

            first_processor = self.pipe.transformer.blocks[0].attn2.processor
            expected_dim = getattr(getattr(first_processor, "to_k_ip", None), "in_features", None)
            if expected_dim is not None and feature_tokens.shape[-1] != expected_dim:
                raise ValueError(
                    "VGGT-Omega feature dimension does not match the full IP-adapter: "
                    f"got {feature_tokens.shape[-1]}, expected {expected_dim}. "
                    "The released full Lynx adapter is built for its own resampler output. "
                    "Train/provide a VGGT-Omega-to-IP adapter or projection before using this path."
                )

            feature_tokens = feature_tokens.to(device=self.pipe.device, dtype=self.pipe.dtype)
            ip_hidden_states = [feature_tokens]
            ip_hidden_states_uncond = [torch.zeros_like(feature_tokens)]

        elif hasattr(self.pipe, "resampler"):
            #subject_info中已经提取了人脸特征向量，这一步就是把subject_info.face_embeds这个numpy数组转换成tensor
            arcface_embed = torch.from_numpy(subject_info.face_embeds)
            #将数据搬运到正确的硬件（GPU）上，并转换成一致的精度格式。
            arcface_embed = arcface_embed.to(device=self.pipe.device, dtype=self.pipe.dtype)
            #经过 [None, None, :] 处理后，原本形状为 (512,) 的数据，变成了 (1, 1, 512)。满足Transformer的输入维度
            arcface_embed = arcface_embed[None,None,:]

            # 提取特征向量 -> 送入重采样器 -> 生成条件/非条件特征
            face_embeds = self.pipe.resampler(arcface_embed)
            ip_hidden_states = [face_embeds]
            face_embeds_uncond = self.pipe.resampler(arcface_embed * 0)
            ip_hidden_states_uncond = [face_embeds_uncond]
        else:
            ip_hidden_states = None
            ip_hidden_states_uncond = None

        if hasattr(self.pipe.transformer.blocks[0].attn1.processor, "to_k_ref") and subject_info.landmarks is not None:

            from ..common.face_utils import align_face
            aligned_face_image_pil = align_face(subject_info.image_pil, subject_info.landmarks, extend_face_crop=True, face_size=256)
            aligned_face_image_np = np.array(aligned_face_image_pil)
            ref_generator = torch.Generator().manual_seed(style_info.seed + 1) if style_info.seed >= 0 else None
            ref_buffer = self.pipe.encode_reference_images([aligned_face_image_pil], generator=ref_generator)
            ref_generator = torch.Generator().manual_seed(style_info.seed + 1) if style_info.seed >= 0 else None
            ref_buffer_uncond = self.pipe.encode_reference_images([aligned_face_image_pil], drop=True, generator=ref_generator)
        else:
            if hasattr(self.pipe.transformer.blocks[0].attn1.processor, "to_k_ref"):
                logger.warning("Skipping Ref-adapter because no face landmarks are available for this subject")
            ref_buffer = None
            ref_buffer_uncond = None
        
        generator = torch.Generator().manual_seed(style_info.seed) if style_info.seed >= 0 else None

        #调用Wan2.1来生成视频，其中attention_kwargs和attention_kwargs_uncond为注入到模型中的人脸特征
        result_frames = self.pipe(
            prompt=style_info.prompt,
            negative_prompt=style_info.negative_prompt,
            height=style_info.height,
            width=style_info.width,
            num_inference_steps=style_info.num_inference_steps,
            num_frames=style_info.num_frames,
            guidance_scale=style_info.guidance_scale,
            guidance_scale_i=getattr(style_info, "guidance_scale_i", None),
            generator=generator, 
            output_type="pil",

            #这两个参数很重要，将重采样后的人脸特征和参考特征注入了生成流程（论文中提到的resampler输出的16个5120维向量与16个register_token拼接在哪里？
            #在resampler.forword中，resampler输入时就拼接了
            attention_kwargs={"ip_hidden_states": ip_hidden_states, "ip_scale": style_info.ip_scale, "ref_buffer": ref_buffer, "ref_scale": style_info.ref_scale},
            attention_kwargs_uncond={"ip_hidden_states": ip_hidden_states_uncond, "ip_scale": style_info.ip_scale, "ref_buffer": ref_buffer_uncond, "ref_scale": style_info.ref_scale},
        ).frames[0]

        nsfw_detected = False
        if self.enable_nsfw_check:
            # Safety check
            logger.info("Running first NSFW classifier")
            nsfw_scores = []
            for frame in result_frames:
                nsfw_score = 0.0
                for item in self.nsfw_classifier(frame):
                    if item['label'] == 'nsfw':
                        nsfw_score = item['score']
                nsfw_scores.append(nsfw_score)
            nsfw_score = max(nsfw_scores)

            logger.info("Running second NSFW classifier")
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            import opennsfw2 as n2
            nsfw_scores2 = n2.predict_images(result_frames)
            nsfw_score2 = max(nsfw_scores2)
            nsfw_detected = nsfw_score >= 0.85 or nsfw_score2 >= 0.75
        else:
            logger.warning("Skipping NSFW check")

        if nsfw_detected:
            logger.warning("NSFW detected! Not saving video")
        else:
            result_frames = np.array(result_frames)
            out_video_name = "{sub}/{style}-fr{frame}-s{seed}.{ext}".format(
                sub=subject_info.name,
                style=style_info.style_name,
                frame=style_info.num_frames,
                seed=style_info.seed,
                ext=ext
            )

            out_video_path = os.path.join(output_dir, out_video_name)
            os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

            self._export_video(result_frames, out_video_path, fps)

    @staticmethod
    def load_pipeline_and_models(
        adapter_path: str,
        base_model_path: str = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        save_memory: bool = False
    ) -> LynxWanPipeline:

        loaded = {}
        
        pipe = LynxWanPipeline.from_pretrained(base_model_path, torch_dtype=dtype)

        # Use UniPCMultistepScheduler for potentially better quality.
        pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
            base_model_path, subfolder="scheduler", torch_dtype=dtype
        )
        
        if adapter_path is not None:
            logger.info("Loading adapter layers")
            pipe.init_image_proj_modules(adapter_path, device=device, dtype=dtype)
            pipe.init_ref_adapter_modules(adapter_path)

        pipe.to(device)

        if save_memory:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        loaded["pipe"] = pipe

        return loaded

    def _export_video(self, frames: np.ndarray, path: str, fps: int = 8) -> None:
        if path.endswith("webp"):
            # Export to webp for easy preview
            save_numpy_to_webp(frames, path, fps=fps)
        else:
            save_numpy_to_mp4(frames, path, fps=fps)
