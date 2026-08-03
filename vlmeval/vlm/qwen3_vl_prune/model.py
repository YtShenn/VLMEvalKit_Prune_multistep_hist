from __future__ import annotations

import ast
import logging
import os
import tempfile
import time
import warnings
import re

import torch
from PIL import Image, ImageDraw, ImageFont

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory, listinstr
# from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
# from .modeling_qwen3_vl_self import Qwen3VLForConditionalGeneration
# from .modeling_qwen3_vl_self_spvlm import Qwen3VLForConditionalGeneration
#===========roi========
from .modeling_qwen3_vl_self_spvlm_after import Qwen3VLForConditionalGeneration, ScreenSpotROICropEnsembler
# from .modeling_qwen3_vl_self_spvlm_my import Qwen3VLForConditionalGeneration
#=========roi end=========
# from .modeling_qwen3_vl_self_spvlm_ky import Qwen3VLForConditionalGeneration

VLLM_MAX_IMAGE_INPUT_NUM = 24


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model."""
    path_parts = model_path.split('/')
    non_moe_patterns = ['2B','4B','8B','32B']
    for part in path_parts:
        if any(pattern in part for pattern in non_moe_patterns):
            return False
    return True


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')

#=======================timer================================
class _StageTimer:
    def __init__(self, use_cuda_events: bool = True, sync_cuda: bool = False) -> None:
        self.use_cuda_events = bool(use_cuda_events) and torch.cuda.is_available()
        self.sync_cuda = bool(sync_cuda)
        self._handles = []
        self._cpu_stacks = {}
        self._cuda_events = {}
        self.seconds = {}

    def add_module(self, key: str, module: torch.nn.Module | None) -> None:
        if module is None:
            return
        if self.use_cuda_events:
            def pre_hook(_, __):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self._cuda_events.setdefault(key, []).append((start, end))

            def post_hook(_, __, ___):
                evs = self._cuda_events.get(key)
                if not evs:
                    return
                evs[-1][1].record()

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))
        else:
            def pre_hook(_, __):
                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._cpu_stacks.setdefault(key, []).append(time.perf_counter())

            def post_hook(_, __, ___):
                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                st = self._cpu_stacks.get(key)
                if not st:
                    return
                start_t = st.pop()
                self.seconds[key] = self.seconds.get(key, 0.0) + (time.perf_counter() - start_t)

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))

    def finalize(self) -> None:
        if not self.use_cuda_events:
            return
        torch.cuda.synchronize()
        for key, evs in self._cuda_events.items():
            total_ms = 0.0
            for start, end in evs:
                total_ms += float(start.elapsed_time(end))
            self.seconds[key] = self.seconds.get(key, 0.0) + total_ms / 1000.0

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


def _pick_first_attr(module: torch.nn.Module, names: list[str]) -> torch.nn.Module | None:
    for n in names:
        m = getattr(module, n, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def _pick_vision_and_llm_modules(model: torch.nn.Module) -> tuple[torch.nn.Module | None, torch.nn.Module | None, str, str]:
    vision = _pick_first_attr(model, ['vision_model', 'visual', 'vision_tower', 'vision_encoder', 'image_encoder', 'vision'])
    llm = _pick_first_attr(model, ['language_model', 'text_model', 'llm', 'transformer', 'decoder'])
    if llm is None and vision is not None:
        llm = _pick_first_attr(model, ['model'])

    vision_name = vision.__class__.__name__ if vision is not None else 'None'
    llm_name = llm.__class__.__name__ if llm is not None else 'None'
    return vision, llm, vision_name, llm_name
#=======================timer================================

def _parse_index_filter(expr: str | None) -> tuple[set[str], list[tuple[int, int]]]:
    """Parse index filter expression like '1,5,10-20' into exact string ids and numeric ranges.

    Returns:
      - exact: set of string indices that must match exactly (e.g. {'1','abc'})
      - ranges: list of inclusive integer ranges [(10,20)]
    """
    exact: set[str] = set()
    ranges: list[tuple[int, int]] = []
    if not expr:
        return exact, ranges
    parts = [p.strip() for p in expr.split(',') if p.strip()]
    for p in parts:
        # range like 10-20
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            ranges.append((a, b))
            continue
        # single token: keep as exact string, but also allow numeric match through ranges later
        exact.add(p)
    return exact, ranges

def _index_allowed(sample_index: str | None, expr: str | None) -> bool:
    """Return True if sample_index matches filter expr. Empty expr means allow all."""
    if not expr:
        return True
    if sample_index is None:
        return False
    exact, ranges = _parse_index_filter(expr)
    if sample_index in exact:
        return True
    # numeric range match if sample_index is int-like
    if sample_index.isdigit():
        v = int(sample_index)
        for a, b in ranges:
            if a <= v <= b:
                return True
    return False


#===========roi========
def _compute_generation_confidence(generation_output, input_len: int) -> float | None:
    """Estimate confidence from generated token probabilities."""
    try:
        sequences = generation_output.sequences
        scores = generation_output.scores
        if sequences is None or scores is None or len(scores) == 0:
            return None
        generated_ids = sequences[:, input_len:]
        if generated_ids.numel() == 0:
            return None
        token_probs = []
        max_steps = min(len(scores), generated_ids.shape[1])
        for step_idx in range(max_steps):
            step_logits = scores[step_idx]
            step_probs = torch.softmax(step_logits.float(), dim=-1)
            chosen = generated_ids[:, step_idx].unsqueeze(-1)
            chosen_prob = step_probs.gather(-1, chosen).squeeze(-1)
            token_probs.append(chosen_prob)
        if not token_probs:
            return None
        all_probs = torch.stack(token_probs, dim=1)
        return float(all_probs.mean().item())
    except Exception:
        return None
#=========roi end=========


def _parse_all_points(response: str) -> list[tuple[float, float]]:
    """Extract all coordinates from model response in order."""
    if not isinstance(response, str):
        return []
    pattern = r"x\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)"
    out = []
    for m in re.finditer(pattern, response):
        try:
            out.append((float(m.group(1)), float(m.group(2))))
        except Exception:
            continue
    return out


def _compute_generation_confidence_per_sample(generation_output, input_lens) -> list[float | None]:
    """Estimate confidence for each sample in a batched generation."""
    try:
        sequences = generation_output.sequences
        scores = generation_output.scores
        if sequences is None or scores is None or len(scores) == 0:
            batch = int(sequences.shape[0]) if sequences is not None else 0
            return [None] * batch

        batch = int(sequences.shape[0])
        if isinstance(input_lens, int):
            input_lens = [int(input_lens)] * batch
        else:
            input_lens = [int(x) for x in input_lens]
            if len(input_lens) != batch:
                input_lens = [int(input_lens[0])] * batch if len(input_lens) > 0 else [0] * batch

        confidences: list[float | None] = []
        for b in range(batch):
            input_len_b = max(0, min(int(input_lens[b]), int(sequences.shape[1])))
            generated_ids_b = sequences[b, input_len_b:]
            if generated_ids_b.numel() == 0:
                confidences.append(None)
                continue

            token_probs_b = []
            max_steps = min(len(scores), int(generated_ids_b.shape[0]))
            for step_idx in range(max_steps):
                step_logits = scores[step_idx][b:b+1]
                step_probs = torch.softmax(step_logits.float(), dim=-1)
                chosen = generated_ids_b[step_idx].view(1, 1)
                chosen_prob = step_probs.gather(-1, chosen).squeeze(-1)
                token_probs_b.append(chosen_prob)

            if not token_probs_b:
                confidences.append(None)
                continue

            all_probs_b = torch.stack(token_probs_b, dim=0)
            confidences.append(float(all_probs_b.mean().item()))

        return confidences
    except Exception:
        try:
            batch = int(getattr(generation_output.sequences, 'shape', [0])[0])
        except Exception:
            batch = 0
        return [None] * batch


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems else -1
        if max_gpu_mem <= 0:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                max_gpu_mem = int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
            else:
                logging.warning(
                    'Unable to query free GPU memory from nvidia-smi; continuing without memory-based guard.'
                )

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=8,
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,
            )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='sdpa'
                )
                # self.model = AutoModelForImageTextToText.from_pretrained(
                #     model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                # )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    #===========roi========
    def _find_first_local_image_path(self, message: list[dict]) -> str | None:
        for s in message:
            if isinstance(s, dict) and s.get("type") == "image":
                value = s.get("value")
                # print("====_find_first_local_image_path====")
                # print("value: ",value)
                if isinstance(value, str) and os.path.exists(value):
                    return value
        return None

    def _build_crop_message(self, message: list[dict], crop_path: str, crop_size: tuple[int, int] | None = None) -> list[dict]:
        crop_message = []
        replaced = False
        w = h = None
        if isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
            try:
                w, h = int(crop_size[0]), int(crop_size[1])
            except Exception:
                w, h = None, None

        roi_instruction = (
            "You are a GUI agent. You are given a task and a cropped screenshot. You need to perform pyautogui click/moveTo action to complete the task. The provided image is a cropped patch (not the full screenshot). "
            "Predict the click point in THIS cropped image's local pixel coordinates. "
            "Use top-left as (0,0). "
            f"Valid x range: [0, {w}), valid y range: [0, {h}). "
            "Return only in format: x=<number>, y=<number>. "
            "Do not output full-screen/global coordinates."
            if (w is not None and h is not None)
            else "\n[ROI_CROP_MODE] The provided image is a cropped patch (not the full screenshot). "
                 "Predict the click point in THIS cropped image's local pixel coordinates. "
                 "Use top-left as (0,0). Return only in format: x=<number>, y=<number>. "
                 "Do not output full-screen/global coordinates."
        )

        injected = False
        for s in message:
            if isinstance(s, dict):
                item = dict(s)
                if not replaced and item.get("type") == "image":
                    item["value"] = crop_path
                    replaced = True
                # if item.get("type") == "text" and isinstance(item.get("value"), str) and not injected:
                #     item["value"] = item["value"] + roi_instruction
                #     injected = True
                crop_message.append(item)
            else:
                crop_message.append(s)

        # if not injected:
        #     crop_message.append({"type": "text", "value": roi_instruction.strip()})
        # print("crop_message: ", crop_message)
        return crop_message

    def _build_multi_crop_single_message(self, message: list[dict], crop_paths: list[str], crop_sizes: list[tuple[int, int]]) -> list[dict]:
        """Build one message that contains all crop images in order."""
        packed_message = []
        replaced = False

        for s in message:
            if isinstance(s, dict):
                item = dict(s)
                if item.get("type") == "image":
                    if not replaced:
                        for idx, crop_path in enumerate(crop_paths):
                            img_item = dict(item)
                            img_item["value"] = crop_path
                            img_item["crop_index"] = str(idx + 1)
                            packed_message.append(img_item)
                        replaced = True
                    continue
                packed_message.append(item)
            else:
                packed_message.append(s)

        if not replaced:
            for idx, crop_path in enumerate(crop_paths):
                packed_message.append({"type": "image", "value": crop_path, "crop_index": str(idx + 1)})

        range_desc = []
        for i, (w, h) in enumerate(crop_sizes, 1):
            range_desc.append(f"crop{i}: x in [0,{int(w)}), y in [0,{int(h)})")
        range_text = "; ".join(range_desc)

        instruction = (
            "You are given multiple cropped screenshots from the same original image. "
            "For EACH crop image in order, predict one click point in THAT crop's local pixel coordinates. "
            "Top-left is (0,0) for each crop. "
            f"Ranges: {range_text}. "
            "Output exactly one line per crop in this format: crop<idx>: x=<number>, y=<number>. "
            "Do not output global/full-screen coordinates."
        )

        replaced_text = False
        # Replace every text prompt in packed_message, including role=system.
        num=0
        for idx, s in enumerate(packed_message):
            if isinstance(s, dict) and s.get("type") == "text" and isinstance(s.get("value"), str):
                packed_message[idx]["value"] = instruction
                replaced_text = True
                num+=1
        print(f"replaced_text: ", replaced_text)
        print("num: ", num)

        if not replaced_text:
            packed_message.append({"type": "text", "value": instruction})
        # print("packed_message: ", packed_message)
        return packed_message

    def _generate_inner_transformers_batch(self, messages_batch, dataset=None, **kwargs):
        """Run a single batched forward for multiple messages (used by ROI crops)."""
        if not messages_batch:
            return []

        debug_batch = os.environ.get("VLMPRUNE_SSP_ROI_BATCH_DEBUG", "1") == "1"
        call_id = getattr(self, "_vlm_roi_batch_call_id", 0) + 1
        self._vlm_roi_batch_call_id = call_id
        return_confidence = bool(kwargs.pop("_vlm_return_confidence", False))
        is_omni = listinstr(['omni'], self.model_path.lower())
        # if debug_batch:
        #     print(
        #         f"[ROI-BATCH-INNER] call_id={call_id} requested={len(messages_batch)} return_confidence={return_confidence} is_omni={is_omni}",
        #         flush=True,
        #     )

        if is_omni:
            # if debug_batch:
            #     print(
            #         f"[ROI-BATCH-INNER] call_id={call_id} mode=fallback_serial reason=omni_model",
            #         flush=True,
            #     )
            outputs = []
            for message in messages_batch:
                out = self.generate_inner_transformers(
                    message,
                    dataset=dataset,
                    _vlm_roi_internal=True,
                    _vlm_return_confidence=return_confidence,
                    **kwargs,
                )
                outputs.append(out)
            return outputs

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("Please install it via 'pip install qwen-vl-utils'")
            raise err

        texts = []
        images_batch = []
        for message in messages_batch:
            messages = []
            if self.system_prompt is not None:
                messages.append({'role': 'system', 'content': self.system_prompt})
            messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, _video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            if videos is not None:
                # if debug_batch:
                #     print(
                #         f"[ROI-BATCH-INNER] call_id={call_id} mode=fallback_serial reason=video_input sample_idx={len(texts)}",
                #         flush=True,
                #     )
                outputs = []
                for message_single in messages_batch:
                    out = self.generate_inner_transformers(
                        message_single,
                        dataset=dataset,
                        _vlm_roi_internal=True,
                        _vlm_return_confidence=return_confidence,
                        **kwargs,
                    )
                    outputs.append(out)
                return outputs

            if images is None or len(images) == 0:
                # if debug_batch:
                #     print(
                #         f"[ROI-BATCH-INNER] call_id={call_id} mode=fallback_serial reason=no_image sample_idx={len(texts)}",
                #         flush=True,
                #     )
                outputs = []
                for message_single in messages_batch:
                    out = self.generate_inner_transformers(
                        message_single,
                        dataset=dataset,
                        _vlm_roi_internal=True,
                        _vlm_return_confidence=return_confidence,
                        **kwargs,
                    )
                    outputs.append(out)
                return outputs

            texts.append(text)
            images_batch.append(images[0])

        inputs = self.processor(
            text=texts,
            images=images_batch,
            videos=None,
            do_resize=False,
            return_tensors='pt',
            padding=True,
        )
        # if debug_batch:
        #     input_bs = int(inputs["input_ids"].shape[0]) if "input_ids" in inputs else -1
        #     input_seq = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else -1
        #     attn_shape = tuple(inputs["attention_mask"].shape) if "attention_mask" in inputs else None
        #     img_grid_shape = tuple(inputs["image_grid_thw"].shape) if "image_grid_thw" in inputs else None
        #     print(
        #         f"[ROI-BATCH-INNER] call_id={call_id} mode=batched processor_input_bs={input_bs} seq_len={input_seq} attention_mask_shape={attn_shape} image_grid_thw_shape={img_grid_shape}",
        #         flush=True,
        #     )

        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        extra_generate_kwargs = dict(kwargs)
        if return_confidence:
            extra_generate_kwargs["return_dict_in_generate"] = True
            extra_generate_kwargs["output_scores"] = True

        merged_generate_kwargs = dict(self.generate_kwargs)
        merged_generate_kwargs.update(extra_generate_kwargs)
        generated_out = self.model.generate(
            **inputs,
            **merged_generate_kwargs,
        )
        # if debug_batch:
        #     generated_seqs = generated_out.sequences if return_confidence else generated_out
        #     gen_bs = int(generated_seqs.shape[0]) if hasattr(generated_seqs, "shape") else -1
        #     gen_len = int(generated_seqs.shape[1]) if hasattr(generated_seqs, "shape") and len(generated_seqs.shape) > 1 else -1
        #     print(
        #         f"[ROI-BATCH-INNER] call_id={call_id} mode=batched generate_calls=1 output_bs={gen_bs} output_seq_len={gen_len}",
        #         flush=True,
        #     )

        if return_confidence:
            generated_ids = generated_out.sequences
            if "attention_mask" in inputs:
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            else:
                input_lens = [int(inputs["input_ids"].shape[1])] * int(generated_ids.shape[0])
            confidences = _compute_generation_confidence_per_sample(generated_out, input_lens)
        else:
            generated_ids = generated_out
            confidences = [None] * int(generated_ids.shape[0])

        if "attention_mask" in inputs:
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        else:
            input_lens = [int(inputs["input_ids"].shape[1])] * int(generated_ids.shape[0])

        trimmed_ids = []
        for l, output_ids in zip(input_lens, generated_ids):
            trimmed_ids.append(output_ids[int(l):])

        responses = self.processor.tokenizer.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if return_confidence:
            return list(zip(responses, confidences))
        return responses

    def _run_screenspot_roi_ensemble(self, message, dataset, kwargs) -> str | tuple[str, float | None]:
        image_path = self._find_first_local_image_path(message)
        # print("image_path: ", image_path)
        if image_path is None:
            return self.generate_inner_transformers(message, dataset=dataset, _vlm_roi_internal=True, **kwargs)

        roi_json_path = os.environ.get("VLMPRUNE_SSP_ROI_JSON", None)
        roi_map = ScreenSpotROICropEnsembler.load_roi_map(roi_json_path)
        boxes = ScreenSpotROICropEnsembler.find_boxes(self.model.config.text_config, message, roi_map, image_path)
        if not boxes:
            return self.generate_inner_transformers(message, dataset=dataset, _vlm_roi_internal=True, **kwargs)

        agg_method = os.environ.get("VLMPRUNE_SSP_ROI_AGG", "max_prob")
        cluster_thr = float(os.environ.get("VLMPRUNE_SSP_ROI_CLUSTER_THR", "0.08"))
        include_full = os.environ.get("VLMPRUNE_SSP_ROI_INCLUDE_FULL", "1") == "1"
        max_crops = int(os.environ.get("VLMPRUNE_SSP_ROI_MAX_CROPS", "5"))
        # Use a much smaller decoding budget for ROI mode to avoid long generation and OOM.
        roi_max_new_tokens_raw = os.environ.get("VLMPRUNE_SSP_ROI_MAX_NEW_TOKENS", "64")
        try:
            roi_max_new_tokens = int(roi_max_new_tokens_raw)
        except Exception:
            roi_max_new_tokens = 64
        if roi_max_new_tokens <= 0:
            roi_max_new_tokens = 64
        roi_kwargs = dict(kwargs)
        roi_kwargs["max_new_tokens"] = roi_max_new_tokens

        candidates = []
        created_temp_files = []
        try:
            with Image.open(image_path).convert("RGB") as orig_img:
                orig_w, orig_h = orig_img.size
                norm_boxes = []
                max_crops = min(max_crops ,len(boxes))
                print("max_crops: ", max_crops)
                for box in boxes[:max_crops]:
                    # pix_box = ScreenSpotROICropEnsembler.normalize_box(box, orig_w, orig_h)
                    pix_box = box
                    # print("pix_box: ", pix_box)
                    if pix_box is not None:
                        norm_boxes.append(pix_box)
                if not norm_boxes:
                    return self.generate_inner_transformers(message, dataset=dataset, _vlm_roi_internal=True, **kwargs)

                if include_full:
                    full_rsp, full_prob = self.generate_inner_transformers(
                    # full_rsp = self.generate_inner_transformers(
                        message,
                        dataset=dataset,
                        _vlm_roi_internal=True,
                        # _vlm_return_confidence=True,
                        **roi_kwargs,
                    )
                    # print("out: ", full_rsp)
                    full_point = ScreenSpotROICropEnsembler.parse_point(full_rsp)
                    if full_point is not None:
                        candidates.append({"point": full_point, "prob": full_prob, "response": full_rsp, "crop": None})

                #======可视化一下==========
                # vis_img = getattr(self.model.config.text_config, "_vlmeval_current_vis_image_pil", None)
                
                sample_index: str | None = None
                question: str | None = None
                for s in message:
                    if isinstance(s, dict) and s.get('type') == 'image':
                        v = s.get('value')
                        if sample_index is None and isinstance(s.get('sample_index'), str):
                            sample_index = s.get('sample_index')
                        if question is None and isinstance(s.get('question'), str):
                            question = s.get('question')
                if str(sample_index).endswith('0'):
                    # sample_index = getattr(config, "_vlmeval_current_sample_index", None) or "na"
                    # question = getattr(config, "_vlmeval_current_question", None) or "na"
                    q_slug = ScreenSpotROICropEnsembler._vlmprune_sanitize_for_filename(str(question), max_len=80)
                    base_name = os.path.basename(image_path)
                    stem = os.path.splitext(base_name)[0]
                    save_name = f"idx{sample_index}_{q_slug}_{stem}——pred.png"

                    img = orig_img.copy() 
                    
                    vis_dir = os.getenv("VLMPRUNE_ATTN_VIS_DIR", None)
                    os.makedirs(vis_dir, exist_ok=True)
                    save_path = os.path.join(
                        vis_dir,
                        save_name
                    )
                

                crop_entries = []
                for pix_box in norm_boxes:
                    crop = orig_img.crop(pix_box)
                    crop_w, crop_h = crop.size
                    with tempfile.NamedTemporaryFile(suffix=".png", prefix="ssp_roi_", delete=False) as f:
                        crop.save(f.name, format="PNG")
                        crop_path = f.name
                    created_temp_files.append(crop_path)
                    crop_message = self._build_crop_message(message, crop_path, crop_size=(crop_w, crop_h))
                    crop_entries.append(
                        {
                            "pix_box": pix_box,
                            "crop_message": crop_message,
                            "crop_w": crop_w,
                            "crop_h": crop_h,
                        }
                    )

                # Two ROI inference modes are supported:
                # 1) packed (legacy): all crops in one message, one generation.
                # 2) batch (new): one crop per message, single batched generation call.
                # Switch via env: VLMPRUNE_SSP_ROI_INFER_MODE=packed|batch (default=batch).
                infer_mode = os.environ.get("VLMPRUNE_SSP_ROI_INFER_MODE", "batch").strip().lower()
                debug_batch = os.environ.get("VLMPRUNE_SSP_ROI_BATCH_DEBUG", "1") == "1"
                # if debug_batch:
                #     print(
                #         f"[ROI-INFER] mode={infer_mode} sample_index={sample_index} num_crops={len(crop_entries)} include_full={include_full}",
                #         flush=True,
                #     )

                if infer_mode == "packed":
                    # Legacy path kept for A/B comparison and rollback safety.
                    crop_paths = []
                    for entry in crop_entries:
                        cmsg = entry["crop_message"]
                        cpath = None
                        for it in cmsg:
                            if isinstance(it, dict) and it.get("type") == "image":
                                cpath = it.get("value")
                                break
                        crop_paths.append(cpath)
                    packed_message = self._build_multi_crop_single_message(
                        message=message,
                        crop_paths=crop_paths,
                        crop_sizes=[(entry["crop_w"], entry["crop_h"]) for entry in crop_entries],
                    )
                    packed_out = self.generate_inner_transformers(
                        packed_message,
                        dataset=dataset,
                        _vlm_roi_internal=True,
                        **roi_kwargs,
                    )
                    if isinstance(packed_out, tuple):
                        packed_rsp, packed_prob = packed_out
                    else:
                        packed_rsp, packed_prob = packed_out, None

                    local_points = _parse_all_points(packed_rsp)
                    # if debug_batch:
                    #     print(
                    #         f"[ROI-INFER][packed] parsed_points={len(local_points)} response_head={str(packed_rsp)[:120]}",
                    #         flush=True,
                    #     )

                    for idx, entry in enumerate(crop_entries):
                        if idx >= len(local_points):
                            break
                        pix_box = entry["pix_box"]
                        local_point = local_points[idx]
                        global_point, abs_point = ScreenSpotROICropEnsembler.to_global_normalized(
                            self.model.config.text_config,
                            local_xy=local_point,
                            crop_xyxy=pix_box,
                            orig_size=(orig_w, orig_h),
                        )
                        vis_pred = os.environ.get('VLMPRUNE_VIS_PRED','0')
                        print(f"vis_pred: {vis_pred}, sample_index: {sample_index}")
                        if vis_pred=='1' and str(sample_index).endswith('0'):
                            draw = ImageDraw.Draw(img, mode="RGBA")
                            x0, y0, x1, y1 = pix_box
                            x_, y_ = abs_point
                            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)
                            r = 10
                            draw.ellipse([x_ - r, y_ - r, x_ + r, y_ + r], outline=(255, 0, 0, 255), width=3)
                            try:
                                font = ImageFont.truetype("arial.ttf", 20)
                            except Exception:
                                font = ImageFont.load_default()
                            text_position = (x_ - r, y_ + r + 5)
                            text_val = f"{packed_prob:.2f}" if packed_prob is not None else "NA"
                            draw.text(text_position, text_val, fill=(255, 0, 0, 255), font=font)
                            img.save(save_path)
                        candidates.append({"point": global_point, "prob": packed_prob, "response": packed_rsp, "crop": pix_box})
                else:
                    # New path: keep per-crop prompts independent while using one batched model.generate call.
                    crop_messages = [entry["crop_message"] for entry in crop_entries]
                    batch_kwargs = dict(kwargs)
                    batch_kwargs["_vlm_return_confidence"] = True
                    batch_kwargs["max_new_tokens"] = roi_max_new_tokens
                    batch_outs = self._generate_inner_transformers_batch(
                        crop_messages,
                        dataset=dataset,
                        **batch_kwargs,
                    )
                    # if debug_batch:
                    #     print(
                    #         f"[ROI-INFER][batch] enabled=True batch_size={len(crop_messages)} returned_outputs={len(batch_outs)}",
                    #         flush=True,
                    #     )

                    for idx, entry in enumerate(crop_entries):
                        if idx >= len(batch_outs):
                            break
                        crop_out = batch_outs[idx]
                        if isinstance(crop_out, tuple):
                            crop_rsp, crop_prob = crop_out
                        else:
                            crop_rsp, crop_prob = crop_out, None
                        local_point = ScreenSpotROICropEnsembler.parse_point(crop_rsp)
                        # if debug_batch:
                        #     print(
                        #         f"[ROI-INFER][batch] crop={idx+1}/{len(crop_entries)} prob={crop_prob} rsp={str(crop_rsp)[:120]}",
                        #         flush=True,
                        #     )
                        if local_point is None:
                            continue
                        pix_box = entry["pix_box"]
                        global_point, abs_point = ScreenSpotROICropEnsembler.to_global_normalized(
                            self.model.config.text_config,
                            local_xy=local_point,
                            crop_xyxy=pix_box,
                            orig_size=(orig_w, orig_h),
                        )
                        vis_pred = os.environ.get('VLMPRUNE_VIS_PRED','0')
                        print(f"vis_pred: {vis_pred}, sample_index: {sample_index}")
                        if vis_pred=='1' and str(sample_index).endswith('0'):
                            draw = ImageDraw.Draw(img, mode="RGBA")
                            x0, y0, x1, y1 = pix_box
                            x_, y_ = abs_point
                            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)
                            r = 10
                            draw.ellipse([x_ - r, y_ - r, x_ + r, y_ + r], outline=(255, 0, 0, 255), width=3)
                            try:
                                font = ImageFont.truetype("arial.ttf", 20)
                            except Exception:
                                font = ImageFont.load_default()
                            text_position = (x_ - r, y_ + r + 5)
                            text_val = f"{crop_prob:.2f}" if crop_prob is not None else "NA"
                            draw.text(text_position, text_val, fill=(255, 0, 0, 255), font=font)
                            img.save(save_path)
                            print(f"Saved visualization for sample_index {sample_index} at {save_path}")
                        candidates.append({"point": global_point, "prob": crop_prob, "response": crop_rsp, "crop": pix_box})
        finally:
            for path in created_temp_files:
                try:
                    os.remove(path)
                except Exception:
                    pass

        final_point = ScreenSpotROICropEnsembler.aggregate(candidates, method=agg_method, cluster_thr=cluster_thr)
        if final_point is None:
            response = "x=0, y=0"
            # return self.generate_inner_transformers(message, dataset=dataset, _vlm_roi_internal=True, **roi_kwargs)
        else:
            response = f"x={int(final_point[0])}, y={int(final_point[1])}"
        top_prob = None
        valid_probs = [c.get("prob") for c in candidates if c.get("prob") is not None]
        if valid_probs:
            top_prob = max(valid_probs)
        if kwargs.get("_vlm_return_confidence", False):
            return response, top_prob
        return response
    #=========roi end=========

    def generate_inner_transformers(self, message, dataset=None, **kwargs): # jingyz1
        # print("in generate_inner_transformers")
        #===========roi========
        roi_internal = bool(kwargs.pop("_vlm_roi_internal", False))
        return_confidence = bool(kwargs.pop("_vlm_return_confidence", False))
        # print("return_confidence: ", return_confidence)
        roi_enable = os.environ.get("VLMPRUNE_SSP_ROI_ENABLE", "0") == "1"
        dataset_name = (dataset or "").lower() if isinstance(dataset, str) else ""
        if roi_enable and (not roi_internal) and ("screenspot" in dataset_name):
            # print("in if 1")
            return self._run_screenspot_roi_ensemble(
                message=message,
                dataset=dataset,
                kwargs={**kwargs, "_vlm_return_confidence": True},
            )
        #=========roi end=========

        is_omni = listinstr(['omni'], self.model_path.lower())
        confidence = None
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')


        #######################292-324：实现可视化token加入的功能代码#############################
        # Expose current sample's local image paths to the model for downstream visualization/debug.
        # ScreenSpot_Pro passes local file paths in `message` items like: {'type': 'image', 'value': '/abs/path.png'}.
        # try:
        image_paths: list[str] = []
        sample_index: str | None = None
        question: str | None = None
        for s in message:
            if isinstance(s, dict) and s.get('type') == 'image':
                v = s.get('value')
                if isinstance(v, str) and os.path.exists(v):
                    image_paths.append(v)
                if sample_index is None and isinstance(s.get('sample_index'), str):
                    sample_index = s.get('sample_index')
                if question is None and isinstance(s.get('question'), str):
                    question = s.get('question')
        # Store on config to be accessible inside `modeling_qwen3_vl_self.py`
        if hasattr(self, 'model') and hasattr(self.model, 'config'):
            self.model.config.text_config._vlmeval_current_image_paths = image_paths
            self.model.config.text_config._vlmeval_current_sample_index = sample_index
            self.model.config.text_config._vlmeval_current_question = question
            # optional visualization control via env vars
            import datetime
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_sub_dir = os.environ.get('VIS_DIR', None)

            # 获取当前文件的绝对路径
            abs_path = os.path.abspath(__file__)
            # 获取该路径所属的目录
            current_dir = os.path.dirname(abs_path)
            current_dir = os.path.dirname(current_dir)
            current_dir = os.path.dirname(current_dir)
            current_dir = os.path.dirname(current_dir)
            vis_dir = f"{current_dir}/visualize_token/{vis_sub_dir}"
            vis_attn_dir = f"{current_dir}/visualize_token_attn/{vis_sub_dir}"
            # vis_dir = os.environ.get('VLMPRUNE_VIS_DIR', None)
            vis_enable = os.environ.get('VLMPRUNE_VIS_ENABLE', '0')
            attn_enable = os.environ.get('VLMPRUNE_ATTN_VIS_ENABLE', '0')
            attn_layers = os.environ.get('VLMPRUNE_ATTN_VIS_LAYERS', None)
            single_layer_attn_only = (os.environ.get("SINGLE_LAYER_ATTN_ONLY", "0") == "1") or (
                os.environ.get("VLMPRUNE_LAYER16_ATTN_ONLY", "0") == "1"
            )
            disable_pruning = single_layer_attn_only or os.environ.get("VLMPRUNE_DISABLE_PRUNING", "0") == "1"
            # # optional: only visualize selected sample indices
            # # e.g. export VLMPRUNE_VIS_INDEXES="1,5,10-20"
            # vis_indexes = os.environ.get('VLMPRUNE_VIS_INDEXES', None)
            # allow_this = _index_allowed(sample_index, vis_indexes)
            # self.model.config._vlmprune_vis_enable = (vis_enable == '1') and allow_this
            # print(f"sample_index: {sample_index}")
            if sample_index.endswith('0'):
            # if sample_index=='10':
                allow_this = True
            else:
                allow_this = False
            allow_this = True
            self.model.config.text_config._vlmprune_vis_enable = (vis_enable == '1') and allow_this
            self.model.config.text_config._vlmprune_attn_vis_enable = ((attn_enable == '1') or single_layer_attn_only) and allow_this
            # print(f"Visualization enabled: {self.model.config.text_config._vlmprune_vis_enable}")
            self.model.config.text_config._vlmprune_vis_dir = vis_dir
            self.model.config.text_config._vlmprune_attn_vis_dir = vis_attn_dir
            if single_layer_attn_only and not attn_layers:
                attn_layers = "15"
            self.model.config.text_config._vlmprune_attn_vis_layers = attn_layers
            if disable_pruning:
                self.model.config.text_config._vlmprune_disable_pruning = True
            if single_layer_attn_only:
                self.model.config.text_config._vlmprune_attn_only_exit = True
                self.model.config.text_config._vlmprune_attn_only_exit_layer = 15
            bbox = kwargs.get("bbox", None)
            img_size = kwargs.get("img_size", None)
            if bbox is None or img_size is None:
                for s in message:
                    if isinstance(s, dict):
                        if bbox is None and "bbox" in s:
                            bbox = s.get("bbox")
                        if img_size is None and "img_size" in s:
                            img_size = s.get("img_size")
            self.model.config.text_config._vlmeval_current_bbox = bbox
            self.model.config.text_config._vlmeval_current_img_size = img_size
            # print("in model.py")
            # print(f"model.config: {self.model.config}")
        # except Exception:
        #     # keep inference robust even if visualization metadata fails
        #     pass
        #######################292-324：实现可视化token加入的功能代码#############################

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            self.processor.image_processor.size['shortest_edge']=0
            print("self.processor.image_processor.size['shortest_edge']", self.processor.image_processor.size['shortest_edge'])
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            # for img in images:
            #     if hasattr(img, "size"):
            #         # print("original image size: ", img.size)
            input_scale = os.environ.get("VLMPRUNE_INPUT_SCALE", "1")
            try:
                input_scale = float(input_scale)
            except Exception:
                input_scale = 1.0
            if input_scale <= 0:
                input_scale = 1.0
            if images is not None and input_scale != 1.0:
                patch_size = 16
                merge_size = 1
                if hasattr(self, "model") and hasattr(self.model, "config"):
                    merge_size = getattr(self.model.config.vision_config, "spatial_merge_size", 1) or 1
                align_base = max(1, int(patch_size * merge_size))
                first_scale = None
                scaled_images = []
                for img in images:
                    if hasattr(img, "size"):
                        w, h = img.size
                        new_w = max(1, int(round(w * input_scale)))
                        new_h = max(1, int(round(h * input_scale)))
                        new_w = max(align_base, (new_w // align_base) * align_base)
                        new_h = max(align_base, (new_h // align_base) * align_base)
                        if first_scale is None:
                            first_scale = (new_w / w, new_h / h)
                        scaled_images.append(img.resize((new_w, new_h), Image.BILINEAR))
                    else:
                        scaled_images.append(img)
                images = scaled_images
                if hasattr(self, "model") and hasattr(self.model, "config"):
                    cfg = self.model.config.text_config
                    img_size = getattr(cfg, "_vlmeval_current_img_size", None)
                    bbox = getattr(cfg, "_vlmeval_current_bbox", None)
                    if first_scale is None:
                        first_scale = (input_scale, input_scale)
                    scale_x, scale_y = first_scale
                    if isinstance(img_size, str):
                        try:
                            img_size = ast.literal_eval(img_size)
                        except Exception:
                            img_size = None
                    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                        scaled_size = [img_size[0] * scale_x, img_size[1] * scale_y]
                        cfg._vlmeval_current_img_size = scaled_size
                        if isinstance(bbox, str):
                            try:
                                bbox = ast.literal_eval(bbox)
                            except Exception:
                                bbox = None
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            max_val = max(bbox)
                            if max_val > 1.5 and max_val <= max(img_size) + 1:
                                cfg._vlmeval_current_bbox = [
                                    bbox[0] * scale_x,
                                    bbox[1] * scale_y,
                                    bbox[2] * scale_x,
                                    bbox[3] * scale_y,
                                ]

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )

            #############缓存当前样本的视觉输入图像和网格信息，用于可视化################
            # If visualization is enabled for this sample, cache the *actual* vision input image and grid info.
            # IMPORTANT: Use `images` returned by `process_vision_info` (already aligned to patch/grid),
            # and `inputs["image_grid_thw"]` for correct token-to-pixel mapping.
            try:
                if hasattr(self, 'model') and hasattr(self.model, 'config'):
                    if bool(getattr(self.model.config.text_config, "_vlmprune_vis_enable", False)) or bool(
                        getattr(self.model.config.text_config, "_vlmprune_attn_vis_enable", False)
                    ):
                        if images is not None and len(images) > 0:
                            # store PIL image used by processor (not the original file)
                            self.model.config.text_config._vlmeval_current_vis_image_pil = images[0]
                            print(f"images[0]: {images[0]}")
                        self.model.config.text_config._vlmeval_current_image_grid_thw = inputs.get("image_grid_thw", None)
                        print(f"inputs.get('image_grid_thw', None): {inputs.get('image_grid_thw', None)}")
            except Exception:
                pass
            #############缓存当前样本的视觉输入图像和网格信息，用于可视化################
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        if bool(getattr(self.model.config.text_config, "_vlmprune_attn_only_exit", False)):
            #=======================timer================================
            stage_timing = os.getenv('VLM_STAGE_TIMING', '0') == '1'
            stage_device = os.getenv('VLM_STAGE_TIMING_DEVICE', 'auto')
            use_cuda_events = (stage_device in ['auto', 'cuda']) and torch.cuda.is_available()
            sync_cuda = os.getenv('VLM_STAGE_TIMING_SYNC', '0') == '1'
            timer = None
            total_start = None
            vision_name = 'None'
            llm_name = 'None'
            if stage_timing:
                timer = _StageTimer(use_cuda_events=use_cuda_events, sync_cuda=sync_cuda)
                vision_m, llm_m, vision_name, llm_name = _pick_vision_and_llm_modules(self.model)
                timer.add_module('vision', vision_m)
                timer.add_module('llm', llm_m)
                if sync_cuda and torch.cuda.is_available() and not use_cuda_events:
                    torch.cuda.synchronize()
                total_start = time.perf_counter()
            #=======================timer================================
            with torch.no_grad():
                _ = self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
            #=======================timer================================
            # print("stage_timing: ", stage_timing, "timer: ", timer, "total_start: ", total_start)
            
            if stage_timing and timer is not None and total_start is not None:
                if sync_cuda and torch.cuda.is_available() and not use_cuda_events:
                    torch.cuda.synchronize()
                total_s = time.perf_counter() - total_start
                timer.finalize()
                vision_s = float(timer.seconds.get('vision', 0.0))
                llm_s = float(timer.seconds.get('llm', 0.0))
                other_s = max(0.0, total_s - vision_s - llm_s)
                sample_index = None
                try:
                    sample_index = getattr(self.model.config.text_config, "_vlmeval_current_sample_index", None)
                except Exception:
                    sample_index = None
                prefix = f'[StageTiming] index={sample_index} ' if sample_index is not None else '[StageTiming] '
                print(
                    f'{prefix}total_s={total_s:.6f} vision_s={vision_s:.6f} llm_s={llm_s:.6f} other_s={other_s:.6f} '
                    f'vision_mod={vision_name} llm_mod={llm_name}',
                    flush=True,
                )
                if not hasattr(self, '_vlmeval_stage_records'):
                    self._vlmeval_stage_records = []
                self._vlmeval_stage_records.append(
                    dict(
                        sample_index=sample_index,
                        total_s=total_s,
                        vision_s=vision_s,
                        llm_s=llm_s,
                        other_s=other_s,
                        vision_mod=vision_name,
                        llm_mod=llm_name,
                    )
                )
                timer.close()
            #=======================timer================================
            
            return ""

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            #=======================timer================================
            stage_timing = os.getenv('VLM_STAGE_TIMING', '0') == '1'
            stage_device = os.getenv('VLM_STAGE_TIMING_DEVICE', 'auto')
            use_cuda_events = (stage_device in ['auto', 'cuda']) and torch.cuda.is_available()
            sync_cuda = os.getenv('VLM_STAGE_TIMING_SYNC', '0') == '1'
            timer = None
            total_start = None
            vision_name = 'None'
            llm_name = 'None'
            if stage_timing:
                timer = _StageTimer(use_cuda_events=use_cuda_events, sync_cuda=sync_cuda)
                vision_m, llm_m, vision_name, llm_name = _pick_vision_and_llm_modules(self.model)
                timer.add_module('vision', vision_m)
                timer.add_module('llm', llm_m)
                if sync_cuda and torch.cuda.is_available() and not use_cuda_events:
                    torch.cuda.synchronize()
                total_start = time.perf_counter()
            #=======================timer================================

            #===========roi========
            extra_generate_kwargs = dict(kwargs)
            if return_confidence:
                # print("in if 2")
                extra_generate_kwargs["return_dict_in_generate"] = True
                extra_generate_kwargs["output_scores"] = True
            #=========roi end=========

            # print("here")
            merged_generate_kwargs = dict(self.generate_kwargs)
            merged_generate_kwargs.update(extra_generate_kwargs)
            generated_out = self.model.generate(
                **inputs,
                **merged_generate_kwargs, # jingyz1
            )
            # print("generated_out: ", generated_out)

            #=======================timer================================
            print("stage_timing: ", stage_timing, "timer: ", timer, "total_start: ", total_start)
            
            if stage_timing and timer is not None and total_start is not None:
                if sync_cuda and torch.cuda.is_available() and not use_cuda_events:
                    torch.cuda.synchronize()
                total_s = time.perf_counter() - total_start
                timer.finalize()
                vision_s = float(timer.seconds.get('vision', 0.0))
                llm_s = float(timer.seconds.get('llm', 0.0))
                other_s = max(0.0, total_s - vision_s - llm_s)
                sample_index = None
                try:
                    sample_index = getattr(self.model.config.text_config, "_vlmeval_current_sample_index", None)
                except Exception:
                    sample_index = None
                prefix = f'[StageTiming] index={sample_index} ' if sample_index is not None else '[StageTiming] '
                print(
                    f'{prefix}total_s={total_s:.6f} vision_s={vision_s:.6f} llm_s={llm_s:.6f} other_s={other_s:.6f} '
                    f'vision_mod={vision_name} llm_mod={llm_name}',
                    flush=True,
                )
                if not hasattr(self, '_vlmeval_stage_records'):
                    self._vlmeval_stage_records = []
                self._vlmeval_stage_records.append(
                    dict(
                        sample_index=sample_index,
                        total_s=total_s,
                        vision_s=vision_s,
                        llm_s=llm_s,
                        other_s=other_s,
                        vision_mod=vision_name,
                        llm_mod=llm_name,
                    )
                )
                timer.close()
            #=======================timer================================
            
            #===========roi========
            confidence = None
            if return_confidence:
                # print("in if 3")
                confidence = _compute_generation_confidence(generated_out, int(inputs["input_ids"].shape[1]))
                generated_ids = generated_out.sequences
            else:
                generated_ids = generated_out
            #=========roi end=========

            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if hasattr(self, 'model') and hasattr(self.model, 'config'):
            match = re.search(r"x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)", response)
            if match:
                self.model.config.text_config._vlmeval_current_pred_point = [
                    float(match.group(1)),
                    float(match.group(2)),
                ]
            else:
                self.model.config.text_config._vlmeval_current_pred_point = None

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        #===========roi========
        if return_confidence:
            # print("in if 4")
            return response, confidence
        #=========roi end=========
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None, **kwargs):#syt
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset, **kwargs)#syt
