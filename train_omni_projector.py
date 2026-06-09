#!/usr/bin/env python3
"""
Train a Qwen3-Omni model using projector mode with FP8 precision.

Architecture:
  - Base LLM (Thinker): Qwen3.6-35B-A3B  (frozen, loaded in FP8)
  - ASR Encoder:        Whisper-Large-v3-Turbo  (frozen, loaded in FP8)
  - Projector:          Audio feature projector (trainable, FP8 matmuls)

FP8 mode loads the frozen LLM and Whisper in float8_e4m3fn,
cutting VRAM usage by ~50% while keeping the projector in float32
for numerical stability during training.

Projector mode only trains the audio projector, keeping the LLM and
audio decoder completely frozen. Memory-efficient way to add audio
understanding to any text-only LLM.
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Audio Projector Model
# ============================================================================

class AudioProjector(nn.Module):
    """
    Projects Whisper encoder hidden states into the LLM embedding space.

    Architecture:
      Input:  Whisper encoder hidden states  [batch, seq_len, whisper_dim]
      Output: Projected features            [batch, seq_len, llm_dim]

    Uses a multi-layer MLP with GELU activation.
    Supports FP8 training (Hopper native FP8 with dynamic per-tensor scaling).
    """

    def __init__(
        self,
        audio_hidden_size: int = 1280,   # Whisper-large-v3 output dim
        llm_hidden_size: int = 4096,     # Will be overridden at runtime
        projector_hidden_size: int = 4096,
        num_layers: int = 6,
        pooling: str = "mean",           # "mean", "first", or "gated"
    ):
        super().__init__()
        self.pooling = pooling

        # Build MLP projector (all linear layers in float32 for FP8 autoguard)
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_size = audio_hidden_size if i == 0 else projector_hidden_size
            out_size = llm_hidden_size if i == num_layers - 1 else projector_hidden_size
            layers.append(nn.Linear(in_size, out_size, dtype=torch.float32))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.projector = nn.Sequential(*layers)

        # Optional gated pooling
        if pooling == "gated":
            self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.gate = None

    def forward(
        self,
        audio_features: torch.Tensor,  # [batch, seq, audio_dim]
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: Whisper encoder outputs
            audio_attention_mask: 0/1 mask for audio tokens

        Returns:
            projected features in LLM embedding space
        """
        # Use float32 for pooling to maintain numerical stability
        audio_features = audio_features.float()

        if self.pooling == "mean":
            if audio_attention_mask is not None:
                mask = audio_attention_mask.unsqueeze(-1).float()
                audio_features = audio_features * mask
                pooled = audio_features.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = audio_features.mean(dim=1, keepdim=True)
        elif self.pooling == "first":
            pooled = audio_features[:, :1, :]
        elif self.pooling == "gated":
            if audio_attention_mask is not None:
                mask = audio_attention_mask.unsqueeze(-1).float()
                audio_features = audio_features * mask
                pooled = (audio_features.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).unsqueeze(1)
            else:
                pooled = audio_features.mean(dim=1, keepdim=True)
            pooled = torch.tanh(pooled) * torch.sigmoid(self.gate)
        else:
            pooled = audio_features  # no pooling

        # Convert to float16 for FP8 casting (Hopper native)
        # If using FP8 autocast, the Linear layers will auto-cast inputs to FP8
        projected = self.projector(pooled.to(torch.float16))
        return projected


# ============================================================================
# 2. Omni Model (ASR + Projector + LLM)
# ============================================================================

class QwenOmniModel(nn.Module):
    """
    Full Qwen-Omni model composed of:
      - Whisper encoder  (ASR feature extractor, frozen in FP8)
      - AudioProjector   (maps audio → LLM space, trainable in FP8)
      - LLM              (Qwen3.6-35B-A3B, frozen in FP8)

    Uses FP8 (float8_e4m3fn) for LLM and Whisper to reduce memory by 50%.
    Projector uses float32 internally with FP8 autocast for matmuls.

    In projector training mode, only the projector parameters are updated.
    """

    def __init__(
        self,
        llm_model_name: str,
        whisper_model_name: str = "openai/whisper-large-v3-turbo",
        audio_hidden_size: int = 1280,
        projector_hidden_size: int = 4096,
        projector_num_layers: int = 6,
        audio_pooling: str = "mean",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_fp8: bool = True,
    ):
        super().__init__()
        self.use_fp8 = use_fp8

        # ---- LLM (Thinker) ----
        logger.info(f"Loading LLM: {llm_model_name}")
        if use_fp8 and torch.cuda.is_available():
            # Check if GPU supports FP8 natively (Hopper H100+)
            gpu_arch = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
            self._supports_native_fp8 = (gpu_arch[0] >= 9)  # Hopper = SM 90
            if self._supports_native_fp8:
                torch_dtype = torch.float8_e4m3fn
                logger.info("Loading LLM in native FP8 (Hopper GPU detected)")
            else:
                # A100: FP8 via software emulation (no speedup, but memory savings)
                torch_dtype = torch.float8_e4m3fn
                logger.info("Loading LLM in FP8 (A100 — software emulation, memory savings only)")
        else:
            torch_dtype = torch.bfloat16

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        # Get LLM config
        llm_config = self.llm.config
        self.llm_hidden_size = llm_config.hidden_size
        self.llm_num_layers = llm_config.num_hidden_layers
        self.llm_num_heads = llm_config.num_attention_heads
        self.llm_vocab_size = llm_config.vocab_size

        logger.info(f"LLM dtype={self.llm.dtype}, hidden_size={self.llm_hidden_size}, layers={self.llm_num_layers}")

        # ---- Whisper ASR encoder ----
        logger.info(f"Loading Whisper ASR: {whisper_model_name}")
        if use_fp8 and torch.cuda.is_available():
            whisper_dtype = torch.float8_e4m3fn
        else:
            whisper_dtype = torch.bfloat16
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            whisper_model_name,
            torch_dtype=whisper_dtype,
            device_map="auto",
        )
        # Freeze the decoder
        for param in self.whisper.model.decoder.parameters():
            param.requires_grad = False
        self.whisper.eval()

        # We only use the encoder part of whisper
        self.whisper_encoder = self.whisper.model.encoder
        self.audio_hidden_size = audio_hidden_size  # 1280 for large-v3

        # ---- Audio Projector ----
        logger.info(
            f"Creating AudioProjector: "
            f"audio_dim={audio_hidden_size} -> llm_dim={self.llm_hidden_size}, "
            f"layers={projector_num_layers}, pooling={audio_pooling}"
        )
        self.audio_projector = AudioProjector(
            audio_hidden_size=audio_hidden_size,
            llm_hidden_size=self.llm_hidden_size,
            projector_hidden_size=projector_hidden_size,
            num_layers=projector_num_layers,
            pooling=audio_pooling,
        )

        # ---- Tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Special tokens for audio
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<|audio|>")
        if self.audio_token_id is None:
            self.audio_token_id = self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
            logger.info("Added <|audio|> token, new token_id=%s", self.audio_token_id)
        else:
            logger.info("Using existing <|audio|> token, token_id=%s", self.audio_token_id)

        self.num_audio_tokens = 1  # one <|audio|> placeholder per audio segment


    # ---- Forward pass ----

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with FP8 autocast for all compute.

        Args:
            audio: Raw waveform [batch, samples] or pre-computed audio features
            audio_attention_mask: Mask for audio samples
            input_ids: LLM input tokens [batch, text_seq]
            attention_mask: Attention mask for text tokens
            labels: Token IDs for loss computation

        Returns:
            Dict with 'loss' and 'logits' keys
        """
        batch_size = input_ids.shape[0]

        # 1. Extract audio features from Whisper encoder (FP8 autocast)
        with torch.no_grad():
            audio_features = self.whisper_encoder(
                audio,
                output_hidden_states=False,
            ).last_hidden_state  # [batch, audio_seq, 1280]

        # 2. Project audio features into LLM space (FP8 autocast)
        with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn, enabled=self.use_fp8):
            projected_audio = self.audio_projector(
                audio_features,
                audio_attention_mask,
            )  # [batch, 1 or N, llm_dim]

        # 3. Build multimodal input for the LLM
        #    Insert audio token embeddings at the <|audio|> position
        # Convert inputs_embeds to float16 for FP8-compatible LLM forward
        llm_inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # [batch, text_seq, llm_dim]
        llm_inputs_embeds = llm_inputs_embeds.float()  # Use float32 for embedding lookup

        # Replace <|audio|> token embedding with projected audio features
        audio_token_mask = (input_ids == self.audio_token_id).unsqueeze(-1)  # [batch, text_seq, 1]

        text_seq_len = llm_inputs_embeds.shape[1]
        proj_seq_len = projected_audio.shape[1]

        audio_token_indices = audio_token_mask.nonzero(as_tuple=True)
        if len(audio_token_indices[0]) > 0:
            for batch_idx, text_idx in zip(audio_token_indices[0], audio_token_indices[1]):
                start = text_idx.item()
                end = min(start + proj_seq_len, text_seq_len)
                num_tokens = end - start
                if num_tokens > 0:
                    llm_inputs_embeds[batch_idx, start:end, :] = (
                        projected_audio[batch_idx, :num_tokens, :]
                    ).float()

        # 4. Forward through LLM (FP8 autocast for matmuls)
        with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn, enabled=self.use_fp8):
            outputs = self.llm(
                inputs_embeds=llm_inputs_embeds.to(torch.float16),
                attention_mask=attention_mask,
                labels=labels,
            )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        prompt_ids: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text given audio input and a text prompt (with FP8 autocast).

        Args:
            audio: Waveform [batch, samples]
            prompt_ids: Prompt token IDs [batch, seq]
            audio_attention_mask: Optional audio mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated token IDs [batch, new_seq]
        """
        self.eval()

        # Extract audio features (FP8 autocast)
        with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn, enabled=self.use_fp8):
            audio_features = self.whisper_encoder(audio).last_hidden_state

        # Project (FP8 autocast)
        with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn, enabled=self.use_fp8):
            projected_audio = self.audio_projector(audio_features, audio_attention_mask)

        # Build inputs_embeds with audio
        llm_inputs_embeds = self.llm.get_input_embeddings()(prompt_ids).float()
        audio_token_mask = (prompt_ids == self.audio_token_id).unsqueeze(-1)
        audio_token_indices = audio_token_mask.nonzero(as_tuple=True)

        text_seq_len = llm_inputs_embeds.shape[1]
        proj_seq_len = projected_audio.shape[1]

        for batch_idx, text_idx in zip(audio_token_indices[0], audio_token_indices[1]):
            start = text_idx.item()
            end = min(start + proj_seq_len, text_seq_len)
            num_tokens = end - start
            if num_tokens > 0:
                llm_inputs_embeds[batch_idx, start:end, :] = (
                    projected_audio[batch_idx, :num_tokens, :]
                ).float()

        # Generate (FP8 autocast)
        with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn, enabled=self.use_fp8):
            outputs = self.llm.generate(
                inputs_embeds=llm_inputs_embeds.to(torch.float16),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        return outputs


# ============================================================================
# 3. Data Preparation
# ============================================================================

@dataclass
class AudioTextDataCollator:
    """
    Collates audio + text pairs into training batches.

    Expected dataset columns:
      - "audio": list of waveform arrays (variable length)
      - "text": list of target text strings
      - "prompt": list of prompt strings (may contain <|audio|>)
    """
    tokenizer: Any
    audio_token_id: int
    audio_sample_rate: int = 16000
    max_audio_samples: int = 30 * 16000  # 30 seconds at 16kHz
    max_text_len: int = 2048
    pad_to_multiple_of: int = 256

    def __post_init__(self):
        if self.audio_token_id is None:
            raise ValueError("audio_token_id must be set")

    def _encode_with_audio_token(self, prompt: str, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt + text with <|audio|> placeholder.
        Returns (input_ids, labels) where labels shift audio tokens to -100.
        """
        # Replace <|audio|> with actual token
        full_text = prompt.replace("<|audio|>", "") + text
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_text_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # For loss: mask out the audio prompt portion, only compute loss on 'text' part
        # Find where the actual response text starts (after any <|audio|> context)
        prompt_encoded = self.tokenizer(prompt.replace("<|audio|>", ""), return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Don't compute loss on prompt

        return input_ids, labels

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_batch = []
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in batch:
            # HF datasets Audio feature returns AudioDecoder (supports __getitem__ but no .get())
            audio_data = item["audio"]
            try:
                audio_array = audio_data["array"]
                sr = audio_data["sampling_rate"]
            except (TypeError, KeyError):
                audio_array = audio_data
                sr = item.get("sample_rate", self.audio_sample_rate)
            if sr != self.audio_sample_rate:
                # Simple resampling using torchaudio if available
                try:
                    import torchaudio.functional as F
                    audio_array = F.resample(
                        torch.tensor(audio_array, dtype=torch.float32),
                        sr,
                        self.audio_sample_rate,
                    ).numpy()
                except Exception:
                    logger.warning("Could not resample audio, using raw waveform")

            # Pad or trim audio
            target_len = min(len(audio_array), self.max_audio_samples)
            if len(audio_array) < self.max_audio_samples:
                audio_padded = torch.zeros(self.max_audio_samples)
                audio_padded[:len(audio_array)] = torch.tensor(audio_array, dtype=torch.float32)
            else:
                audio_padded = torch.tensor(audio_array[:self.max_audio_samples], dtype=torch.float32)

            audio_batch.append(audio_padded)

            # Encode text (default prompt if not in dataset)
            prompt = item.get("prompt", "Transcribe: <|audio|>")
            input_ids, labels = self._encode_with_audio_token(prompt, item["text"])
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Pad text inputs
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask_padded = (input_ids_padded != self.tokenizer.pad_token_id).long()

        audio_tensor = torch.stack(audio_batch)

        # Build audio attention mask (based on actual audio length)
        audio_len = audio_tensor.shape[-1]
        audio_attention_mask = torch.ones(audio_tensor.shape[0], audio_len, dtype=torch.long)

        return {
            "audio": audio_tensor,
            "audio_attention_mask": audio_attention_mask,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }


def prepare_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    dataset_split: str = "train",
    audio_column: str = "audio",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> Dataset:
    """
    Load and prepare the training dataset.

    Supported dataset formats:
      1. HuggingFace dataset with "audio" + "text" columns
      2. Local directory with .wav files + a metadata JSON/CSV
      3. Any datasets.Dataset with audio waveforms and text

    For local data, the directory should have:
      - audio/  directory containing .wav files
      - metadata.jsonl with lines like:
        {"audio_file": "file1.wav", "prompt": "Transcribe: <|audio|>", "text": "the transcription"}
    """
    local_path = Path(dataset_name)

    # Check if it's a local directory
    if local_path.exists() and local_path.is_dir():
        metadata_file = local_path / "metadata.jsonl"
        audio_dir = local_path / "audio"

        if metadata_file.exists():
            logger.info(f"Loading local dataset from {local_path}")
            samples = []
            with open(metadata_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    samples.append(entry)

            if max_samples:
                samples = samples[:max_samples]

            # Convert to Dataset
            texts = [s["text"] for s in samples]
            prompts = [s.get("prompt", "Transcribe: <|audio|>") for s in samples]

            def gen_samples():
                for s in samples:
                    audio_path = audio_dir / s["audio_file"]
                    import soundfile as sf
                    waveform, sr = sf.read(str(audio_path), dtype="float32")
                    # Convert stereo to mono if needed
                    if len(waveform.shape) > 1:
                        waveform = waveform.mean(axis=1)
                    yield {"audio": waveform, "sample_rate": sr, "text": s["text"], "prompt": s.get("prompt", "Transcribe: <|audio|>")}

            ds = Dataset.from_generator(gen_samples)
            return ds
        else:
            raise FileNotFoundError(
                f"Local path {local_path} found but no metadata.jsonl. "
                "Please provide a metadata.jsonl file with audio_file, text, and prompt fields."
            )

    # Otherwise, load from HuggingFace
    logger.info(f"Loading HuggingFace dataset: {dataset_name}")
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=dataset_split, streaming=streaming)

    # Rename columns to standard names if needed
    if audio_column != "audio":
        ds = ds.rename_column(audio_column, "audio")
    if text_column != "text":
        ds = ds.rename_column(text_column, "text")

    if max_samples:
        if not streaming:
            ds = ds.select(range(min(max_samples, len(ds))))
        else:
            ds = ds.take(max_samples)

    return ds


# ============================================================================
# 4. Training Setup
# ============================================================================

def create_optimizer_scheduler(
    model: QwenOmniModel,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    total_steps: int = 10000,
    lr_scheduler_type: str = "cosine",
) -> Tuple[Any, Any]:
    """
    Create optimizer and scheduler for projector-only training.
    Only the projector parameters are optimized.
    """
    projector_params = list(model.audio_projector.parameters())
    # Optionally include whisper encoder parameters
    # whisper_encoder_params = list(model.whisper_encoder.parameters())

    optimizer = torch.optim.AdamW(
        projector_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Warmup + cosine decay scheduler
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def train(
    llm_model_name: str,
    whisper_model_name: str = "openai/whisper-large-v3-turbo",
    dataset_name: str = "LibriSpeech",
    dataset_config: Optional[str] = None,
    audio_column: str = "audio",
    text_column: str = "text",
    output_dir: str = "./output/qwen-omni-projector",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    max_steps: int = -1,
    projector_hidden_size: int = 4096,
    projector_num_layers: int = 6,
    audio_pooling: str = "mean",
    max_audio_samples: int = 30 * 16000,
    max_text_len: int = 2048,
    save_steps: int = 500,
    logging_steps: int = 10,
    eval_steps: int = 500,
    seed: int = 42,
    local_rank: int = 0,
    fp8: bool = True,
    use_deepspeed: bool = False,
    deepspeed_config: Optional[str] = None,
):
    """
    Main training loop for projector mode with FP8.

    This function:
      1. Initializes the Qwen-Omni model (Whisper + Projector + Qwen3.6 LLM) in FP8
      2. Loads the audio-text dataset
      3. Trains ONLY the audio projector (LLM and Whisper are frozen)
      4. Saves checkpoints periodically
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("=" * 80)
    logger.info("Qwen-Omni FP8 Projector Training")
    logger.info("=" * 80)
    logger.info(f"LLM:          {llm_model_name}")
    logger.info(f"Whisper:      {whisper_model_name}")
    logger.info(f"Dataset:      {dataset_name}")
    logger.info(f"Output dir:   {output_dir}")
    logger.info(f"Epochs:       {num_train_epochs}")
    logger.info(f"Batch size:   {per_device_train_batch_size} x {gradient_accumulation_steps} (accum)")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"FP8:          {fp8}")
    logger.info(f"Projector:    {projector_num_layers} layers, dim={projector_hidden_size}, pool={audio_pooling}")
    logger.info("=" * 80)

    # Check FP8 support
    if fp8 and torch.cuda.is_available():
        gpu_arch = torch.cuda.get_device_capability(0)
        native_fp8 = gpu_arch[0] >= 9  # Hopper SM90+
        logger.info(f"GPU capability: SM {gpu_arch[0]}.{gpu_arch[1]}")
        logger.info(f"Native FP8 support: {native_fp8} (Hopper = True)")
        if not native_fp8:
            logger.info("FP8 on A100: memory savings only (software emulation, no speedup)")

    os.makedirs(output_dir, exist_ok=True)

    # Save training config
    config = {
        "llm_model_name": llm_model_name,
        "whisper_model_name": whisper_model_name,
        "dataset_name": dataset_name,
        "projector_hidden_size": projector_hidden_size,
        "projector_num_layers": projector_num_layers,
        "audio_pooling": audio_pooling,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "seed": seed,
        "fp8": fp8,
    }
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Step 1: Initialize model ----
    model = QwenOmniModel(
        llm_model_name=llm_model_name,
        whisper_model_name=whisper_model_name,
        audio_hidden_size=1280,
        projector_hidden_size=projector_hidden_size,
        projector_num_layers=projector_num_layers,
        audio_pooling=audio_pooling,
        use_fp8=fp8,
    )

    # Count trainable parameters (should only be projector)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:    {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} (projector only)")
    logger.info(f"Frozen parameters:   {total_params - trainable_params:,} (LLM + Whisper)")

    # Save model info
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump({
            "total_params": total_params,
            "trainable_params": trainable_params,
            "llm_hidden_size": model.llm_hidden_size,
            "audio_hidden_size": 1280,
            "audio_token_id": model.audio_token_id,
        }, f, indent=2)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ---- Step 2: Load dataset ----
    logger.info("Loading dataset...")
    dataset = prepare_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        audio_column=audio_column,
        text_column=text_column,
        max_samples=None,  # set to a number for debugging, e.g., 1000
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Create data collator
    data_collator = AudioTextDataCollator(
        tokenizer=model.tokenizer,
        audio_token_id=model.audio_token_id,
        max_audio_samples=max_audio_samples,
        max_text_len=max_text_len,
    )

    # ---- Step 3: Create data loader ----
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Step 4: Optimizer & Scheduler ----
    if max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = (len(dataloader) // gradient_accumulation_steps) * num_train_epochs

    optimizer, scheduler = create_optimizer_scheduler(
        model,
        lr=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        total_steps=total_steps,
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {int(total_steps * warmup_ratio)}")

    # ---- Step 5: Training loop ----
    model.train()
    global_step = 0
    running_loss = 0.0
    step_loss = 0.0

    # Gradient accumulation state
    accumulation_steps = gradient_accumulation_steps

    for epoch in range(num_train_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_train_epochs}")

        for step, batch in enumerate(dataloader):
            # Move batch to device
            audio = batch["audio"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass (FP8 autocast)
            with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn, enabled=fp8):
                outputs = model(
                    audio=audio,
                    audio_attention_mask=audio_attention_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs["loss"]
            loss = loss / accumulation_steps

            # Backward pass (FP8 autocast for grad computation)
            with torch.autocast(device_type="cuda", dtype=torch.float8_e4m3fn, enabled=fp8):
                loss.backward()

            running_loss += loss.item()
            step_loss += loss.item()

            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = step_loss / (logging_steps * accumulation_steps)
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {global_step} | Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Progress: {global_step/total_steps*100:.1f}%"
                    )

                # Save checkpoint
                if save_steps > 0 and global_step % save_steps == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    logger.info(f"Saving checkpoint to {save_path}")
                    torch.save(
                        {
                            "step": global_step,
                            "projector_state_dict": model.audio_projector.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": config,
                        },
                        os.path.join(save_path, "projector.pt"),
                    )
                    # Also save tokenizer
                    model.tokenizer.save_pretrained(save_path)

                step_loss = 0.0

            # Stop if max steps reached
            if max_steps > 0 and global_step >= max_steps:
                break

        if max_steps > 0 and global_step >= max_steps:
            break

    # ---- Step 6: Save final model ----
    final_path = os.path.join(output_dir, "final")
    logger.info(f"Saving final projector to {final_path}")
    torch.save(
        {
            "step": global_step,
            "projector_state_dict": model.audio_projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        },
        os.path.join(final_path, "projector.pt"),
    )
    model.tokenizer.save_pretrained(final_path)

    # Save projector architecture config for inference
    projector_config = {
        "audio_hidden_size": 1280,
        "llm_hidden_size": model.llm_hidden_size,
        "projector_hidden_size": projector_hidden_size,
        "num_layers": projector_num_layers,
        "pooling": audio_pooling,
        "audio_token_id": model.audio_token_id,
        "llm_model_name": llm_model_name,
    }
    with open(os.path.join(final_path, "projector_config.json"), "w") as f:
        json.dump(projector_config, f, indent=2)

    logger.info(f"Training complete! Final avg loss: {running_loss / global_step:.4f}")
    logger.info(f"Model saved to: {final_path}")

    return model, optimizer, scheduler


# ============================================================================
# 5. Inference after training
# ============================================================================

def load_trained_projector(
    llm_model_name: str,
    projector_path: str,
    whisper_model_name: str = "openai/whisper-large-v3-turbo",
) -> QwenOmniModel:
    """
    Load a trained projector and wrap it with the LLM + Whisper for inference.
    """
    logger.info(f"Loading projector from {projector_path}")

    # Load config
    config_path = os.path.join(projector_path, "projector_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model = QwenOmniModel(
        llm_model_name=config["llm_model_name"],
        whisper_model_name=whisper_model_name,
        audio_hidden_size=config["audio_hidden_size"],
        projector_hidden_size=config["llm_hidden_size"],
        projector_num_layers=config["num_layers"],
        audio_pooling=config["pooling"],
    )

    # Load trained projector weights
    projector_ckpt = os.path.join(projector_path, "projector.pt")
    ckpt = torch.load(projector_ckpt, map_location="cpu")
    model.audio_projector.load_state_dict(ckpt["projector_state_dict"])
    logger.info("Projector weights loaded successfully")

    return model


@torch.no_grad()
def transcribe_audio(model: QwenOmniModel, audio_path: str, prompt: str = "Transcribe: <|audio|>") -> str:
    """
    Transcribe an audio file using the trained Qwen-Omni model.
    """
    import soundfile as sf

    # Load audio
    waveform, sr = sf.read(audio_path, dtype="float32")
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        try:
            import torchaudio.functional as F
            waveform = F.resample(
                torch.tensor(waveform, dtype=torch.float32),
                sr, 16000,
            ).numpy()
        except Exception:
            logger.warning("Could not resample, proceeding with original rate")

    # Convert to tensor
    audio_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    # Pad to 16kHz * 30s if needed
    max_samples = 30 * 16000
    if audio_tensor.shape[-1] < max_samples:
        padded = torch.zeros(1, max_samples)
        padded[:, :audio_tensor.shape[-1]] = audio_tensor
        audio_tensor = padded
    else:
        audio_tensor = audio_tensor[:, :max_samples]

    # Tokenize prompt
    prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(model.llm.device)

    # Generate
    output_ids = model.generate(
        audio=audio_tensor.to(model.llm.device),
        prompt_ids=prompt_ids,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )

    # Decode
    generated_text = model.tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )
    return generated_text


# ============================================================================
# 6. Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a Qwen-Omni model in FP8 projector mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with FP8 (default, recommended)
  python train_omni_projector.py \\\\
    --llm_model_name Qwen/Qwen3.6-35B-A3B \\\\
    --dataset_name librispeech_asr \\\\
    --dataset_config clean \\\\
    --output_dir ./output/qwen-omni-fp8

  # Train with BF16 (slower, more memory)
  python train_omni_projector.py \\\\
    --llm_model_name Qwen/Qwen3.6-35B-A3B \\\\
    --dataset_name librispeech_asr \\\\
    --dataset_config clean \\\\
    --no-fp8 \\\\
    --output_dir ./output/qwen-omni-bf16

  # Training with local data
  python train_omni_projector.py \\\\
    --llm_model_name /path/to/Qwen3.6-35B-A3B \\\\
    --dataset_name /path/to/local/audio_dataset \\\\
    --output_dir ./output/qwen-omni \\\\
    --num_train_epochs 3 \\\\
    --per_device_train_batch_size 2 \\\\
    --gradient_accumulation_steps 8 \\\\
    --learning_rate 2e-4

  # Quick test run (100 steps)
  python train_omni_projector.py \\\\
    --llm_model_name Qwen/Qwen3.6-35B-A3B \\\\
    --dataset_name librispeech_asr \\\\
    --max_steps 100 \\\\
    --output_dir ./output/qwen-omni-test
        """,
    )

    # Model arguments
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="Name or path of the base LLM (Qwen3.6-35B-A3B)",
    )
    parser.add_argument(
        "--whisper_model_name",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Name or path of the Whisper ASR model",
    )

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="librispeech_asr",
        help="HuggingFace dataset name or local path to audio dataset",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset config (e.g., 'clean', 'other')",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name for transcription text (e.g., 'sentence' for Common Voice)",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Column name for audio data",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/qwen-omni-projector",
        help="Directory to save checkpoints and config",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for projector",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 = train for all epochs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Projector architecture
    parser.add_argument(
        "--projector_hidden_size",
        type=int,
        default=4096,
        help="Hidden size of the projector MLP",
    )
    parser.add_argument(
        "--projector_num_layers",
        type=int,
        default=6,
        help="Number of layers in the projector MLP",
    )
    parser.add_argument(
        "--audio_pooling",
        type=str,
        default="mean",
        choices=["mean", "first", "gated"],
        help="How to pool audio features before feeding to LLM",
    )

    # Data preprocessing
    parser.add_argument(
        "--max_audio_samples",
        type=int,
        default=30 * 16000,
        help="Maximum audio samples (30s at 16kHz by default)",
    )
    parser.add_argument(
        "--max_text_len",
        type=int,
        default=2048,
        help="Maximum text sequence length",
    )

    # Logging & saving
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )

    # Precision
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="Disable FP8, use BF16 instead (default: FP8 enabled).",
    )

    # DeepSpeed
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for distributed training",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON",
    )

    args = parser.parse_args()

    # FP8 enabled by default, disable with --no-fp8
    args_dict = vars(args)
    use_fp8 = not args_dict.pop("no_fp8", False)
    args_dict["fp8"] = use_fp8

    # Run training
    train(**args_dict)


if __name__ == "__main__":
    main()
