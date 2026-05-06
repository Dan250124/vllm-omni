"""Pre-compute Qwen3-TTS custom voice speaker embeddings for fast loading.

Encodes reference audio through the ECAPA-TDNN speaker encoder and saves the
resulting speaker embedding as a safetensors file. The server can then serve
the voice via ``voice="<name>"`` without per-request audio processing.

Usage:
    # Single voice (xvec mode, recommended default):
    python precompute_custom_voice.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --voice-name alice \
        --ref-audio /path/to/alice_ref.wav \
        --output-dir ./custom_voices/

    # Single voice (ICL mode, better quality, needs ref_text):
    python precompute_custom_voice.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --voice-name alice \
        --ref-audio /path/to/alice_ref.wav \
        --ref-text "Transcript of the reference audio." \
        --mode icl \
        --output-dir ./custom_voices/

    # Batch mode:
    python precompute_custom_voice.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --manifest voices.json \
        --output-dir ./custom_voices/

voices.json example:
    [
        {
            "name": "alice",
            "ref_audio": "/path/to/alice_ref.wav",
            "ref_text": "transcript of alice ref audio",
            "speaker_description": "warm female narrator"
        },
        {
            "name": "bob",
            "ref_audio": "/path/to/bob_ref.wav",
            "mode": "xvec"
        }
    ]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Resolve absolute paths for internal vllm_omni imports without requiring
# an editable install. Just append the repository root.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[4]
_vllm_root = _repo_root / "vllm"
if str(_vllm_root) not in sys.path:
    sys.path.insert(0, str(_vllm_root))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _read_audio_mono(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file and convert to mono float32 numpy array."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    return data, sr


def _resample_np(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D numpy waveform using vLLM's AudioResampler."""
    from vllm.multimodal.audio import AudioResampler

    resampler = AudioResampler(target_sr=target_sr)
    return resampler.resample(wav.astype(np.float32), orig_sr=int(orig_sr))


def _compute_mel_spectrogram(wav_1d: torch.Tensor) -> torch.Tensor:
    """Compute 128-bin mel spectrogram from a 24kHz waveform (mirrors talker code)."""
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import mel_spectrogram

    return mel_spectrogram(
        wav_1d.unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)


def _load_speaker_encoder_from_checkpoint(model_path: str, device: str | torch.device = "cpu"):
    """Load the ECAPA-TDNN speaker encoder directly from a HuggingFace checkpoint.

    Returns (encoder_module, hidden_size) on the specified device in bfloat16.
    """
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSSpeakerEncoder,
    )

    # Resolve model path: support both local dirs and HuggingFace repo ids
    if os.path.isdir(model_path):
        config_path = model_path
    else:
        from transformers.utils.hub import cached_file

        cfg_f = cached_file(model_path, "config.json")
        if cfg_f is None:
            raise FileNotFoundError(f"Cannot find config.json for {model_path}")
        config_path = os.path.dirname(cfg_f)

    config = Qwen3TTSConfig.from_pretrained(config_path)
    spk_config = config.speaker_encoder_config
    encoder = Qwen3TTSSpeakerEncoder(spk_config)

    # Discover safetensors files and weight map
    index_path = os.path.join(config_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map: dict[str, str] = index.get("weight_map", {})
        needed_files = sorted(
            {fn for key, fn in weight_map.items() if key.startswith("speaker_encoder.")}
        )
        state_dict: dict[str, torch.Tensor] = {}
        for st_name in needed_files:
            st_path = os.path.join(config_path, st_name)
            for key, tensor in load_safetensors(st_path).items():
                if key.startswith("speaker_encoder."):
                    target_key = key[len("speaker_encoder."):]
                    state_dict[target_key] = tensor
    else:
        # Single-file checkpoint (0.6B, etc.)
        candidates = sorted(Path(config_path).glob("model*.safetensors"))
        state_dict = {}
        for st_path in candidates:
            for key, tensor in load_safetensors(str(st_path)).items():
                if key.startswith("speaker_encoder."):
                    target_key = key[len("speaker_encoder."):]
                    state_dict[target_key] = tensor

    if not state_dict:
        raise RuntimeError(f"No speaker_encoder.* weights found in {config_path}")

    encoder.load_state_dict(state_dict)
    encoder.to(device=device, dtype=torch.bfloat16)
    encoder.eval()
    return encoder, spk_config.enc_dim


def _load_speech_tokenizer_from_checkpoint(model_path: str):
    """Load the Qwen3TTS SpeechTokenizer for ref_code encoding (ICL mode only)."""
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import (
        Qwen3TTSTokenizer,
    )

    if os.path.isdir(model_path):
        tok_dir = os.path.join(model_path, "speech_tokenizer")
    else:
        from transformers.utils.hub import cached_file

        cfg_f = cached_file(model_path, "speech_tokenizer/config.json")
        if cfg_f is None:
            raise FileNotFoundError(f"Cannot find speech_tokenizer/config.json for {model_path}")
        tok_dir = os.path.dirname(cfg_f)

    return Qwen3TTSTokenizer.from_pretrained(tok_dir, torch_dtype=torch.bfloat16)


def precompute_single_voice(
    *,
    encoder: torch.nn.Module,
    voice_name: str,
    ref_audio_path: str,
    ref_text: str | None = None,
    mode: str = "xvec",
    device: str | torch.device = "cpu",
    tokenizer=None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Extract speaker embedding (and optionally ref_code) from reference audio.

    Returns (profile_dict, tensors_dict).
    """
    if mode not in ("xvec", "icl"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'xvec' or 'icl'.")

    # Read and resample audio
    wav_np, sr = _read_audio_mono(ref_audio_path)
    if wav_np.size < 1024:
        raise ValueError(f"Reference audio too short: {wav_np.size} samples")

    target_sr = 24000
    if sr != target_sr:
        wav_np = _resample_np(wav_np, orig_sr=int(sr), target_sr=target_sr)

    # Compute mel spectrogram and extract speaker embedding
    wav_t = torch.from_numpy(wav_np).to(device=device, dtype=torch.float32)
    mels = _compute_mel_spectrogram(wav_t)
    with torch.no_grad():
        spk_emb = encoder(mels.to(device=device, dtype=torch.bfloat16))[0]  # (hidden_size,)
    spk_emb = spk_emb.to(device="cpu", dtype=torch.float32)

    tensors: dict[str, torch.Tensor] = {
        "speaker_embedding": spk_emb.contiguous(),
    }

    profile: dict[str, object] = {
        "mode": mode,
        "embedding_dim": int(spk_emb.shape[0]),
    }

    # ICL mode: also compute ref_code via speech tokenizer
    if mode == "icl":
        if tokenizer is None:
            raise ValueError("Speech tokenizer is required for ICL mode")
        # Encode through tokenizer's encoder
        audio_t = torch.from_numpy(wav_np).unsqueeze(0).to(device=device)
        codes = tokenizer.encode(audio_t, target_sr)
        if isinstance(codes, torch.Tensor):
            ref_code = codes[0].contiguous().cpu().to(torch.int32)
        elif isinstance(codes, list):
            ref_code = torch.tensor(codes[:1] if len(codes) == 1 else codes, dtype=torch.int32)
        else:
            raise TypeError(f"Unexpected tokenizer output type: {type(codes)}")
        tensors["ref_code"] = ref_code
        profile["ref_code_length"] = int(ref_code.shape[0])

    if ref_text:
        profile["ref_text"] = ref_text

    return profile, tensors


def load_or_create_manifest(output_dir: Path, hidden_size: int) -> dict:
    """Load existing manifest or create a new one."""
    manifest_path = output_dir / "custom_voice_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing_dim = manifest.get("hidden_size")
        if existing_dim and existing_dim != hidden_size:
            print(
                f"WARNING: manifest hidden_size={existing_dim} differs from "
                f"model hidden_size={hidden_size}. "
                f"Existing voices may be incompatible."
            )
        return manifest
    return {"model": "Qwen3-TTS", "hidden_size": hidden_size, "voices": {}}


def save_manifest(manifest: dict, output_dir: Path) -> None:
    manifest_path = output_dir / "custom_voice_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute Qwen3-TTS custom voice speaker embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model path or HF repo ID",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for speaker profiles",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="JSON manifest file listing voices to pre-compute (batch mode)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for computation (cuda, cpu)",
    )

    single = parser.add_argument_group("single voice (ignored when --manifest is set)")
    single.add_argument("--voice-name", type=str, default=None)
    single.add_argument("--ref-audio", type=str, default=None)
    single.add_argument("--ref-text", type=str, default=None)
    single.add_argument(
        "--mode", type=str, default="xvec", choices=("xvec", "icl"),
        help="xvec: speaker embedding only; icl: embedding + codec tokens (needs ref_text)",
    )
    single.add_argument("--speaker-description", type=str, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest and not args.voice_name:
        parser.error("Either --manifest (batch mode) or --voice-name (single mode) is required")
    if not args.manifest and not args.ref_audio:
        parser.error("--ref-audio is required for single voice mode")
    if args.mode == "icl" and not args.manifest and not args.ref_text:
        parser.error("--ref-text is required for ICL mode")

    # Collect voices to process
    voices: list[dict] = []
    if args.manifest:
        with open(args.manifest) as f:
            voices = json.load(f)
        if not isinstance(voices, list):
            parser.error("--manifest must contain a JSON array of voice entries")
        for v in voices:
            if "name" not in v or "ref_audio" not in v:
                parser.error("Each voice entry must have 'name' and 'ref_audio' fields")
            v.setdefault("mode", "xvec")
    else:
        voices = [
            {
                "name": args.voice_name,
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "mode": args.mode,
                "speaker_description": args.speaker_description,
            }
        ]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Loading speaker encoder from: {args.model}")
    encoder, hidden_size = _load_speaker_encoder_from_checkpoint(args.model, device=device)
    print(f"Speaker encoder loaded. hidden_size={hidden_size}, device={device}")

    # Load tokenizer lazily (only when ICL mode is requested)
    _tokenizer = None

    def _get_tokenizer():
        nonlocal _tokenizer
        if _tokenizer is None:
            print(f"Loading speech tokenizer from: {args.model}")
            _tokenizer = _load_speech_tokenizer_from_checkpoint(args.model)
            _tokenizer.model.encoder.to(device=device, dtype=torch.bfloat16)
            print("Speech tokenizer loaded.")
        return _tokenizer

    manifest = load_or_create_manifest(output_dir, hidden_size)

    for voice_cfg in voices:
        name = voice_cfg["name"]
        ref_audio = voice_cfg["ref_audio"]
        ref_text = voice_cfg.get("ref_text")
        mode = voice_cfg.get("mode", "xvec")
        description = voice_cfg.get("speaker_description")

        if not os.path.exists(ref_audio):
            print(f"ERROR: reference audio not found: {ref_audio}")
            continue
        if mode == "icl" and not ref_text:
            print(f"ERROR: --ref-text is required for ICL mode (voice: {name})")
            continue

        tok = _get_tokenizer() if mode == "icl" else None

        print(f"\nProcessing voice: {name}")
        print(f"  ref_audio: {ref_audio}")
        print(f"  mode: {mode}")
        if ref_text:
            print(f"  ref_text:  {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")

        profile, tensors = precompute_single_voice(
            encoder=encoder,
            voice_name=name,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            mode=mode,
            device=device,
            tokenizer=tok,
        )

        # Save tensors
        st_path = output_dir / f"{name}.safetensors"
        save_file(tensors, str(st_path))
        print(f"  Saved: {st_path} ({st_path.stat().st_size:,} bytes)")
        print(f"  speaker_embedding shape: {tensors['speaker_embedding'].shape}")
        if "ref_code" in tensors:
            print(f"  ref_code shape: {tensors['ref_code'].shape}")
        print(f"  mode: {profile['mode']}")
        if profile.get("ref_text"):
            t = str(profile["ref_text"])
            print(f"  ref_text: {t[:80]}{'...' if len(t) > 80 else ''}")

        # Update manifest
        manifest["voices"][name] = {
            "file": f"{name}.safetensors",
            "mode": mode,
        }
        if ref_text:
            manifest["voices"][name]["ref_text"] = ref_text
        if description:
            manifest["voices"][name]["speaker_description"] = description

    save_manifest(manifest, output_dir)
    print(f"\nManifest saved: {output_dir / 'custom_voice_manifest.json'}")
    print(f"Total custom voices: {len(manifest['voices'])}")


if __name__ == "__main__":
    main()
