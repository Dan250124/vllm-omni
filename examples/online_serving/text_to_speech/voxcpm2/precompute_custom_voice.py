"""Pre-compute VoxCPM2 custom voice features for fast loading at server startup.

Encodes reference audio through the AudioVAE and saves the resulting speaker
profile as a safetensors file. The server can then serve the voice via
``voice="<name>"`` without per-request VAE encoding.

Usage:
    # Reference-only mode:
    python precompute_custom_voice.py \
        --model openbmb/VoxCPM2 \
        --voice-name alice \
        --ref-audio /path/to/alice_ref.wav \
        --output-dir ./custom_voices/

    # Reference + prompt continuation mode (higher quality):
    python precompute_custom_voice.py \
        --model openbmb/VoxCPM2 \
        --voice-name alice \
        --ref-audio /path/to/alice_ref.wav \
        --prompt-audio /path/to/alice_prompt.wav \
        --prompt-text "transcript of the prompt audio" \
        --output-dir ./custom_voices/

    # Batch mode — process all voices listed in a JSON manifest:
    python precompute_custom_voice.py \
        --model openbmb/VoxCPM2 \
        --manifest voices.json \
        --output-dir ./custom_voices/

voices.json example:
    [
        {
            "name": "alice",
            "ref_audio": "/path/to/alice_ref.wav",
            "ref_text": "transcript of alice ref audio",
            "prompt_audio": "/path/to/alice_prompt.wav",
            "prompt_text": "transcript of prompt audio"
        },
        {
            "name": "bob",
            "ref_audio": "/path/to/bob_ref.wav"
        }
    ]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm_omni.model_executor.models.voxcpm2.voxcpm2_import_utils import import_voxcpm2_core


def _read_audio_mono(path: str) -> tuple[list[float], int]:
    """Read an audio file and convert to mono float32 list."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    return data.tolist(), sr


def precompute_single_voice(
    tts_model,
    voice_name: str,
    ref_audio_path: str,
    ref_text: str | None = None,
    prompt_audio_path: str | None = None,
    prompt_text: str | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Encode reference audio through the AudioVAE and return the profile dict and tensors.

    Three modes:
      1. --ref-audio only            → mode="reference" (voice cloning)
      2. --ref-audio --ref-text      → mode="continuation" (same audio as both
         ref and prompt, like the original API's ref_audio+ref_text flow)
      3. --ref-audio --prompt-audio  → mode="ref_continuation" (separate ref
         --prompt-text                 and prompt audio)
    """
    tensors: dict[str, torch.Tensor] = {}

    ref_feat = tts_model._encode_wav(ref_audio_path, padding_mode="right")
    ref_len = ref_feat.size(0)  # number of latent frames

    # Mode 3: separate ref + prompt (best quality)
    if prompt_audio_path and prompt_text:
        tensors["ref_audio_feat"] = ref_feat
        tensors["audio_feat"] = tts_model._encode_wav(prompt_audio_path, padding_mode="left")
        mode = "ref_continuation"
        p_len = tensors["audio_feat"].size(0)
        prompt_ids = list(tts_model.text_tokenizer(prompt_text))
        extra_prefill_tokens = ref_len + 2 + len(prompt_ids) + p_len
        profile: dict[str, object] = {
            "mode": mode,
            "ref_audio_feat_shape": list(ref_feat.shape),
            "audio_feat_shape": list(tensors["audio_feat"].shape),
            "extra_prefill_tokens": extra_prefill_tokens,
            "prompt_text": prompt_text,
        }
    # Mode 2: same audio as both ref and prompt (continuation)
    elif ref_text and not prompt_audio_path:
        tensors["audio_feat"] = tts_model._encode_wav(ref_audio_path, padding_mode="left")
        mode = "continuation"
        a_len = tensors["audio_feat"].size(0)
        prompt_ids = list(tts_model.text_tokenizer(ref_text))
        extra_prefill_tokens = len(prompt_ids) + a_len
        profile = {
            "mode": mode,
            "audio_feat_shape": list(tensors["audio_feat"].shape),
            "extra_prefill_tokens": extra_prefill_tokens,
            "prompt_text": ref_text,
        }
    # Mode 1: reference only
    else:
        tensors["ref_audio_feat"] = ref_feat
        mode = "reference"
        extra_prefill_tokens = ref_len + 2
        profile = {
            "mode": mode,
            "ref_audio_feat_shape": list(ref_feat.shape),
            "extra_prefill_tokens": extra_prefill_tokens,
        }

    if ref_text and "ref_text" not in profile:
        profile["ref_text"] = ref_text

    # safetensors requires contiguous tensors
    tensors = {k: v.contiguous() for k, v in tensors.items()}

    return profile, tensors


def load_or_create_manifest(output_dir: Path) -> dict:
    """Load existing manifest or create a new one."""
    manifest_path = output_dir / "custom_voice_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"voices": {}}


def save_manifest(manifest: dict, output_dir: Path) -> None:
    manifest_path = output_dir / "custom_voice_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute VoxCPM2 custom voice speaker profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, default="openbmb/VoxCPM2", help="VoxCPM2 model path or HF repo ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for speaker profiles")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="JSON manifest file listing voices to pre-compute (batch mode)",
    )

    # Single voice args (ignored in batch mode)
    single = parser.add_argument_group("single voice (ignored when --manifest is set)")
    single.add_argument("--voice-name", type=str, default=None, help="Name for the custom voice")
    single.add_argument("--ref-audio", type=str, default=None, help="Path to reference audio WAV file")
    single.add_argument("--ref-text", type=str, default=None, help="Transcript of the reference audio")
    single.add_argument("--prompt-audio", type=str, default=None, help="Path to prompt audio WAV file")
    single.add_argument("--prompt-text", type=str, default=None, help="Transcript of the prompt audio")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest and not args.voice_name:
        parser.error("Either --manifest (batch mode) or --voice-name (single mode) is required")
    if not args.manifest and not args.ref_audio:
        parser.error("--ref-audio is required for single voice mode")

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
    else:
        voices = [
            {
                "name": args.voice_name,
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "prompt_audio": args.prompt_audio,
                "prompt_text": args.prompt_text,
            }
        ]

    # Load native VoxCPM2 model
    print(f"Loading model: {args.model}")
    VoxCPM = import_voxcpm2_core()
    native = VoxCPM.from_pretrained(args.model, load_denoiser=False, optimize=False)
    tts_model = native.tts_model
    print(f"Model loaded. patch_size={tts_model.patch_size}, feat_dim={tts_model.feat_dim}")

    manifest = load_or_create_manifest(output_dir)

    for voice_cfg in voices:
        name = voice_cfg["name"]
        ref_audio = voice_cfg["ref_audio"]
        ref_text = voice_cfg.get("ref_text")
        prompt_audio = voice_cfg.get("prompt_audio")
        prompt_text = voice_cfg.get("prompt_text")

        if not os.path.exists(ref_audio):
            print(f"ERROR: reference audio not found: {ref_audio}")
            continue
        if prompt_audio and not os.path.exists(prompt_audio):
            print(f"ERROR: prompt audio not found: {prompt_audio}")
            continue

        # Validate: prompt_text required when prompt_audio is given
        if prompt_audio and not prompt_text:
            print(f"ERROR: --prompt-text is required when --prompt-audio is provided (voice: {name})")
            continue

        print(f"\nProcessing voice: {name}")
        print(f"  ref_audio: {ref_audio}")
        if ref_text:
            print(f"  ref_text:  {ref_text[:80]}...")
        if prompt_audio:
            print(f"  prompt_audio: {prompt_audio}")
            print(f"  prompt_text:  {prompt_text[:80]}...")

        profile, tensors = precompute_single_voice(
            tts_model,
            voice_name=name,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            prompt_audio_path=prompt_audio,
            prompt_text=prompt_text,
        )

        # Save tensors
        st_path = output_dir / f"{name}.safetensors"
        save_file(tensors, str(st_path))
        print(f"  Saved: {st_path} ({st_path.stat().st_size:,} bytes)")
        if "ref_audio_feat_shape" in profile:
            print(f"  ref_audio_feat shape: {profile['ref_audio_feat_shape']}")
        if "audio_feat_shape" in profile:
            print(f"  audio_feat shape:    {profile['audio_feat_shape']}")
        print(f"  mode: {profile['mode']}")
        if profile.get("prompt_text"):
            print(f"  prompt_text: {profile['prompt_text'][:80]}...")
        print(f"  extra_prefill_tokens: {profile['extra_prefill_tokens']}")

        # Update manifest
        manifest["voices"][name] = {
            "file": f"{name}.safetensors",
            "mode": profile["mode"],
            "extra_prefill_tokens": profile["extra_prefill_tokens"],
        }
        if ref_text:
            manifest["voices"][name]["ref_text"] = ref_text
        if profile.get("prompt_text"):
            manifest["voices"][name]["prompt_text"] = profile["prompt_text"]

    save_manifest(manifest, output_dir)
    print(f"\nManifest saved: {output_dir / 'custom_voice_manifest.json'}")
    print(f"Total custom voices: {len(manifest['voices'])}")


if __name__ == "__main__":
    main()
