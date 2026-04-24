"""Create a VoxCPM2 voice preset from a reference audio file.

Encodes reference audio through AudioVAE and saves the latent tensor as a
safetensors file.  The preset can later be loaded by the server (via
``VOXCPM2_VOICE_PRESETS_DIR``) so that clients only need to pass ``voice``
instead of ``ref_audio`` on every request.

Usage::

    python create_voice_preset.py \
        --model VoxCPM2 \
        --audio speaker.wav \
        --name alice \
        --output-dir ./voice_presets/
"""

from __future__ import annotations

import argparse
import os

import librosa
import safetensors.torch as st
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a VoxCPM2 voice preset")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--audio", type=str, required=True, help="Path to reference WAV file")
    parser.add_argument("--name", type=str, required=True, help="Preset name (filename stem)")
    parser.add_argument("--ref-text", type=str, default=None, help="Optional transcript of reference audio")
    parser.add_argument("--output-dir", type=str, default="./voice_presets/", help="Directory for preset files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_import_utils import import_voxcpm2_core
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import _encode_raw_audio

    VoxCPM = import_voxcpm2_core()
    print(f"Loading model from {args.model} ...")
    native = VoxCPM.from_pretrained(args.model, load_denoiser=False, optimize=False)
    tts = native.tts_model.to("cuda")

    # Load audio
    samples, sr = librosa.load(args.audio, sr=None, mono=True)
    samples = samples.astype("float32").tolist()
    duration = len(samples) / sr
    print(f"Audio: {args.audio} ({duration:.1f}s, {sr}Hz)")

    # Encode
    print("Encoding through AudioVAE ...")
    feat = _encode_raw_audio(tts, samples, sr)
    print(f"Encoded latent shape: {tuple(feat.shape)}")

    # Save
    data = {
        "ref_audio_feat": feat.float().contiguous(),
        "sample_rate": torch.tensor(sr, dtype=torch.int32),
        "audio_duration_sec": torch.tensor(duration, dtype=torch.float32),
    }
    if args.ref_text is not None:
        # Store ref_text as a scalar tensor so safetensors can hold it
        data["ref_text"] = torch.tensor([ord(c) for c in args.ref_text], dtype=torch.int32)

    out_path = os.path.join(args.output_dir, f"{args.name}.safetensors")
    st.save_file(data, out_path)
    print(f"Saved preset: {out_path}")


if __name__ == "__main__":
    main()
