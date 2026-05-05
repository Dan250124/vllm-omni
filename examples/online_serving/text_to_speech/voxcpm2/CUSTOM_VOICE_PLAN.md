# VoxCPM2 Custom Voice — Technical Plan

## Goal

Allow users to pre-clone voices and configure them in a directory. When
`vllm serve openbmb/VoxCPM2 --omni` starts, it loads the pre-computed custom
voice features and serves them via `voice="<name>"` without per-request
AudioVAE encoding.

## Terminology

| Term | Meaning | Scope |
|------|---------|-------|
| **Custom Voice** | 用户预先克隆的音色 | User-facing API / feature name |
| **Speaker Profile** | 预计算的 VAE 特征 + 元数据 | Internal implementation |

## Current Architecture (baseline)

```
POST /v1/audio/speech {input, ref_audio, ref_text}
  │
  ▼
serving_speech._build_voxcpm2_prompt()
  ├─ _resolve_ref_audio(ref_audio) → (wav_samples, sr)
  └─ build_voxcpm2_prompt(hf_config, tokenizer, split_map, text, ref_audio, ref_sr, ref_text)
       └─ pads prompt_token_ids to full prefill length

  ▼  (vLLM scheduler dispatches to worker)

voxcpm2_talker.preprocess()
  ├─ _build_prompt_cache(ref_audio, prompt_audio, prompt_text)   ← HOTSPOT
  │    ├─ AudioResampler (if sr mismatch)
  │    ├─ AudioVAE.encode(audio) → ref_audio_feat  (expensive)
  │    └─ returns {"mode": "reference", "ref_audio_feat": tensor}
  └─ _build_prefill_inputs(token_ids, dev, req_id)
       └─ consumes prompt_cache → builds text/audio mask + feature sequence
```

**Key insight**: `_build_prompt_cache()` output is deterministic given the same
reference audio — it depends only on the audio waveform, not on the request
text. The text-dependent part lives in `_build_prefill_inputs()`.

## What Gets Pre-computed

The `prompt_cache` dict for each custom voice (speaker profile):

```python
# For reference-only cloning (mode="reference"):
{
    "mode": "reference",
    "ref_audio_feat": Tensor(n_frames, patch_size, latent_dim),
}

# For reference + prompt continuation (mode="ref_continuation"):
{
    "mode": "ref_continuation",
    "ref_audio_feat": Tensor(n_frames, patch_size, latent_dim),
    "audio_feat": Tensor(n_prompt_frames, patch_size, latent_dim),
    "prompt_text": "transcript of the prompt audio",
}
```

`latent_dim` comes from `tts.audio_vae.latent_dim` and `patch_size` from
`tts.patch_size` — both fixed per model instance.

## Implementation Plan

### Phase 1 — Pre-computation CLI tool

New file: `examples/online_serving/text_to_speech/voxcpm2/precompute_custom_voice.py`

```
Usage:
  python precompute_custom_voice.py \
      --model openbmb/VoxCPM2 \
      --voice-name alice \
      --ref-audio /path/to/alice_ref.wav \
      --ref-text "transcript of alice ref audio" \
      --output-dir ./custom_voices/

  # With prompt continuation (higher quality cloning):
  python precompute_custom_voice.py \
      --model openbmb/VoxCPM2 \
      --voice-name alice \
      --ref-audio /path/to/alice_ref.wav \
      --prompt-audio /path/to/alice_prompt.wav \
      --prompt-text "transcript of prompt audio" \
      --output-dir ./custom_voices/
```

What it does:
1. Loads VoxCPM2 native model via `VoxCPM.from_pretrained()`
2. Reads reference audio, runs through `_encode_wav()` / `_encode_raw_audio()`
   to produce `ref_audio_feat` tensor
3. (Optionally) encodes prompt audio → `audio_feat` tensor
4. Saves as `{name}.safetensors` + a `custom_voice_manifest.json` index file

Output directory structure:

```
./custom_voices/
├── custom_voice_manifest.json    # index of all custom voices
├── alice.safetensors             # {"ref_audio_feat": tensor, "mode": "reference"}
├── bob.safetensors
└── carol.safetensors             # {"ref_audio_feat": ..., "audio_feat": ..., "mode": "ref_continuation", "prompt_text": "..."}
```

`custom_voice_manifest.json` schema:

```json
{
  "voices": {
    "alice": {
      "file": "alice.safetensors",
      "mode": "reference",
      "ref_text": "transcript of alice ref audio"
    },
    "bob": {
      "file": "bob.safetensors",
      "mode": "ref_continuation",
      "ref_text": "transcript of bob ref audio",
      "prompt_text": "transcript of prompt audio"
    }
  }
}
```

### Phase 2 — Model-side cache loading

Modify `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`:

#### 2a. Add `custom_voice_dir` support to `__init__`

```python
# In VoxCPM2TalkerForConditionalGeneration.__init__:

# path resolved from deploy config or env var:
# VLLM_OMNI_VOXCPM2_CUSTOM_VOICE_DIR
self._custom_voice_dir: str | None = None
self._speaker_profiles: dict[str, dict] = {}

def _load_speaker_profiles(self) -> None:
    """Load pre-computed custom voice features from custom_voice_dir."""
    if not self._custom_voice_dir:
        return
    manifest_path = os.path.join(self._custom_voice_dir, "custom_voice_manifest.json")
    if not os.path.exists(manifest_path):
        logger.warning("custom_voice_dir set but no custom_voice_manifest.json found")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    for name, info in manifest.get("voices", {}).items():
        file_path = os.path.join(self._custom_voice_dir, info["file"])
        tensors = load_file(file_path)  # safetensors
        entry = {
            "mode": info["mode"],
            "ref_audio_feat": tensors["ref_audio_feat"],
        }
        if "audio_feat" in tensors:
            entry["audio_feat"] = tensors["audio_feat"]
            entry["prompt_text"] = info.get("prompt_text", "")
        self._speaker_profiles[name] = entry
    logger.info("Loaded %d custom voice speaker profiles from %s",
                len(self._speaker_profiles), self._custom_voice_dir)
```

#### 2b. Modify `_build_prompt_cache` to check speaker profile cache first

```python
def _build_prompt_cache(self, ref_audio=None, prompt_audio=None,
                        prompt_text=None, voice_name=None) -> dict | None:
    # Fast path: custom voice — use pre-computed speaker profile
    if (voice_name and ref_audio is None and prompt_audio is None
            and voice_name in self._speaker_profiles):
        return dict(self._speaker_profiles[voice_name])  # shallow copy
    # Slow path: full VAE encoding (existing logic)
    ...
```

#### 2c. Modify `preprocess` to pass `voice_name`

Add a `voice_name` key to the `additional_information` dict passed from the
serving layer, and thread it through to `_build_prompt_cache`.

### Phase 3 — Serving layer changes

Modify `vllm_omni/entrypoints/openai/serving_speech.py`:

#### 3a. Pass `voice_name` in `_build_voxcpm2_prompt`

```python
async def _build_voxcpm2_prompt(self, request):
    ...
    return build_voxcpm2_prompt(
        ...,
        ref_audio=ref_audio,
        ref_sr=ref_sr,
        ref_text=request.ref_text,
        voice_name=request.voice,  # NEW: resolve custom voice by name
    )
```

Update `build_voxcpm2_prompt()` in `voxcpm2_talker.py` to thread `voice_name`
into the additional_information dict so `preprocess()` receives it.

#### 3b. Update validation to allow `voice` field

Currently `_validate_voxcpm_request` (line 985) does not reject `voice`, but
the README says it's unsupported. The validation logic stays the same — if
`voice` is provided and matches a pre-loaded custom voice, it's used; otherwise
`ref_audio` path is used as before.

#### 3c. Voices endpoint integration (optional, can be Phase 4)

Make VoxCPM2 compatible with `GET/POST/DELETE /v1/audio/voices`:
- `GET`: list pre-loaded custom voices + uploaded voices
- `POST`: upload audio → run VAE encoding on-the-fly → cache as speaker profile in memory
- `DELETE`: remove from memory

### Phase 4 — Runtime upload with on-the-fly encoding

Once the speaker profile cache infrastructure is in place,
`POST /v1/audio/voices` can encode uploaded audio through the same VAE path and
add it to `_speaker_profiles` at runtime. This reuses the existing
`uploaded_speakers` mechanism from `serving_speech.py` and adds VAE encoding
as an extra step.

### Phase 5 — Deploy config integration

Add to `vllm_omni/deploy/voxcpm2.yaml`:

```yaml
# Optional: directory containing pre-computed custom voice .safetensors files
# and a custom_voice_manifest.json index.
custom_voice_dir: ""
```

And/or support the env var `VLLM_OMNI_VOXCPM2_CUSTOM_VOICE_DIR` for runtime override.

## Request Flow (After Changes)

```
POST /v1/audio/speech {input: "Hello", voice: "alice"}
  │
  ▼
serving_speech._build_voxcpm2_prompt()
  ├─ voice="alice" → builds prompt with voice_name="alice" in additional_info
  └─ no ref_audio → skip _resolve_ref_audio (fast path)

  ▼
voxcpm2_talker.preprocess()
  ├─ _build_prompt_cache(voice_name="alice")
  │    └─ speaker profile cache hit → returns pre-computed dict (no VAE encoding!)
  └─ _build_prefill_inputs(token_ids, dev, req_id)
       └─ uses cached ref_audio_feat → constructs prefill sequence
```

## Files Changed

| File | Change | Complexity |
|------|--------|------------|
| `examples/.../voxcpm2/precompute_custom_voice.py` | **New** — CLI tool for offline pre-computation | Medium |
| `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py` | Speaker profile cache, `_load_speaker_profiles()`, modify `_build_prompt_cache` and `preprocess` | Medium |
| `vllm_omni/entrypoints/openai/serving_speech.py` | Thread `voice_name` through to prompt builder | Low |
| `vllm_omni/deploy/voxcpm2.yaml` | Add `custom_voice_dir` config key | Trivial |

## Risks & Mitigations

1. **Tensor device/dtype mismatch**: Pre-computed tensors are saved in float32
   on CPU. At load time, they must be moved to the model's device/dtype
   (`_side_dtype`). Same pattern already used in `_encode_raw_audio` — the
   features are moved to device inside `_build_prefill_inputs`.

2. **VAE weights change across model versions**: The pre-computed VAE features
   are tied to a specific model checkpoint. If the model is updated, custom
   voices must be re-computed. Add a `model_revision` field to
   `custom_voice_manifest.json`.

3. **Memory overhead**: Each custom voice's `ref_audio_feat` is ~N_frames ×
   patch_size × latent_dim floats. For a typical 3-second reference at 48kHz
   with patch_size=4 and latent_dim=64, that's roughly ~150 × 4 × 64 = ~38K
   floats ≈ 150KB per voice. Negligible.

4. **Concurrent access**: The speaker profile cache is read-only after startup
   (except for runtime upload). Python dict reads are thread-safe for read-only
   access. For runtime upload, add a lock around cache mutation.
