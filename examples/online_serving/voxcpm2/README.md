# VoxCPM2 Online Serving

Serve VoxCPM2 TTS via the OpenAI-compatible `/v1/audio/speech` endpoint.

## Start the Server

```bash
vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000
```

The deploy config is auto-loaded from `vllm_omni/deploy/voxcpm2.yaml`. Pass
`--deploy-config <path>` to override, or `--stage-N-<field> <value>` (e.g.
`--stage-0-max-num-seqs 8`) for per-stage runtime overrides.

## Zero-shot Synthesis

```bash
python openai_speech_client.py --text "Hello, this is VoxCPM2."
```

Or with curl:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "voxcpm2", "input": "Hello, this is VoxCPM2.", "voice": "default"}' \
  --output output.wav
```

## Voice Cloning

Clone a speaker's voice using a reference audio file:

```bash
python openai_speech_client.py \
    --text "This should sound like the reference speaker." \
    --ref-audio /path/to/reference.wav
```

The `--ref-audio` parameter accepts:
- Local file path (auto-encoded to base64)
- URL (`https://...`)
- Base64 data URI (`data:audio/wav;base64,...`)

## Voice Presets

Voice presets let you pre-compute the AudioVAE encoding of a reference audio
offline, so that API requests only need to pass a `voice` name instead of
uploading the full audio on every call.

### Create a preset (offline, requires GPU)

```bash
python create_voice_preset.py \
    --model openbmb/VoxCPM2 \
    --audio speaker.wav \
    --name alice \
    --output-dir ./voice_presets/
```

Optional flags:
- `--ref-text "transcript of the audio"` — stores the transcript alongside the preset
- `--output-dir` — output directory (default `./voice_presets/`)

This produces `voice_presets/alice.safetensors` containing the AudioVAE latent.

### Start the server with presets

The server loads all `*.safetensors` from the presets directory at startup.
The directory is resolved in this order:

1. `VOXCPM2_VOICE_PRESETS_DIR` environment variable
2. `tts_args.voice_presets_dir` in the stage config YAML
3. `<model_path>/voice_presets/` (e.g. `openbmb/VoxCPM2/voice_presets/`)

```bash
VOXCPM2_VOICE_PRESETS_DIR=./voice_presets python -m vllm_omni.entrypoints.openai.api_server \
    --model openbmb/VoxCPM2 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm2.yaml
```

Check the server logs for a line like:

```
VoxCPM2: loaded 1 voice presets from ./voice_presets: ['alice']
```

### Use a voice preset

Pass the preset name via `--voice` (client script) or the `voice` field in the JSON payload:

```bash
# Client script
python openai_speech_client.py --text "你好世界" --voice alice

# curl
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "openbmb/VoxCPM2", "input": "你好世界", "voice": "alice"}' \
  --output output.wav
```

`voice` and `ref_audio` are mutually exclusive — use one or the other.
