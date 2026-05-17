# deepfake-analyzer

Interactive FastAPI + CLI app for screening images and videos for possible
deepfake, synthetic media, or manipulation risk using the Google Gemini API.

This project is designed as a risk and anomaly analyzer, not a definitive
forensic detector. It returns structured, uncertainty-aware reports that help
you decide what to review next, not prove whether media is authentic.

## Live demo

- Interactive UI: [deepfake-analyzer-nine.vercel.app](https://deepfake-analyzer-nine.vercel.app/)
- Local UI: `http://127.0.0.1:8000/`
- Health check: `GET /health`
- JSON analysis endpoint: `POST /analyze`

## What it does

- Accepts image and video uploads
- Validates extension and size before analysis
- Uploads files to Gemini using the Files API
- Runs a local Hugging Face fake/real classifier signal when enabled
- Requests structured JSON output using a Pydantic schema
- Renders a polished interactive UI for upload, analysis, and report review
- Exposes the same analyzer through a CLI for local workflows

Supported media:

- Images: `.jpg`, `.jpeg`, `.png`, `.webp`
- Videos: `.mp4`, `.mov`, `.mkv`, `.webm`

## Report shape

Each analysis returns:

- `label`: `likely_authentic`, `uncertain`, `suspicious`, or `likely_manipulated`
- `risk_score`: integer from `0` to `100`
- `confidence`: `low`, `medium`, or `high`
- `summary`
- `detector_signal`: optional Hugging Face fake/real classifier score
- `evidence[]`
- `limitations[]`
- `recommended_next_steps[]`

The prompt instructs Gemini to inspect:

- face boundary artifacts
- lighting and shadow inconsistencies
- unnatural skin texture
- distorted eyes, teeth, ears, hair, glasses, and jewelry
- lip-sync mismatch
- blinking and gaze anomalies
- temporal flicker
- compression artifacts
- metadata inconsistencies
- low-quality media false positives

## Detector model choice

The default local detector is
[`xRayon/convnext-ai-images-detector`](https://huggingface.co/xRayon/convnext-ai-images-detector).
It was selected over older ViT/SigLIP detectors because its model card describes
training on about 400k real-vs-AI images plus continual learning for newer
generators, including DALL-E 3, Flux, SDXL, SD3.5, and Midjourney V6. Its card
reports a 90.40% fake detection rate on an out-of-distribution EvalGen set.

The smaller standard Transformers fallback is
[`prithivMLmods/deepfake-detector-model-v1`](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1),
which is easier to deploy but less targeted to the app's clean AI-generated
image failure case.

## Tech stack

- Python 3.10+
- FastAPI
- Google Gemini via `google-genai`
- Hugging Face Transformers
- timm / TorchVision for ConvNeXt
- Pydantic
- OpenCV
- Pillow
- NumPy
- Pytest

## Quickstart

```bash
cd deepfake-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey),
then configure it locally:

```bash
cp .env.example .env
```

Set:

```bash
GEMINI_API_KEY=your_real_key_here
```

Optional fallback models can be configured for temporary Gemini overload or
rate-limit failures:

```bash
GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite
```

The app runs `xRayon/convnext-ai-images-detector` locally as the default
classifier signal. It was chosen because its model card describes newer
training coverage for modern generators such as DALL-E 3, Flux, SDXL, SD3.5,
and Midjourney V6. The first request downloads its checkpoint, which is roughly
1.05 GB in the Hugging Face cache.

```bash
HF_DETECTOR_ENABLED=true
HF_DETECTOR_MODEL=xRayon/convnext-ai-images-detector
HF_DETECTOR_VIDEO_FRAMES=8
HF_DETECTOR_FAKE_THRESHOLD=0.55
```

For a smaller standard Transformers fallback, set:

```bash
HF_DETECTOR_MODEL=prithivMLmods/deepfake-detector-model-v1
```

Never hard-code or commit your API key.

## Run locally

Start the server:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Analyze a file through the API:

```bash
curl -X POST \
  -F "file=@path/to/file.mp4" \
  http://127.0.0.1:8000/analyze
```

Run the CLI:

```bash
python cli.py path/to/file.mp4
```

## API notes

`POST /analyze` accepts a multipart file upload and returns a `DeepfakeReport`
object. The app hides internal stack traces from API responses and returns
user-safe error messages for validation, configuration, and Gemini failures.

## Development

Run tests:

```bash
pytest
```

Current project layout:

```text
deepfake-analyzer/
  app/
    main.py
    analyzer.py
    media_utils.py
    prompts.py
    schemas.py
    static/
  tests/
  cli.py
```

## Privacy warning

Uploaded files are sent to the Google Gemini API for analysis. Do not upload
private, sensitive, confidential, or regulated media unless you are authorized
to do so and your usage complies with Google's terms and your own privacy and
security requirements.

## Important limitation

This tool does not produce forensic proof. Results can be affected by
compression, editing, motion blur, lighting, low resolution, short clips, and
model limitations. For high-stakes decisions, pair this output with source
verification, provenance checks, metadata review, and expert forensic analysis.
