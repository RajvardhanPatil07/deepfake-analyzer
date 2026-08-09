## Contribution by Bhavesh Patil
# deepfake-analyzer

Interactive FastAPI, web UI, and CLI app for screening images and videos for
possible deepfake, synthetic media, or manipulation risk.

The analyzer combines a Gemini vision review with an optional local Hugging Face
fake/real classifier signal. It is designed as a risk and anomaly analyzer, not
a definitive forensic detector. The output helps decide what to review next; it
does not prove whether media is authentic.

## Live demo

- Interactive UI: [deepfake-analyzer-nine.vercel.app](https://deepfake-analyzer-nine.vercel.app/)
- Local UI: `http://127.0.0.1:8000/`
- Health check: `GET /health`
- JSON analysis endpoint: `POST /analyze`

## What it does

- Accepts image and video uploads
- Validates extension and size before analysis
- Uploads supported media to Gemini through the Files API
- Runs an optional local Hugging Face classifier before Gemini analysis
- Samples video frames for the local classifier signal
- Injects the classifier signal into the Gemini prompt as a non-forensic clue
- Requests structured JSON output using a Pydantic schema
- Calibrates the final report when the classifier strongly disagrees with a
  low-risk Gemini result
- Renders a web UI for upload, analysis, detector signal review, and report
  inspection
- Exposes the same analyzer through a CLI for local workflows
- Returns user-safe API errors for validation, configuration, Gemini response,
  and temporary overload failures

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

Example response:

```json
{
  "label": "suspicious",
  "risk_score": 64,
  "confidence": "medium",
  "summary": "The media contains several visual signals that warrant review.",
  "detector_signal": {
    "model": "xRayon/convnext-ai-images-detector",
    "media_type": "image",
    "label": "fake",
    "fake_probability": 0.72,
    "real_probability": 0.28,
    "confidence": "medium",
    "frames_analyzed": 1
  },
  "evidence": [],
  "limitations": [],
  "recommended_next_steps": []
}
```

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
It was selected over older ViT and SigLIP detectors because its model card
describes training on about 400k real-vs-AI images plus continual learning for
newer generators, including DALL-E 3, Flux, SDXL, SD3.5, and Midjourney V6. Its
card reports a 90.40% fake detection rate on an out-of-distribution EvalGen set.

The smaller standard Transformers fallback is
[`prithivMLmods/deepfake-detector-model-v1`](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1),
which is easier to deploy but less targeted to the app's clean AI-generated
image failure case.

The detector is optional. If dependencies are missing, model loading fails, or
`HF_DETECTOR_ENABLED=false`, the app still runs the Gemini analysis without the
local classifier signal.

## Tech stack

- Python 3.10+
- FastAPI
- Google Gemini via `google-genai`
- Hugging Face Transformers, timm, and TorchVision for the optional local detector
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

The Hugging Face detector is optional because the default xRayon checkpoint and
PyTorch stack are too large for small serverless bundles. To enable it locally,
install the detector extras and turn it on in `.env`:

```bash
pip install -r requirements-detector.txt
```

The default detector is `xRayon/convnext-ai-images-detector`. It was chosen
because its model card describes newer training coverage for modern generators
such as DALL-E 3, Flux, SDXL, SD3.5, and Midjourney V6. The first detector
request downloads its checkpoint, which is roughly 1.05 GB in the Hugging Face
cache.

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

### Environment variables

| Variable | Required | Default | Notes |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | None | Google Gemini API key used by `google-genai`. |
| `GEMINI_FALLBACK_MODELS` | No | `gemini-2.5-flash,gemini-2.5-flash-lite` | Comma-separated fallback models tried after transient Gemini errors. |
| `HF_DETECTOR_ENABLED` | No | `false` | Set to `true` after installing `requirements-detector.txt` to run the local classifier. |
| `HF_DETECTOR_MODEL` | No | `xRayon/convnext-ai-images-detector` | Hugging Face detector model id. |
| `HF_DETECTOR_VIDEO_FRAMES` | No | `8` | Number of evenly spaced video frames to classify. Clamped from 1 to 24. |
| `HF_DETECTOR_FAKE_THRESHOLD` | No | `0.55` | Fake probability threshold used to label the detector signal. |

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

Choose a Gemini model or upload limit:

```bash
python cli.py path/to/file.jpg --model gemini-2.5-flash --max-mb 50
```

## API notes

`POST /analyze` accepts a multipart file upload and returns a `DeepfakeReport`
object. The app hides internal stack traces from API responses and returns
user-safe error messages for validation, configuration, and Gemini failures.

Common responses:

- `200`: analysis completed and returned a structured report
- `400`: unsupported extension, empty upload, or file size violation
- `500`: missing analyzer configuration such as `GEMINI_API_KEY`
- `502`: Gemini rejected the request or returned an invalid structured response
- `503`: transient Gemini overload or rate-limit failure after all configured
  fallback models were tried

The maximum upload size is `100 MB`. Supported images are `.jpg`, `.jpeg`,
`.png`, and `.webp`. Supported videos are `.mp4`, `.mov`, `.mkv`, and `.webm`.

## How analysis works

1. The API or CLI validates the media path, extension, and size.
2. If enabled, the local Hugging Face detector classifies the image or sampled
   video frames.
3. The media file is uploaded to Gemini with a strict JSON response schema.
4. Gemini inspects visual, temporal, compression, metadata, and low-quality
   false-positive signals.
5. The app validates the Gemini JSON with Pydantic.
6. If the local detector reports a strong fake probability, the app adds the
   detector signal, appends classifier evidence, and raises the risk label or
   score when needed.

Gemini is asked not to return `detector_signal` directly. The app adds that
field after validation so the model cannot invent classifier metadata.

## Local detector notes

- The base `requirements.txt` keeps production deployments lean and runs the
  Gemini-only analyzer by default.
- Install `requirements-detector.txt` to add `torch`, `torchvision`, `timm`,
  `safetensors`, `transformers`, and `huggingface-hub` for local detector use.
- The default xRayon checkpoint is large and downloads on first use into the
  Hugging Face cache.
- On memory-limited hosts, leave `HF_DETECTOR_ENABLED=false` or switch to the
  smaller Transformers fallback model.
- Video classification uses evenly spaced frames and aggregates the strongest
  fake probabilities, so it is a screening signal for review priority, not a
  frame-by-frame forensic result.

## Deployment notes

The repository includes `.vercelignore` and a lean base `requirements.txt` for
Vercel. Configure `GEMINI_API_KEY` in the deployment environment. Keep the local
detector disabled for Vercel because the PyTorch detector stack exceeds the
serverless bundle storage limit:

```bash
HF_DETECTOR_ENABLED=false
```

The app remains useful with Gemini-only analysis when the detector is disabled.

## Development

Run tests:

```bash
python -m pytest -q
```

Current project layout:

```text
deepfake-analyzer/
  app/
    analyzer.py
    detector.py
    main.py
    media_utils.py
    prompts.py
    schemas.py
    static/
      app.js
      index.html
      styles.css
  tests/
    test_analyzer.py
    test_schemas.py
  cli.py
  requirements-detector.txt
  requirements.txt
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
