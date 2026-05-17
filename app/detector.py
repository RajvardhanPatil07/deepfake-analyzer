"""Optional Hugging Face classifier signal for synthetic media risk."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from dotenv import load_dotenv

from app.media_utils import MediaInfo, extract_video_frames
from app.schemas import DetectorSignal

DEFAULT_HF_DETECTOR_MODEL = "xRayon/convnext-ai-images-detector"
TRANSFORMERS_DETECTOR_MODEL = "prithivMLmods/deepfake-detector-model-v1"
XRAYON_CHECKPOINT = "AI Images Detector/checkpoints/checkpoint_phase2.pth"
DEFAULT_FAKE_THRESHOLD = 0.55
DEFAULT_VIDEO_FRAMES = 8
MAX_VIDEO_FRAMES = 24


class DetectorUnavailableError(RuntimeError):
    """Raised when optional detector dependencies or model loading are unavailable."""


@dataclass(frozen=True)
class ImageScore:
    fake_probability: float
    real_probability: float


def _env_flag(name: str, *, default: bool) -> bool:
    load_dotenv()
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_float(name: str, *, default: float, minimum: float, maximum: float) -> float:
    load_dotenv()
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return min(max(parsed, minimum), maximum)


def _env_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    load_dotenv()
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return min(max(parsed, minimum), maximum)


def _configured_model() -> str:
    load_dotenv()
    return os.getenv("HF_DETECTOR_MODEL", DEFAULT_HF_DETECTOR_MODEL).strip() or DEFAULT_HF_DETECTOR_MODEL


def _configured_threshold() -> float:
    return _env_float(
        "HF_DETECTOR_FAKE_THRESHOLD",
        default=DEFAULT_FAKE_THRESHOLD,
        minimum=0.5,
        maximum=0.95,
    )


def _configured_video_frames() -> int:
    return _env_int(
        "HF_DETECTOR_VIDEO_FRAMES",
        default=DEFAULT_VIDEO_FRAMES,
        minimum=1,
        maximum=MAX_VIDEO_FRAMES,
    )


@lru_cache(maxsize=2)
def _load_transformers_classifier(model_name: str) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise DetectorUnavailableError(
            "Install torch and transformers to enable the Hugging Face detector."
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        trust_remote_code=False,
        use_safetensors=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, torch, device


@lru_cache(maxsize=1)
def _load_xrayon_classifier(model_name: str) -> tuple[Callable[[Any], Any], Any, Any, str]:
    try:
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise DetectorUnavailableError(
            "Install timm, torchvision, torch, and huggingface-hub to enable the xRayon detector."
        ) from exc

    checkpoint_path = hf_hub_download(
        repo_id=model_name,
        filename=XRAYON_CHECKPOINT,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("convnextv2_base", pretrained=False, num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(288),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform, model, torch, device


def _label_indices(model: Any, probability_count: int) -> tuple[int, int]:
    id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
    normalized: dict[int, str] = {}
    for key, value in id2label.items():
        try:
            normalized[int(key)] = str(value).lower()
        except (TypeError, ValueError):
            continue

    fake_index = next((index for index, label in normalized.items() if "fake" in label), None)
    real_index = next((index for index, label in normalized.items() if "real" in label), None)

    if fake_index is None:
        fake_index = 0
    if real_index is None:
        real_index = 1 if probability_count > 1 else 0

    last_index = max(probability_count - 1, 0)
    return min(fake_index, last_index), min(real_index, last_index)


def _classify_image_with_transformers(image_path: Path, model_name: str) -> ImageScore:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    processor, model, torch, device = _load_transformers_classifier(model_name)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().detach().cpu()

    values = probabilities.tolist()
    if not isinstance(values, list):
        values = [float(values)]

    fake_index, real_index = _label_indices(model, len(values))
    return ImageScore(
        fake_probability=float(values[fake_index]),
        real_probability=float(values[real_index]),
    )


def _classify_image_with_xrayon(image_path: Path, model_name: str) -> ImageScore:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    transform, model, torch, device = _load_xrayon_classifier(model_name)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().detach().cpu()

    values = probabilities.tolist()
    return ImageScore(
        fake_probability=float(values[1]),
        real_probability=float(values[0]),
    )


def _classify_image(image_path: Path, model_name: str) -> ImageScore:
    if model_name == DEFAULT_HF_DETECTOR_MODEL:
        return _classify_image_with_xrayon(image_path, model_name)
    return _classify_image_with_transformers(image_path, model_name)


def _confidence(fake_probability: float) -> str:
    margin = abs(fake_probability - 0.5)
    if margin >= 0.35:
        return "high"
    if margin >= 0.2:
        return "medium"
    return "low"


def _aggregate_scores(scores: list[ImageScore]) -> ImageScore:
    if len(scores) == 1:
        return scores[0]

    top_scores = sorted(scores, key=lambda score: score.fake_probability, reverse=True)[:3]
    fake_probability = mean(score.fake_probability for score in top_scores)
    return ImageScore(
        fake_probability=fake_probability,
        real_probability=1 - fake_probability,
    )


def analyze_media_with_detector(media: MediaInfo) -> DetectorSignal | None:
    """Run the optional Hugging Face detector and return a compact signal."""

    if not _env_flag("HF_DETECTOR_ENABLED", default=False):
        return None

    model_name = _configured_model()
    try:
        if media.media_type == "image":
            scores = [_classify_image(media.path, model_name)]
        else:
            with tempfile.TemporaryDirectory(prefix="deepfake-detector-frames-") as tmp_dir:
                frames = extract_video_frames(
                    media.path,
                    tmp_dir,
                    max_frames=_configured_video_frames(),
                )
                scores = [_classify_image(Path(frame), model_name) for frame in frames]
    except Exception:
        return None

    aggregate = _aggregate_scores(scores)
    label = "fake" if aggregate.fake_probability >= _configured_threshold() else "real"
    return DetectorSignal(
        model=model_name,
        media_type=media.media_type,
        label=label,
        fake_probability=aggregate.fake_probability,
        real_probability=aggregate.real_probability,
        confidence=_confidence(aggregate.fake_probability),
        frames_analyzed=len(scores),
    )
