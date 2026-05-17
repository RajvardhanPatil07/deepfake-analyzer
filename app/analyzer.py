"""Gemini-backed deepfake risk analyzer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Union

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from pydantic import ValidationError

from app.detector import analyze_media_with_detector
from app.media_utils import DEFAULT_MAX_UPLOAD_MB, MediaValidationError, validate_media_path
from app.prompts import DEEPFAKE_ANALYSIS_PROMPT
from app.schemas import DeepfakeReport, DetectorSignal, EvidenceItem

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_FALLBACK_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
TRANSIENT_GEMINI_STATUS_CODES = {429, 500, 503, 504}


class AnalyzerError(RuntimeError):
    """Base class for user-safe analyzer errors."""


class AnalyzerConfigurationError(AnalyzerError):
    """Raised when required analyzer configuration is missing."""


class AnalyzerResponseError(AnalyzerError):
    """Raised when Gemini returns an invalid or unusable response."""


class AnalyzerTemporaryError(AnalyzerError):
    """Raised when Gemini is temporarily unavailable after fallback attempts."""


def _get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise AnalyzerConfigurationError(
            "GEMINI_API_KEY is not set. Add it to your environment or a .env file."
        )
    return api_key


def create_client() -> genai.Client:
    """Create a Google GenAI client from GEMINI_API_KEY."""

    return genai.Client(api_key=_get_api_key())


def _generation_config() -> dict[str, Any]:
    return {
        "response_mime_type": "application/json",
        "response_json_schema": _gemini_response_schema(),
    }


def _gemini_response_schema() -> dict[str, Any]:
    schema = DeepfakeReport.model_json_schema()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("detector_signal", None)
    required = schema.get("required")
    if isinstance(required, list) and "detector_signal" in required:
        required.remove("detector_signal")
    return schema


def _configured_fallback_models() -> tuple[str, ...]:
    load_dotenv()
    configured = os.getenv("GEMINI_FALLBACK_MODELS")
    if not configured:
        return DEFAULT_FALLBACK_MODELS
    return tuple(model.strip() for model in configured.split(",") if model.strip())


def _model_key(model: str) -> str:
    return model.removeprefix("models/")


def _model_candidates(primary_model: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in (primary_model, *_configured_fallback_models()):
        key = _model_key(candidate)
        if key in seen:
            continue
        candidates.append(candidate)
        seen.add(key)
    return candidates


def _api_status_code(exc: genai_errors.APIError) -> int | None:
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _is_transient_gemini_error(exc: genai_errors.APIError) -> bool:
    status_code = _api_status_code(exc)
    return status_code in TRANSIENT_GEMINI_STATUS_CODES


def _model_list_text(models: Iterable[str]) -> str:
    return ", ".join(_model_key(model) for model in models)


def _analysis_prompt(detector_signal: DetectorSignal | None) -> str:
    if detector_signal is None:
        return DEEPFAKE_ANALYSIS_PROMPT

    fake_percent = round(detector_signal.fake_probability * 100)
    real_percent = round(detector_signal.real_probability * 100)
    return (
        f"{DEEPFAKE_ANALYSIS_PROMPT}\n\n"
        "Separate classifier signal available before this analysis:\n"
        f"- Model: {detector_signal.model}\n"
        f"- Classifier label: {detector_signal.label}\n"
        f"- Fake probability: {fake_percent}%\n"
        f"- Real probability: {real_percent}%\n"
        f"- Frames/images analyzed: {detector_signal.frames_analyzed}\n\n"
        "Use this classifier output as one non-forensic signal. If it conflicts "
        "with visible evidence, explain the uncertainty instead of ignoring it. "
        "Do not include detector_signal in the JSON; the app will add it."
    )


def _generate_content_with_fallbacks(
    client: genai.Client,
    uploaded_file: Any,
    *,
    model: str,
    prompt: str,
) -> Any:
    attempted_models: list[str] = []
    last_transient_error: genai_errors.APIError | None = None

    for candidate_model in _model_candidates(model):
        attempted_models.append(candidate_model)
        try:
            return client.models.generate_content(
                model=candidate_model,
                contents=[uploaded_file, prompt],
                config=_generation_config(),
            )
        except genai_errors.APIError as exc:
            if not _is_transient_gemini_error(exc):
                raise AnalyzerError(
                    "Gemini rejected the analysis request. Check model access, quota, "
                    "and the uploaded media."
                ) from exc
            last_transient_error = exc

    attempted = _model_list_text(attempted_models)
    raise AnalyzerTemporaryError(
        f"Gemini is temporarily overloaded or rate limited after trying: {attempted}. "
        "Please retry in a few minutes."
    ) from last_transient_error


LABEL_RANK = {
    "likely_authentic": 0,
    "uncertain": 1,
    "suspicious": 2,
    "likely_manipulated": 3,
}


def _higher_risk_label(current: str, candidate: str) -> str:
    return candidate if LABEL_RANK[candidate] > LABEL_RANK[current] else current


def _detector_evidence(signal: DetectorSignal, severity: str) -> EvidenceItem:
    fake_percent = round(signal.fake_probability * 100)
    frame_text = "image" if signal.frames_analyzed == 1 else f"{signal.frames_analyzed} frames"
    return EvidenceItem(
        category="Classifier signal",
        finding=(
            f"{signal.model} classified the {frame_text} as {signal.label} "
            f"with {fake_percent}% fake probability."
        ),
        severity=severity,
        timestamp=None,
    )


def _with_unique_item(items: list[str], item: str) -> list[str]:
    return items if item in items else [*items, item]


def _calibrate_with_detector(
    report: DeepfakeReport,
    detector_signal: DetectorSignal | None,
) -> DeepfakeReport:
    if detector_signal is None:
        return report

    updates: dict[str, Any] = {"detector_signal": detector_signal}
    fake_probability = detector_signal.fake_probability

    if detector_signal.label != "fake" or fake_probability < 0.55:
        return report.model_copy(update=updates)

    if fake_probability >= 0.75:
        minimum_score = max(72, round(fake_probability * 100))
        target_label = "likely_manipulated"
        severity = "high"
    elif fake_probability >= 0.6:
        minimum_score = max(55, round(fake_probability * 100))
        target_label = "suspicious"
        severity = "medium"
    else:
        minimum_score = max(46, round(fake_probability * 100))
        target_label = "suspicious"
        severity = "low"

    original_score = report.risk_score
    original_label = report.label
    calibrated_score = max(report.risk_score, minimum_score)
    calibrated_label = _higher_risk_label(report.label, target_label)

    evidence = [*report.evidence, _detector_evidence(detector_signal, severity)]
    limitations = _with_unique_item(
        list(report.limitations),
        "The Hugging Face classifier is a screening signal, not forensic proof; "
        "compression, editing, and out-of-domain images can still cause false positives.",
    )

    summary = report.summary
    if original_score < 46 or original_label == "likely_authentic":
        fake_percent = round(fake_probability * 100)
        summary = (
            f"A local classifier found an elevated synthetic-media signal "
            f"({fake_percent}% fake probability), so the final risk was raised even "
            f"though Gemini's artifact review was less decisive. {report.summary}"
        )

    confidence = report.confidence
    if detector_signal.confidence == "high" and original_score >= 46:
        confidence = "high"
    elif original_score < 46:
        confidence = "medium" if detector_signal.confidence != "low" else "low"

    updates.update(
        {
            "label": calibrated_label,
            "risk_score": calibrated_score,
            "confidence": confidence,
            "summary": summary,
            "evidence": evidence,
            "limitations": limitations,
        }
    )
    return report.model_copy(update=updates)


def analyze_media_file(
    file_path: Union[str, Path],
    *,
    model: str = DEFAULT_MODEL,
    max_mb: int = DEFAULT_MAX_UPLOAD_MB,
) -> DeepfakeReport:
    """Validate, upload, analyze, and parse a media file."""

    try:
        media = validate_media_path(file_path, max_mb=max_mb)
    except MediaValidationError:
        raise

    client = create_client()
    detector_signal = analyze_media_with_detector(media)

    try:
        uploaded_file = client.files.upload(file=str(media.path))
        response = _generate_content_with_fallbacks(
            client,
            uploaded_file,
            model=model,
            prompt=_analysis_prompt(detector_signal),
        )
    except AnalyzerError:
        raise
    except Exception as exc:  # pragma: no cover - exercised with integration tests/mocks
        raise AnalyzerError("Gemini analysis request failed.") from exc

    response_text = getattr(response, "text", None)
    if not response_text:
        raise AnalyzerResponseError("Gemini returned an empty response.")

    try:
        report = DeepfakeReport.model_validate_json(response_text)
    except ValidationError as exc:
        raise AnalyzerResponseError("Gemini returned JSON that did not match the schema.") from exc
    return _calibrate_with_detector(report, detector_signal)
