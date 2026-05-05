"""Gemini-backed deepfake risk analyzer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Union

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from pydantic import ValidationError

from app.media_utils import DEFAULT_MAX_UPLOAD_MB, MediaValidationError, validate_media_path
from app.prompts import DEEPFAKE_ANALYSIS_PROMPT
from app.schemas import DeepfakeReport

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODELS = ("gemini-2.5-flash-lite", "gemini-2.0-flash")
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
        "response_json_schema": DeepfakeReport.model_json_schema(),
    }


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


def _generate_content_with_fallbacks(
    client: genai.Client,
    uploaded_file: Any,
    *,
    model: str,
) -> Any:
    attempted_models: list[str] = []
    last_transient_error: genai_errors.APIError | None = None

    for candidate_model in _model_candidates(model):
        attempted_models.append(candidate_model)
        try:
            return client.models.generate_content(
                model=candidate_model,
                contents=[uploaded_file, DEEPFAKE_ANALYSIS_PROMPT],
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

    try:
        uploaded_file = client.files.upload(file=str(media.path))
        response = _generate_content_with_fallbacks(
            client,
            uploaded_file,
            model=model,
        )
    except AnalyzerError:
        raise
    except Exception as exc:  # pragma: no cover - exercised with integration tests/mocks
        raise AnalyzerError("Gemini analysis request failed.") from exc

    response_text = getattr(response, "text", None)
    if not response_text:
        raise AnalyzerResponseError("Gemini returned an empty response.")

    try:
        return DeepfakeReport.model_validate_json(response_text)
    except ValidationError as exc:
        raise AnalyzerResponseError("Gemini returned JSON that did not match the schema.") from exc
