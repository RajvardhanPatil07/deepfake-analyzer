"""Tests for Gemini analyzer fallback behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from google.genai import errors as genai_errors

from app import analyzer
from app.schemas import DeepfakeReport


def valid_report_json() -> str:
    return json.dumps(
        {
            "label": "uncertain",
            "risk_score": 35,
            "confidence": "medium",
            "summary": "The media has only weak, ambiguous manipulation indicators.",
            "evidence": [],
            "limitations": ["Single image analysis cannot prove authenticity."],
            "recommended_next_steps": ["Review source provenance."],
        }
    )


@dataclass
class FakeResponse:
    text: str


class FakeFiles:
    def upload(self, file: str) -> object:
        return object()


class FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def generate_content(self, *, model: str, contents: list[Any], config: dict[str, Any]) -> FakeResponse:
        self.calls.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(text=outcome)


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.files = FakeFiles()
        self.models = FakeModels(outcomes)


def make_jpg(tmp_path: Path) -> Path:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake-jpg")
    return image


def unavailable_error() -> genai_errors.ServerError:
    return genai_errors.ServerError(
        503,
        {
            "error": {
                "code": 503,
                "message": "This model is currently experiencing high demand.",
                "status": "UNAVAILABLE",
            }
        },
        None,
    )


def test_analyze_media_file_uses_fallback_model_after_transient_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_client = FakeClient([unavailable_error(), valid_report_json()])
    monkeypatch.setattr(analyzer, "create_client", lambda: fake_client)

    report = analyzer.analyze_media_file(make_jpg(tmp_path))

    assert isinstance(report, DeepfakeReport)
    assert fake_client.models.calls == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def test_analyze_media_file_reports_temporary_error_after_all_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_client = FakeClient([unavailable_error(), unavailable_error(), unavailable_error()])
    monkeypatch.setattr(analyzer, "create_client", lambda: fake_client)

    with pytest.raises(analyzer.AnalyzerTemporaryError, match="temporarily overloaded"):
        analyzer.analyze_media_file(make_jpg(tmp_path))

    assert fake_client.models.calls == [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]
