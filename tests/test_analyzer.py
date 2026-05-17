"""Tests for Gemini analyzer fallback behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from google.genai import errors as genai_errors

from app import analyzer
from app.schemas import DeepfakeReport, DetectorSignal


def valid_report_json(
    *,
    label: str = "uncertain",
    risk_score: int = 35,
    confidence: str = "medium",
    summary: str = "The media has only weak, ambiguous manipulation indicators.",
) -> str:
    return json.dumps(
        {
            "label": label,
            "risk_score": risk_score,
            "confidence": confidence,
            "summary": summary,
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


@pytest.fixture(autouse=True)
def disable_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer, "analyze_media_with_detector", lambda media: None)


def test_analyze_media_file_uses_fallback_model_after_transient_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_client = FakeClient([unavailable_error(), valid_report_json()])
    monkeypatch.setattr(analyzer, "create_client", lambda: fake_client)

    report = analyzer.analyze_media_file(make_jpg(tmp_path))

    assert isinstance(report, DeepfakeReport)
    assert fake_client.models.calls == ["gemini-2.5-pro", "gemini-2.5-flash"]


def test_analyze_media_file_reports_temporary_error_after_all_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_client = FakeClient([unavailable_error(), unavailable_error(), unavailable_error()])
    monkeypatch.setattr(analyzer, "create_client", lambda: fake_client)

    with pytest.raises(analyzer.AnalyzerTemporaryError, match="temporarily overloaded"):
        analyzer.analyze_media_file(make_jpg(tmp_path))

    assert fake_client.models.calls == [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]


def test_analyze_media_file_raises_low_gemini_result_with_fake_detector_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_client = FakeClient(
        [
            valid_report_json(
                label="likely_authentic",
                risk_score=12,
                confidence="medium",
                summary="No strong visual manipulation artifacts are visible.",
            )
        ]
    )
    detector_signal = DetectorSignal(
        model="xRayon/convnext-ai-images-detector",
        media_type="image",
        label="fake",
        fake_probability=0.88,
        real_probability=0.12,
        confidence="high",
        frames_analyzed=1,
    )
    monkeypatch.setattr(analyzer, "create_client", lambda: fake_client)
    monkeypatch.setattr(analyzer, "analyze_media_with_detector", lambda media: detector_signal)

    report = analyzer.analyze_media_file(make_jpg(tmp_path))

    assert report.label == "likely_manipulated"
    assert report.risk_score == 88
    assert report.detector_signal == detector_signal
    assert report.evidence[-1].category == "Classifier signal"


def test_low_resolution_portrait_is_not_high_confidence_authentic() -> None:
    report = DeepfakeReport.model_validate_json(
        valid_report_json(
            label="likely_authentic",
            risk_score=15,
            confidence="high",
            summary="No strong visual manipulation artifacts are visible.",
        )
    )
    context = analyzer.ImageContext(width=275, height=183, has_prominent_face=True)

    calibrated = analyzer._calibrate_low_information_portrait(report, context)

    assert calibrated.label == "uncertain"
    assert calibrated.risk_score == 42
    assert calibrated.confidence == "low"
    assert calibrated.evidence[-1].category == "Assessment quality"
    assert "Low-resolution face portraits" in calibrated.limitations[-1]


def test_low_resolution_portrait_guard_does_not_lower_existing_risk() -> None:
    report = DeepfakeReport.model_validate_json(
        valid_report_json(
            label="suspicious",
            risk_score=60,
            confidence="medium",
            summary="Several visual anomalies are visible.",
        )
    )
    context = analyzer.ImageContext(width=275, height=183, has_prominent_face=True)

    calibrated = analyzer._calibrate_low_information_portrait(report, context)

    assert calibrated == report
