import json

import httpx
import pytest

from decision_layer import JevDecisionLayer


MODEL_EVIDENCE = {
    "predicted_class": "Tomato___Late_blight",
    "confidence": 0.9,
    "top_3": [
        {"class_name": "Tomato___Late_blight", "confidence": 90.0},
        {"class_name": "Tomato___Early_blight", "confidence": 7.0},
        {"class_name": "Tomato___healthy", "confidence": 3.0},
    ],
    "gradcam_coverage": 25.0,
    "plant_filter_used": "Tomato",
}

ADVISORY = {
    "reliability": "reliable",
    "severity": "Medium",
    "routing": "caution",
    "issue_explanation": "The image pattern is consistent with the model label.",
    "recommended_action": "Inspect nearby leaves and consult a local expert.",
    "hindi_summary": "पत्तियों की जाँच करें और स्थानीय विशेषज्ञ से सलाह लें।",
}


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch):
    monkeypatch.setattr("decision_layer.load_dotenv", lambda **kwargs: None)


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self.content}}


def test_loads_dotenv_with_override_and_defaults_to_qwen(monkeypatch):
    load_options = {}
    monkeypatch.setattr(
        "decision_layer.load_dotenv", lambda **kwargs: load_options.update(kwargs)
    )
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)

    layer = JevDecisionLayer()

    assert load_options == {"override": True}
    assert layer.model_name == "qwen2.5:3b"
    assert layer.base_url == "http://localhost:11434"
    assert layer.timeout == 120.0
    assert layer.mode == "Live GenAI Decision Layer"


def test_posts_required_payload_and_preserves_mobilenet_label(monkeypatch):
    request = {}

    def fake_post(url, **kwargs):
        request.update(url=url, **kwargs)
        return FakeResponse(json.dumps(ADVISORY))

    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.local:11434/")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "120")
    monkeypatch.setattr("decision_layer.httpx.post", fake_post)
    layer = JevDecisionLayer()

    result = layer.evaluate(MODEL_EVIDENCE)
    body = request["json"]
    sent_evidence = json.loads(body["messages"][1]["content"])

    assert request["url"] == "http://ollama.local:11434/api/chat"
    assert request["timeout"] == 120.0
    assert body["model"] == "qwen2.5:3b"
    assert body["format"] == "json"
    assert body["stream"] is False
    assert body["keep_alive"] == "30m"
    assert body["options"] == {
        "num_ctx": 1024,
        "num_predict": 220,
        "temperature": 0.1,
    }
    assert sent_evidence == {
        "predicted_class": MODEL_EVIDENCE["predicted_class"],
        "confidence": MODEL_EVIDENCE["confidence"],
        "top_3": MODEL_EVIDENCE["top_3"],
        "gradcam_coverage": MODEL_EVIDENCE["gradcam_coverage"],
    }
    assert result["predicted_class"] == "Tomato___Late_blight"
    assert result["engine_mode"] == "Live GenAI Decision Layer"
    assert set(ADVISORY).issubset(result)


def test_http_failure_uses_deterministic_fallback(monkeypatch):
    def fail_request(*args, **kwargs):
        raise httpx.ConnectError("Ollama is unavailable")

    monkeypatch.setattr("decision_layer.httpx.post", fail_request)
    layer = JevDecisionLayer()

    result = layer.evaluate(MODEL_EVIDENCE)

    assert result["reliability"] == "reliable"
    assert result["routing"] == "direct_result"
    assert result["predicted_class"] == MODEL_EVIDENCE["predicted_class"]
    assert result["engine_mode"] == "Local Rule-Based Fallback"
    assert "engine_notice" in result


def test_invalid_model_json_uses_deterministic_fallback(monkeypatch):
    monkeypatch.setattr(
        "decision_layer.httpx.post",
        lambda *args, **kwargs: FakeResponse('{"routing":"bad"}'),
    )
    layer = JevDecisionLayer()

    result = layer.evaluate(MODEL_EVIDENCE)

    assert result["predicted_class"] == MODEL_EVIDENCE["predicted_class"]
    assert result["engine_mode"] == "Local Rule-Based Fallback"
    assert "engine_notice" in result


def test_fallback_routes_uncertain_gradcam_for_human_review(monkeypatch):
    monkeypatch.setattr(
        "decision_layer.httpx.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            httpx.ConnectError("Ollama is unavailable")
        ),
    )
    evidence = {**MODEL_EVIDENCE, "gradcam_coverage": 0.0}

    result = JevDecisionLayer().evaluate(evidence)

    assert result["reliability"] == "uncertain"
    assert result["routing"] == "human_review"


def test_healthy_prediction_skips_ollama(monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Ollama should not be called for a healthy prediction")

    monkeypatch.setattr("decision_layer.httpx.post", fail_if_called)
    evidence = {
        **MODEL_EVIDENCE,
        "predicted_class": "Tomato___healthy",
        "confidence": 0.95,
    }

    result = JevDecisionLayer().evaluate(evidence)

    assert result["predicted_class"] == "Tomato___healthy"
    assert result["severity"] == "Low"
    assert result["routing"] == "direct_result"
    assert result["engine_provider"] == "Ollama skipped for healthy prediction"
    assert set(ADVISORY).issubset(result)
