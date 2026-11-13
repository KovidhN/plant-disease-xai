"""Structured advisory generation with a deterministic local fallback."""

import json
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
RELIABILITY_VALUES = {"reliable", "uncertain"}
SEVERITY_VALUES = {"Low", "Medium", "High"}
ROUTING_VALUES = {"direct_result", "caution", "human_review"}
HEALTHY_SUFFIX = "___healthy"
MIN_CAM_COVERAGE = 0.1
MAX_CAM_COVERAGE = 95.0
CONFIDENCE_THRESHOLD = 0.75


class JevDecisionLayer:
    """Interpret model evidence without changing the model's predicted label."""

    def __init__(self) -> None:
        load_dotenv(override=True)
        self.model_name = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        self.base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL).rstrip("/")
        self.timeout = 120.0

    @property
    def is_live(self) -> bool:
        return True

    @property
    def mode(self) -> str:
        return "Live GenAI Decision Layer"

    def evaluate(self, evidence: dict[str, Any]) -> dict[str, str]:
        """Return a validated advisory, always retaining the supplied class label."""
        predicted_class = evidence.get("predicted_class")
        if not isinstance(predicted_class, str) or not predicted_class.strip():
            raise ValueError("evidence.predicted_class must be a non-empty string")

        evidence_fields = (
            "predicted_class",
            "confidence",
            "top_3",
            "gradcam_coverage",
        )
        if not all(field in evidence for field in evidence_fields):
            raise ValueError("evidence is missing required MobileNetV2 fields")
        model_evidence = {field: evidence[field] for field in evidence_fields}

        if predicted_class.endswith(HEALTHY_SUFFIX):
            result = self._healthy_advisory(evidence)
            result["status_reason"] = "Ollama skipped: model predicted a healthy leaf."
            return result

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Use only this MobileNetV2 evidence. Treat "
                                "predicted_class as a possible label, not a confirmed "
                                "diagnosis; do not invent symptoms or infer disease "
                                "severity from confidence or Grad-CAM. Keep the label "
                                "fixed and never recommend chemicals. Return only "
                                "JSON with exactly these string keys: reliability "
                                "(reliable|uncertain), severity (Low|Medium|High), "
                                "routing (direct_result|caution|human_review), "
                                "issue_explanation, recommended_action, and "
                                "hindi_summary. Each text value must be one short "
                                "sentence (max eight words); name predicted_class as "
                                "a prediction and write hindi_summary in Hindi. No "
                                "markdown or extra keys."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(model_evidence, ensure_ascii=False),
                        },
                    ],
                    "stream": False,
                    "format": "json",
                    "keep_alive": "30m",
                    "options": {
                        "num_ctx": 1024,
                        "num_predict": 220,
                        "temperature": 0.1,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Ollama response must be a JSON object")
            message = payload.get("message")
            if not isinstance(message, dict):
                raise ValueError("Ollama response is missing its message")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Ollama response is missing JSON content")
            advisory = self._validate_advisory(json.loads(content))
        except Exception as error:
            if isinstance(error, httpx.ReadTimeout):
                status_reason = "Ollama ReadTimeout after 120s"
            else:
                status_reason = f"Ollama {type(error).__name__}: {error}"
            logger.warning(
                "%s; using deterministic fallback",
                status_reason,
            )
            result = self._fallback(evidence)
            result["status_reason"] = status_reason
            result["engine_notice"] = status_reason
            return result

        advisory["predicted_class"] = predicted_class
        advisory["engine_mode"] = self.mode
        advisory["engine_provider"] = "Ollama"
        advisory["status_reason"] = "Ollama responded successfully."
        return advisory

    def _healthy_advisory(self, evidence: dict[str, Any]) -> dict[str, str]:
        predicted_class = str(evidence["predicted_class"])
        confidence = float(evidence.get("confidence", 0.0))
        coverage = float(evidence.get("gradcam_coverage", 0.0))
        cam_is_localized = MIN_CAM_COVERAGE <= coverage <= MAX_CAM_COVERAGE
        reliable = confidence >= CONFIDENCE_THRESHOLD and cam_is_localized
        return {
            "predicted_class": predicted_class,
            "reliability": "reliable" if reliable else "uncertain",
            "severity": "Low",
            "routing": "direct_result" if reliable else "caution",
            "issue_explanation": (
                "The model classified this leaf as healthy, so no disease issue "
                "was identified in the image."
            ),
            "recommended_action": (
                "Continue routine monitoring. If symptoms appear, capture a clear "
                "image and consult a local agricultural expert."
            ),
            "hindi_summary": (
                "मॉडल ने पत्ती को स्वस्थ बताया है। नियमित निगरानी जारी रखें और "
                "लक्षण दिखें तो स्थानीय कृषि विशेषज्ञ से सलाह लें।"
            ),
            "engine_mode": "Healthy result; Ollama not called",
            "engine_provider": "Ollama skipped for healthy prediction",
        }

    @staticmethod
    def _validate_advisory(payload: Any) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise ValueError("Decision response must be a JSON object")

        required_fields = {
            "reliability",
            "severity",
            "routing",
            "issue_explanation",
            "recommended_action",
            "hindi_summary",
        }
        if payload.keys() != required_fields:
            raise ValueError("Decision response must contain exactly the required fields")

        advisory = {field: payload[field] for field in required_fields}
        if not all(isinstance(value, str) and value.strip() for value in advisory.values()):
            raise ValueError("Decision fields must be non-empty strings")
        if advisory["reliability"] not in RELIABILITY_VALUES:
            raise ValueError("Invalid reliability value")
        if advisory["severity"] not in SEVERITY_VALUES:
            raise ValueError("Invalid severity value")
        if advisory["routing"] not in ROUTING_VALUES:
            raise ValueError("Invalid routing value")
        return advisory

    def _fallback(self, evidence: dict[str, Any]) -> dict[str, str]:
        predicted_class = str(evidence["predicted_class"])
        confidence = float(evidence.get("confidence", 0.0))
        coverage = float(evidence.get("gradcam_coverage", 0.0))
        healthy = predicted_class.endswith(HEALTHY_SUFFIX)
        cam_is_localized = MIN_CAM_COVERAGE <= coverage <= MAX_CAM_COVERAGE

        if confidence >= CONFIDENCE_THRESHOLD and cam_is_localized:
            reliability = "reliable"
            routing = "direct_result"
        elif confidence >= 0.5 and cam_is_localized:
            reliability = "uncertain"
            routing = "caution"
        else:
            reliability = "uncertain"
            routing = "human_review"

        severity = "Low" if healthy else "Medium"
        if healthy:
            action = (
                "The model classified this leaf as healthy. Continue routine "
                "monitoring and seek local expert advice if symptoms appear."
            )
        else:
            action = (
                "Recheck the plant and capture a clear image of affected leaves. "
                "Consider isolating visibly affected plants and consult a local "
                "agricultural extension professional before taking action."
            )

        return {
            "predicted_class": predicted_class,
            "reliability": reliability,
            "severity": severity,
            "routing": routing,
            "issue_explanation": (
                f"The model matched image patterns to {predicted_class}. "
                "This prediction alone cannot confirm symptoms or disease severity."
            ),
            "recommended_action": action,
            "hindi_summary": (
                f"मॉडल का अनुमान: {predicted_class}। यह केवल प्रारंभिक संकेत है; "
                "पुष्टि और सुरक्षित सलाह के लिए स्थानीय कृषि विशेषज्ञ से संपर्क करें।"
            ),
            "engine_mode": "Local Rule-Based Fallback",
        }
