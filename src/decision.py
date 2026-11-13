def formulate_structured_evidence(prediction_result: dict, gradcam_stats: dict, metadata: dict = None) -> dict:
    return {
        "predicted_disease": prediction_result.get("predicted_class"),
        "confidence": prediction_result.get("confidence"),
        "top_k": prediction_result.get("top_k"),
        "gradcam_coverage": gradcam_stats.get("coverage", 0.5),
        "metadata": metadata or {}
    }
