import os
import json
import httpx

class JevDecisionLayer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("AI_GATEWAY_API_KEY")
        self.is_available = bool(self.api_key)
        
        # Configure the Vercel AI Gateway URL or fallback to OpenAI standard
        # Example Vercel URL: https://gateway.ai.vercel.com/v1/projects/{YOUR_PROJECT}/openai/chat/completions
        self.gateway_url = os.getenv("VERCEL_AI_GATEWAY_URL", "https://api.openai.com/v1/chat/completions") 

    def evaluate(self, structured_evidence: dict) -> dict:
        if not self.is_available:
            return self._fallback_evaluation(structured_evidence)
            
        return self._live_evaluation(structured_evidence)
        
    def _live_evaluation(self, structured_evidence: dict) -> dict:
        prompt = f"""
        You are Jev, a TypeSafe AI decision layer for a Plant Disease Detection application.
        Based on the following structured evidence from the visual models:
        {json.dumps(structured_evidence, indent=2)}
        
        Provide a JSON response with exactly these keys:
        - "image_quality": "good", "poor", or "retake"
        - "prediction_reliability": "reliable" or "uncertain"
        - "severity": "Low", "Medium", or "High"
        - "risk": "Low", "Medium", or "High"
        - "recommended_action": "show_result", "caution", "retake_image", or "human_review"
        - "explanation_sufficiency": "acceptable" or "questionable"
        - "final_routing": "direct_result", "caution", or "human_review"
        
        Do NOT change the predicted disease class. Only return valid JSON.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo", # Replace with your preferred model
            "messages": [
                {"role": "system", "content": "You are a strict JSON-only response bot."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.gateway_url, headers=headers, json=payload)
                response.raise_for_status()
                
                result_json = response.json()
                content = result_json["choices"][0]["message"]["content"]
                
                decision = json.loads(content)
                decision["status"] = "success"
                return decision
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 401:
                clean_error = "Invalid API Key. Please check your .env file."
            elif status_code == 429:
                clean_error = "Rate limit exceeded on the AI Gateway."
            elif status_code >= 500:
                clean_error = "Vercel AI Gateway is currently down."
            else:
                clean_error = f"API Error ({status_code})."
            
            print(f"Jev API Error: {e}")
            return self._fallback_evaluation(structured_evidence, error=clean_error)
            
        except Exception as e:
            print(f"Jev Connection Error: {e}")
            return self._fallback_evaluation(structured_evidence, error="Failed to connect to the AI Gateway.")

    def _fallback_evaluation(self, structured_evidence: dict, error=None) -> dict:
        confidence = structured_evidence.get("confidence", 0.0)
        is_reliable = confidence > 0.85
        
        return {
            "status": "unavailable" if not error else "error",
            "message": error or "Jev decision layer unavailable or failed.",
            "image_quality": "good",
            "prediction_reliability": "reliable" if is_reliable else "uncertain",
            "severity": "Medium",
            "risk": "Medium",
            "recommended_action": "show_result" if is_reliable else "human_review",
            "explanation_sufficiency": "acceptable",
            "final_routing": "direct_result" if is_reliable else "caution"
        }
