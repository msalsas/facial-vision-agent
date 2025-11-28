from typing import Dict, Any, Optional
import requests
import json
import re
import logging
from requests import Session, exceptions as req_exceptions

logger = logging.getLogger(__name__)


class VisionLLMClient:
    """Client for interacting with vision-capable LLMs."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        # Reuse a session for connection pooling and consistent headers
        self.session: Session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def call_vision_llm(self, base64_image: str, prompt: str, model: str = "meta-llama/llama-3.2-11b-vision-instruct") -> Dict[str, Any]:
        """
        Call vision-capable LLM for analysis.
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3,
        }

        try:
            result = self._post(payload, timeout=30)
            if not result:
                return self._get_fallback_analysis()

            content = self._safe_get_content(result)
            if not content:
                return self._get_fallback_analysis()

            parsed = self._extract_json(content)
            if parsed is not None:
                return parsed
            return self._get_fallback_analysis()

        except Exception:
            logger.exception("Vision LLM call failed")
            return self._get_fallback_analysis()

    def validate_face_presence(self, base64_image: str) -> bool:
        """
        Validate that the image contains at least one face.
        """
        prompt = (
            "Analyze this image and determine if it contains a human face.\n\n"
            "Return only a JSON object:\n"
            "{\n    \"face_detected\": true/false,\n    \"confidence\": 0.0-1.0\n}\n\n"
            "Be strict: only return true if you can clearly see a human face."
        )

        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1,
        }

        try:
            result = self._post(payload, timeout=15)
            if not result:
                return False

            content = self._safe_get_content(result)
            if not content:
                return False

            parsed = self._extract_json(content)
            if parsed and isinstance(parsed, dict):
                face_detected = parsed.get('face_detected', False)
                confidence = parsed.get('confidence', 0.0)
                try:
                    confidence = float(confidence)
                except Exception:
                    confidence = 0.0
                return bool(face_detected) and confidence > 0.7

            return False

        except Exception:
            logger.exception("Face validation failed")
            return False

    def _post(self, payload: Dict[str, Any], timeout: int) -> Optional[Dict[str, Any]]:
        try:
            resp = self.session.post(self.base_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (req_exceptions.RequestException, ValueError) as e:
            logger.exception("Request to LLM failed: %s", e)
            return None

    def _safe_get_content(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Safely extract textual content from various possible response shapes.
        """
        if not isinstance(result, dict):
            return None
        choices = result.get('choices')
        if not choices or not isinstance(choices, list):
            return None
        first = choices[0] if len(choices) > 0 else None
        if not isinstance(first, dict):
            return None
        message = first.get('message') or first.get('text') or {}
        if not isinstance(message, dict):
            # If message itself is a string, try to return it
            if isinstance(message, str):
                return message
            return None
        content = message.get('content') or message.get('text') or None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    # Common shape: {"type": "text", "text": "..."}
                    if 'text' in item:
                        parts.append(str(item['text']))
                    else:
                        try:
                            parts.append(json.dumps(item))
                        except Exception:
                            parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        try:
            return json.dumps(content)
        except Exception:
            return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text, handling cases where LLM adds extra text.
        """
        if not isinstance(text, str):
            return None
        # Non-greedy match for first JSON object
        json_match = re.search(r'(\{.*?\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.debug('Regex-extracted JSON failed to decode')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug('Direct JSON load failed')
            return None

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """
        Fallback analysis when LLM fails.
        """
        return {
            "facial_analysis": {
                "face_shape": "oval",
                "facial_proportions": {
                    "width_height_ratio": 0.75,
                    "jawline_strength": "medium",
                    "forehead_height": "medium"
                },
                "prominent_features": ["balanced_proportions"]
            },
            "hair_analysis": {
                "type": "straight",
                "length": "medium",
                "color": "brown",
                "density": "medium",
                "condition": "healthy"
            },
            "confidence_metrics": {
                "face_detection": 0.3,
                "hair_analysis": 0.3,
                "overall": 0.3
            }
        }
