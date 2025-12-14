from time import sleep
from typing import Dict, Any, Optional
import requests
import json
import logging
from .prompts import AnalysisPrompts
from requests import Session, exceptions as req_exceptions

logger = logging.getLogger(__name__)


class VisionLLMClient:
    """Client for interacting with vision-capable LLMs."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        self.prompts = AnalysisPrompts()
        # Reuse a session for connection pooling and consistent headers
        self.session: Session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def call_vision_llm(self, base64_image: str, model: str = "meta-llama/llama-3.2-11b-vision-instruct") -> str:
        """
        Call vision-capable LLM for analysis.
        """
        prompt = self.prompts.get_comprehensive_analysis_prompt()
        system_prompt = self.prompts.get_comprehensive_analysis_system_prompt()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.0,
        }

        try:
            result = self._post(payload, timeout=30)
            if not result:
                raise ValueError("No result returned from LLM")

            content = self._safe_get_content(result)
            if not content:
                raise ValueError("No content found in the result")

            logger.debug("call_vision_llm: content=%s", content)

            return content

        except Exception:
            logger.exception("Vision LLM call failed")
            raise

    def validate_face_presence(self, base64_image: str, retry: int = 0) -> bool:
        """
        Validate that the image contains at least one face. The prompt now asks the LLM
        to reply with a single character: 'Y' (yes) if a face is present, or 'N' (no) if not.
        The LLM is instructed to apply a fixed internal confidence threshold of 0.7.
        """
        prompt = self.prompts.get_face_validation_prompt()

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
            "max_tokens": 1,
            "temperature": 0.0,
        }

        try:
            result = self._post(payload, timeout=15)

            if not result:
                logger.debug("validate_face_presence: no result from _post")
                return False

            content = self._safe_get_content(result)

            logger.debug("validate_face_presence - extracted content: %r", content)
            if not content:
                logger.debug("validate_face_presence: empty content, returning False")
                return False

            normalized = str(content).strip().lower()
            if normalized[0] == 'y':
                return True
            elif normalized == 'safe':
                sleep(1)
                if retry > 2:
                    logger.debug("validate_face_presence: exceeded max retries on 'safe' response, returning False")
                    return False
                return self.validate_face_presence(base64_image, retry + 1)
            else:
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

