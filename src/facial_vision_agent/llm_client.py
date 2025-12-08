from time import sleep
from typing import Dict, Any, Optional
import requests
import json
import re
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

    def call_vision_llm(self, base64_image: str, model: str = "meta-llama/llama-3.2-11b-vision-instruct") -> Dict[str, Any]:
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

            logger.debug("call_vision_llm: raw content=%s", content)
            parsed = self._extract_json(content)
            if parsed is not None:
                return parsed

            logger.warning("call_vision_llm: initial response had no JSON, attempting one repair pass")
            repair_prompt = self.prompts.get_comprehensive_analysis_retry_prompt()

            repair_messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": str(content)}]},
                {"role": "user", "content": [{"type": "text", "text": repair_prompt}]},
            ]

            repair_payload = {
                "model": model,
                "messages": repair_messages,
                "max_tokens": 1000,
                "temperature": 0.0,
            }

            repair_result = self._post(repair_payload, timeout=20)
            if not repair_result:
                raise ValueError("No result returned from LLM on repair attempt")

            repair_content = self._safe_get_content(repair_result)
            logger.debug("call_vision_llm: repair raw content=%s", repair_content)
            parsed = self._extract_json(repair_content)
            if parsed is not None:
                return parsed

            raise ValueError("Parsed content is None after repair attempt")

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
                return self.validate_face_presence(base64_image, ++retry)
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

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text, handling cases where LLM adds extra text.
        """
        if not isinstance(text, str):
            return None

        cleaned = re.sub(r"```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        logger.debug("_extract_json: cleaned text=%s", cleaned)

        length = len(cleaned)
        open_chars = {'{': '}', '[': ']'}

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        for start_idx, ch in enumerate(cleaned):
            if ch not in open_chars:
                continue
            stack = [ch]
            for i in range(start_idx + 1, length):
                c = cleaned[i]
                if c == '"' or c == "'":
                    quote = c
                    j = i + 1
                    while j < length:
                        if cleaned[j] == '\\':
                            j += 2
                            continue
                        if cleaned[j] == quote:
                            break
                        j += 1
                    i = j
                    continue
                if c in open_chars:
                    stack.append(c)
                elif c in open_chars.values():
                    if not stack:
                        break
                    last = stack[-1]
                    if open_chars.get(last) == c:
                        stack.pop()
                    else:
                        stack.pop()
                if not stack:
                    candidate = cleaned[start_idx:i + 1]
                    candidate = candidate.strip()
                    try:
                        parsed = json.loads(candidate)
                        logger.debug("_extract_json: successfully parsed candidate starting at %d", start_idx)
                        return parsed
                    except json.JSONDecodeError:
                        break

        if "'" in cleaned and '"' not in cleaned:
            coerced = cleaned.replace("'", '"')
            try:
                return json.loads(coerced)
            except json.JSONDecodeError:
                logger.debug("_extract_json: coercion with single->double quotes failed")

        logger.debug('_extract_json: No parsable JSON object found in text')
        return None
