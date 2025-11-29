import pytest
import requests
from requests import exceptions as req_exceptions

from facial_vision_agent.llm_client import VisionLLMClient


class DummyResponse:
    def __init__(self, data=None, status=200):
        self._data = data or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Status: {self.status_code}")

    def json(self):
        return self._data


@pytest.fixture
def llm_client():
    """Provide a VisionLLMClient configured for tests."""
    return VisionLLMClient(api_key="test_key", base_url="http://example.test")


@pytest.fixture
def happy_payload():
    """Payload simulating a happy-path LLM response with embedded JSON in content."""
    return {
        "choices": [
            {"message": {"content": 'Preamble {"facial_analysis": {"face_shape": "oval"}, "hair_analysis": {"type": "wavy"}, "confidence_metrics": {"overall": 0.9}} trailing'}}
        ]
    }


@pytest.fixture
def non_json_payload():
    """Payload where content is plain non-JSON text."""
    return {
        "choices": [
            {"message": {"content": 'This is plain text and not JSON.'}}
        ]
    }


@pytest.fixture
def chunked_json_payload():
    """Payload where content is a list of text parts that when joined form JSON."""
    # When joined with newlines by _safe_get_content this will become valid JSON
    return {
        "choices": [
            {"message": {"content": [{"type": "text", "text": '{"face_detected": true,'}, {"type": "text", "text": ' "confidence": 0.95}'}]}}
        ]
    }


@pytest.fixture
def network_error_post():
    """Return a callable that raises a requests.RequestException when called (to patch session.post)."""
    def _raise(*args, **kwargs):
        raise req_exceptions.RequestException("network error")
    return _raise
