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


def make_client():
    return VisionLLMClient(api_key="test_key", base_url="http://example.test")


# --- Tests ---

def test_safe_get_content_string():
    client = make_client()
    resp = {"choices": [{"message": {"content": "plain text"}}]}
    assert client._safe_get_content(resp) == "plain text"


def test_safe_get_content_list_of_parts():
    client = make_client()
    resp = {"choices": [{"message": {"content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}}]}
    assert client._safe_get_content(resp) == "part1\npart2"


def test_safe_get_content_alternate_shapes():
    client = make_client()
    # message as dict with 'text'
    resp = {"choices": [{"message": {"text": "simple text"}}]}
    assert client._safe_get_content(resp) == "simple text"


def test_extract_json_direct_and_embedded():
    client = make_client()
    assert client._extract_json('{"a": 1}') == {"a": 1}
    assert client._extract_json('prefix {"b": 2} suffix') == {"b": 2}
    assert client._extract_json('no json here') is None


def test_post_success():
    client = make_client()

    def fake_post(url, json=None, timeout=None):
        return DummyResponse({"ok": True}, status=200)

    original_post = client.session.post
    client.session.post = fake_post
    try:
        result = client._post(payload={"x": "y"}, timeout=1)
        assert result == {"ok": True}
    finally:
        client.session.post = original_post


def test_post_request_exception():
    client = make_client()

    def raise_exc(url, json=None, timeout=None):
        raise req_exceptions.RequestException("network error")

    original_post = client.session.post
    client.session.post = raise_exc
    try:
        result = client._post(payload={"x": "y"}, timeout=1)
        assert result is None
    finally:
        client.session.post = original_post


def test_call_vision_llm_parses_json_from_content():
    client = make_client()
    # Simulate a response where the message content is a string with extra text and JSON inside
    payload_response = {
        "choices": [
            {
                "message": {
                    "content": 'Some preamble text {"facial_analysis": {"face_shape": "oval"}, "hair_analysis": {"type": "wavy"}, "confidence_metrics": {"overall": 0.8}} trailing text.'
                }
            }
        ]
    }

    original_post = client._post
    client._post = lambda payload, timeout: payload_response
    try:
        res = client.call_vision_llm(base64_image="xxx", prompt="analyze")
        assert isinstance(res, dict)
        assert res.get("facial_analysis", {}).get("face_shape") == "oval"
    finally:
        client._post = original_post


def test_validate_face_presence_true_and_false():
    client = make_client()

    # True case: JSON present in message content
    true_resp = {"choices": [{"message": {"content": 'Here is result {"face_detected": true, "confidence": 0.9} end'}}]}
    original_post = client._post
    client._post = lambda payload, timeout: true_resp
    try:
        assert client.validate_face_presence("xxx") is True
    finally:
        client._post = original_post

    # False case: low confidence
    false_resp = {"choices": [{"message": {"content": '{"face_detected": true, "confidence": 0.4}'}}]}
    client._post = lambda payload, timeout: false_resp
    try:
        assert client.validate_face_presence("xxx") is False
    finally:
        client._post = original_post

    # False case: no face_detected
    noface_resp = {"choices": [{"message": {"content": '{"face_detected": false, "confidence": 0.9}'}}]}
    client._post = lambda payload, timeout: noface_resp
    try:
        assert client.validate_face_presence("xxx") is False
    finally:
        client._post = original_post


def run_all_tests():
    print("\U0001F680 Running LLM client tests...\n")
    tests = [
        test_safe_get_content_string,
        test_safe_get_content_list_of_parts,
        test_safe_get_content_alternate_shapes,
        test_extract_json_direct_and_embedded,
        test_post_success,
        test_post_request_exception,
        test_call_vision_llm_parses_json_from_content,
        test_validate_face_presence_true_and_false,
    ]

    passed = 0
    failed = 0

    for t in tests:
        try:
            t()
            print(f"\u2705 {t.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"\u274c {t.__name__} failed: {e}")
            failed += 1

    print(f"\n\U0001F4CA Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
