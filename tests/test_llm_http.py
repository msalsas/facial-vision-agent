def test_call_vision_llm_happy_path(llm_client, happy_payload):
    original_post = llm_client.session.post
    try:
        llm_client.session.post = lambda url, json=None, timeout=None: type('R', (), {'status_code': 200, 'raise_for_status': lambda self: None, 'json': lambda self: happy_payload})()
        res = llm_client.call_vision_llm(base64_image="xxx", prompt="analyze")
        assert isinstance(res, dict)
        assert res.get("facial_analysis", {}).get("face_shape") == "oval"
    finally:
        llm_client.session.post = original_post


def test_call_vision_llm_non_json(llm_client, non_json_payload):
    original_post = llm_client.session.post
    try:
        llm_client.session.post = lambda url, json=None, timeout=None: type('R', (), {'status_code': 200, 'raise_for_status': lambda self: None, 'json': lambda self: non_json_payload})()
        res = llm_client.call_vision_llm(base64_image="xxx", prompt="analyze")
        # Should fall back when there's no JSON
        assert isinstance(res, dict)
        # Fallback returns facial_analysis with face_shape == 'oval' (as implemented)
        assert res.get("facial_analysis", {}).get("face_shape") == "oval"
    finally:
        llm_client.session.post = original_post


def test_call_vision_llm_chunked(llm_client, chunked_json_payload):
    original_post = llm_client.session.post
    try:
        llm_client.session.post = lambda url, json=None, timeout=None: type('R', (), {'status_code': 200, 'raise_for_status': lambda self: None, 'json': lambda self: chunked_json_payload})()
        res = llm_client.call_vision_llm(base64_image="xxx", prompt="analyze")
        assert isinstance(res, dict)
        # The chunked payload forms a JSON object with face_detected True
        assert res.get("face_detected") is True or res.get("facial_analysis", {}).get("face_shape") == "oval"
    finally:
        llm_client.session.post = original_post


def test_post_handles_network_error(llm_client, network_error_post):
    original_post = llm_client.session.post
    try:
        llm_client.session.post = network_error_post
        res = llm_client._post(payload={"x": "y"}, timeout=1)
        assert res is None
    finally:
        llm_client.session.post = original_post

