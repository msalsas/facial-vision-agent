def test_call_vision_llm_happy_path(llm_client, happy_payload):
    original_post = llm_client.session.post
    try:
        llm_client.session.post = lambda url, json=None, timeout=None: type('R', (), {'status_code': 200, 'raise_for_status': lambda self: None, 'json': lambda self: happy_payload})()
        res = llm_client.call_vision_llm(base64_image="xxx")
        # Current implementation returns the raw content string
        assert isinstance(res, str)
        assert '"facial_analysis"' in res
    finally:
        llm_client.session.post = original_post


def test_call_vision_llm_non_json(llm_client, non_json_payload):
    original_post = llm_client.session.post
    try:
        llm_client.session.post = lambda url, json=None, timeout=None: type('R', (), {'status_code': 200, 'raise_for_status': lambda self: None, 'json': lambda self: non_json_payload})()

        # Current behavior: returns the raw text (non-json) string
        res = llm_client.call_vision_llm(base64_image="xxx")
        assert isinstance(res, str)
        assert 'This is plain text' in res
    finally:
        llm_client.session.post = original_post


def test_call_vision_llm_chunked(llm_client, chunked_json_payload):
    original_post = llm_client.session.post
    try:
        llm_client.session.post = lambda url, json=None, timeout=None: type('R', (), {'status_code': 200, 'raise_for_status': lambda self: None, 'json': lambda self: chunked_json_payload})()
        res = llm_client.call_vision_llm(base64_image="xxx")
        assert isinstance(res, str)
        # The chunked payload forms a combined string that includes the JSON pieces
        assert '"face_detected"' in res or 'face_detected' in res
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
