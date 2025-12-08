from facial_vision_agent import FacialVisionAgent
from agent_core_framework import AgentTask


def test_accepts_base64_image_payload():
    agent = FacialVisionAgent("test_api_key")

    # Mock validation and LLM call
    original_validate = agent._validate_face_presence
    agent._validate_face_presence = lambda x: True

    original_llm = agent._call_vision_llm
    agent._call_vision_llm = lambda base64_image: {
        "facial_analysis": {"face_shape": "round"},
        "hair_analysis": {"type": "curly"},
        "confidence_metrics": {"overall": 0.85},
    }

    try:
        task = AgentTask(type="analyze_image", payload={"base64_image": "dGVzdA=="})
        response = agent.process(task)

        assert response.success
        assert "detected_features" in response.data
        assert response.data["detected_features"]["facial_analysis"]["face_shape"] == "round"
    finally:
        agent._validate_face_presence = original_validate
        agent._call_vision_llm = original_llm


def test_base64_face_validation_fail():
    agent = FacialVisionAgent("test_api_key")

    # Mock validation to False
    original_validate = agent._validate_face_presence
    agent._validate_face_presence = lambda x: False

    try:
        task = AgentTask(type="analyze_image", payload={"base64_image": "invalid"})
        response = agent.process(task)

        assert not response.success
        assert "No human face detected" in response.error
    finally:
        agent._validate_face_presence = original_validate
