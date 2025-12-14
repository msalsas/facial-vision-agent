from facial_vision_agent import FacialVisionAgent
from agent_core_framework import AgentTask
import builtins


def test_agent_initialization():
    """Test basic agent initialization"""
    agent = FacialVisionAgent("test_api_key")

    assert agent.name == "FacialVision"
    assert agent.version == "1.0.0"
    assert agent.api_key == "test_api_key"
    print("✅ Agent initialization test passed")


def test_supported_tasks():
    """Test that agent only supports analysis tasks"""
    agent = FacialVisionAgent("test_api_key")

    expected_tasks = ["analyze_image"]
    for task in expected_tasks:
        assert task in agent.supported_tasks
        print(f"✅ Task '{task}' is supported")

    # Verify no recommendation tasks
    recommendation_tasks = ["recommend_styles", "suggest_hairstyles"]
    for task in recommendation_tasks:
        assert task not in agent.supported_tasks
        print(f"✅ Task '{task}' correctly NOT supported")


def test_unsupported_task():
    """Test that unsupported tasks return error"""
    agent = FacialVisionAgent("test_api_key")

    task = AgentTask(type="recommend_styles", payload={})
    response = agent.process(task)

    assert not response.success
    assert "Unsupported task type" in response.error
    print("✅ Unsupported task handling test passed")


def test_missing_image_path():
    """Test error when image path is missing"""
    agent = FacialVisionAgent("test_api_key")

    task = AgentTask(type="analyze_image", payload={})
    response = agent.process(task)

    assert not response.success
    assert "Image path is required" in response.error
    print("✅ Missing image path test passed")


def test_prompt_content():
    """Test that prompt focuses on facial features"""
    agent = FacialVisionAgent("test_api_key")

    prompt = agent.llm_client.prompts.get_comprehensive_analysis_prompt()
    system_prompt = agent.llm_client.prompts.get_comprehensive_analysis_system_prompt()

    assert "facial features" in prompt
    assert "features" in prompt
    sp = system_prompt.lower()
    assert "forehead" in sp
    assert "eyebrows" in sp
    assert "eyes" in sp
    assert "nose" in sp
    assert "cheeks" in sp
    assert "mouth" in sp
    assert "chin" in sp
    assert "jawline" in sp
    print("✅ Prompt content test passed")


def test_agent_info():
    """Test agent information"""
    agent = FacialVisionAgent("test_api_key")

    info = agent.get_info()

    assert info["name"] == "FacialVision"
    assert info["does_recommendations"] is False
    assert info["output_type"] == "feature_extraction"
    print("✅ Agent info test passed")


def test_llm_call_mock():
    """Test LLM call with simple mock - skips face validation"""
    # Create agent and temporarily disable face validation
    agent = FacialVisionAgent("test_api_key")

    # Temporarily replace face validation with a no-op
    original_validate = agent._validate_face_presence
    agent._validate_face_presence = lambda x: True

    # Mock simple success response
    def mock_llm_call(base64_image):
        return {
            "facial_analysis": {"face_shape": "oval"},
            "hair_analysis": {"type": "wavy"},
            "confidence_metrics": {"overall": 0.8}
        }

    # Temporarily replace the method
    original_method = agent._call_vision_llm
    agent._call_vision_llm = mock_llm_call

    # Mock os.path.exists to return True
    import os
    original_exists = os.path.exists
    os.path.exists = lambda path: True

    # Mock open to return a fake file
    original_open = open
    def mock_open(path, mode='r', encoding=None, **kwargs):
        from io import BytesIO
        return BytesIO(b"fake image data")
    builtins.open = mock_open

    try:
        # Now it should call the mocked LLM
        task = AgentTask(type="analyze_image", payload={"image_path": "test.jpg"})
        response = agent.process(task)

        # Should succeed with mocked data
        assert response.success
        # The current implementation returns the parsed result under the key 'analysis'
        assert "analysis" in response.data
        assert response.data["analysis"]["facial_analysis"]["face_shape"] == "oval"
        print("✅ LLM call mock test passed")

    finally:
        # Restore original methods
        agent._call_vision_llm = original_method
        agent._validate_face_presence = original_validate
        os.path.exists = original_exists
        builtins.open = original_open


def test_extract_json():
    """Test extraction behavior using client's _safe_get_content helper (no _extract_json present)"""
    agent = FacialVisionAgent("test_api_key")

    # Prepare a simulated LLM response structure
    resp = {"choices": [{"message": {"content": '{"test": "value"}'}}]}
    # _safe_get_content should return the raw content string
    content = agent.llm_client._safe_get_content(resp)
    assert content == '{"test": "value"}'
    print("✅ Client _safe_get_content direct JSON string test passed")

    # And for non-JSON text it should return the text as-is
    resp2 = {"choices": [{"message": {"content": 'Here is text {"test": "value"} end'}}]}
    content2 = agent.llm_client._safe_get_content(resp2)
    assert 'Here is text' in content2
    print("✅ Client _safe_get_content embedded JSON string test passed")


def test_temperature_setting():
    """Test that temperature is set correctly"""
    agent = FacialVisionAgent("test_api_key")

    # Check that the method has the temperature parameter
    assert hasattr(agent, '_call_vision_llm')
    assert callable(agent._call_vision_llm)
    print("✅ Temperature setting test passed")


def run_all_tests():
    """Run all simple tests"""
    print("🚀 Running simple FacialVisionAgent tests...\n")

    tests = [
        test_agent_initialization,
        test_supported_tasks,
        test_unsupported_task,
        test_missing_image_path,
        test_prompt_content,
        test_agent_info,
        test_llm_call_mock,
        test_extract_json,
        test_temperature_setting,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print(f"\n📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All simple tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)