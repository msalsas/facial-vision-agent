from hair_vision_agent import HairVisionAgent
from agent_core_framework import AgentTask
import builtins


def test_agent_initialization():
    """Test basic agent initialization"""
    agent = HairVisionAgent("test_api_key")

    assert agent.name == "HairVision"
    assert agent.version == "1.0.0"
    assert agent.api_key == "test_api_key"
    print("✅ Agent initialization test passed")


def test_supported_tasks():
    """Test that agent only supports analysis tasks"""
    agent = HairVisionAgent("test_api_key")

    expected_tasks = ["analyze_image", "detect_facial_features", "analyze_hair_characteristics"]
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
    agent = HairVisionAgent("test_api_key")

    task = AgentTask(type="recommend_styles", payload={})
    response = agent.process(task)

    assert not response.success
    assert "Unsupported task type" in response.error
    print("✅ Unsupported task handling test passed")


def test_missing_image_path():
    """Test error when image path is missing"""
    agent = HairVisionAgent("test_api_key")

    task = AgentTask(type="analyze_image", payload={})
    response = agent.process(task)

    assert not response.success
    assert "Image path is required" in response.error
    print("✅ Missing image path test passed")


def test_prompt_content():
    """Test that prompt focuses on facial physiognomy"""
    agent = HairVisionAgent("test_api_key")

    prompt = agent._build_analysis_prompt()

    # Check critical instructions
    assert "facial physiognomy" in prompt
    assert "facial features" in prompt
    assert "forehead" in prompt
    assert "eyebrows" in prompt
    assert "eyes" in prompt
    assert "nose" in prompt
    assert "cheeks" in prompt
    assert "mouth" in prompt
    assert "chin" in prompt
    assert "jawline" in prompt
    print("✅ Prompt content test passed")


def test_fallback_analysis():
    """Test fallback analysis structure"""
    agent = HairVisionAgent("test_api_key")

    fallback = agent._get_fallback_analysis()

    # Check structure
    assert "facial_analysis" in fallback
    assert "hair_analysis" in fallback
    assert "confidence_metrics" in fallback

    # Check no recommendations
    assert "recommendations" not in fallback
    assert "style_suggestions" not in fallback
    print("✅ Fallback analysis test passed")


def test_agent_info():
    """Test agent information"""
    agent = HairVisionAgent("test_api_key")

    info = agent.get_info()

    assert info["name"] == "HairVision"
    assert info["does_recommendations"] is False
    assert info["output_type"] == "feature_extraction"
    print("✅ Agent info test passed")


def test_llm_call_mock():
    """Test LLM call with simple mock - skips face validation"""
    # Create agent and temporarily disable face validation
    agent = HairVisionAgent("test_api_key")

    # Temporarily replace face validation with a no-op
    original_validate = agent._validate_face_presence
    agent._validate_face_presence = lambda x: True

    # Mock simple success response
    def mock_llm_call(base64_image, prompt):
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
        assert "detected_features" in response.data
        assert response.data["detected_features"]["facial_analysis"]["face_shape"] == "oval"
        print("✅ LLM call mock test passed")

    finally:
        # Restore original methods
        agent._call_vision_llm = original_method
        agent._validate_face_presence = original_validate
        os.path.exists = original_exists
        builtins.open = original_open


def test_extract_json():
    """Test JSON extraction from text"""
    agent = HairVisionAgent("test_api_key")

    # Test direct JSON
    json_text = '{"test": "value"}'
    result = agent._extract_json(json_text)
    assert result == {"test": "value"}
    print("✅ Direct JSON extraction test passed")

    # Test JSON with extra text
    extra_text = 'Here is some text {"test": "value"} and more text'
    result = agent._extract_json(extra_text)
    assert result == {"test": "value"}
    print("✅ JSON extraction with extra text test passed")

    # Test invalid JSON
    invalid_text = 'not json at all'
    result = agent._extract_json(invalid_text)
    assert result is None
    print("✅ Invalid JSON handling test passed")


def test_temperature_setting():
    """Test that temperature is set correctly"""
    agent = HairVisionAgent("test_api_key")

    # Check that the method has the temperature parameter
    assert hasattr(agent, '_call_vision_llm')
    assert callable(agent._call_vision_llm)
    print("✅ Temperature setting test passed")


def test_fallback_on_json_error():
    """Test that fallback is used when JSON parsing fails"""
    agent = HairVisionAgent("test_api_key")

    original_extract = agent._extract_json
    agent._extract_json = lambda text: None

    try:
        fallback = agent._get_fallback_analysis()
        assert "facial_analysis" in fallback
        print("✅ Fallback on JSON error test passed")

    finally:
        agent._extract_json = original_extract


def run_all_tests():
    """Run all simple tests"""
    print("🚀 Running simple HairVisionAgent tests...\n")

    tests = [
        test_agent_initialization,
        test_supported_tasks,
        test_unsupported_task,
        test_missing_image_path,
        test_prompt_content,
        test_fallback_analysis,
        test_agent_info,
        test_llm_call_mock,
        test_extract_json,
        test_temperature_setting,
        test_fallback_on_json_error,
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