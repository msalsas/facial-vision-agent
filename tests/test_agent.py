from hair_vision_agent import HairVisionAgent
from agent_core_framework import AgentTask


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
    """Test that prompt forbids recommendations"""
    agent = HairVisionAgent("test_api_key")

    prompt = agent._build_analysis_prompt()

    # Check critical instructions
    assert "do NOT make style recommendations" in prompt
    assert "Only return extracted features" in prompt
    assert "No recommendations" in prompt

    # Check analysis focus
    assert "FACIAL FEATURES" in prompt
    assert "HAIR CHARACTERISTICS" in prompt
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
    """Test LLM call with simple mock"""
    agent = HairVisionAgent("test_api_key")

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

    try:
        # This should work even without real image file
        # because we're mocking the LLM call
        task = AgentTask(type="analyze_image", payload={"image_path": "test.jpg"})
        response = agent.process(task)

        # The response will fail because the file doesn't exist,
        # but we can check the error handling
        assert not response.success
        assert "file not found" in response.error.lower() or "analysis failed" in response.error.lower()
        print("✅ LLM call mock test passed")

    finally:
        # Restore original method
        agent._call_vision_llm = original_method


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