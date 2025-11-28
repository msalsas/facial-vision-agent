from agent_core_framework import BaseAgent, AgentTask, AgentResponse
from typing import Dict, Any, Optional
import os
import base64
import re
import json
from .llm_client import VisionLLMClient
from .prompts import AnalysisPrompts
from .utils import ImageUtils


class HairVisionAgent(BaseAgent):
    """
    Specialized agent for analyzing hair and facial features from images.
    Only performs visual analysis - does NOT make style recommendations.
    """

    def __init__(self, openrouter_api_key: str = None):
        super().__init__("HairVision", "1.0.0")
        self.supported_tasks = [
            "analyze_image",
            "detect_facial_features",
            "analyze_hair_characteristics"
        ]

        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")

        # Initialize components
        self.llm_client = VisionLLMClient(self.api_key)
        self.prompts = AnalysisPrompts()
        self.image_utils = ImageUtils()

    def process(self, task: AgentTask) -> AgentResponse:
        """
        Process analysis tasks. Only performs visual analysis.
        """
        try:
            task_type = task.type
            payload = task.payload

            if task_type in ["analyze_image", "detect_facial_features", "analyze_hair_characteristics"]:
                return self._analyze_image_comprehensive(payload, task_type)
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unsupported task type: {task_type}",
                    agent_name=self.name
                )

        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Vision analysis error: {str(e)}",
                agent_name=self.name
            )

    def _build_analysis_prompt(self) -> str:
        """
        Build prompt focused on feature extraction.
        """
        return self.prompts.get_comprehensive_analysis_prompt()

    def _call_vision_llm(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """
        Call vision-capable LLM for analysis.
        """
        return self.llm_client.call_vision_llm(base64_image, prompt)

    def _validate_face_presence(self, base64_image: str) -> bool:
        """
        Validate that the image contains at least one face.
        Returns True if face detected, False otherwise.
        """
        return self.llm_client.validate_face_presence(base64_image)

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """
        Fallback analysis when LLM fails.
        """
        return {
            "facial_analysis": {
                "face_shape": "oval",
                "facial_proportions": {
                    "width_height_ratio": 0.75,
                    "jawline_strength": "medium",
                    "forehead_height": "medium"
                },
                "prominent_features": ["balanced_proportions"]
            },
            "hair_analysis": {
                "type": "straight",
                "length": "medium",
                "color": "brown",
                "density": "medium",
                "condition": "healthy"
            },
            "confidence_metrics": {
                "face_detection": 0.3,
                "hair_analysis": 0.3,
                "overall": 0.3
            }
        }

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text, handling cases where LLM adds extra text.
        """
        # Try to find JSON object in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # If no match or failed, try direct load
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and capabilities.
        """
        base_info = super().get_info()
        base_info.update({
            "capabilities": [
                "facial_feature_detection",
                "hair_characteristic_analysis",
                "image_analysis"
            ],
            "output_type": "feature_extraction",
            "does_recommendations": False  # Explicitly state no recommendations
        })
        return base_info

    def _process_image_with_validation(self, image_path: str, processing_function) -> AgentResponse:
        """
        Helper method to validate face presence and process image.
        """
        if not image_path:
            return AgentResponse(
                success=False,
                error="Image path is required",
                agent_name=self.name
            )

        if not os.path.exists(image_path):
            return AgentResponse(
                success=False,
                error=f"Image file not found: {image_path}",
                agent_name=self.name
            )

        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Validate face presence in the image
            if not self._validate_face_presence(base64_image):
                return AgentResponse(
                    success=False,
                    error="No human face detected in the image.",
                    agent_name=self.name
                )

            # Call the processing function
            return processing_function(base64_image)

        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Image processing failed: {str(e)}",
                agent_name=self.name
            )

    def _analyze_image_comprehensive(self, payload: Dict[str, Any], task_type: str) -> AgentResponse:
        """
        Comprehensive image analysis with single LLM call - optimized for speed.
        Performs full analysis once, then returns appropriate subset based on task type.
        """
        image_path = payload.get('image_path')

        def process_comprehensive_analysis(base64_image: str) -> AgentResponse:
            # Single comprehensive analysis call
            prompt = self._build_analysis_prompt()
            analysis_result = self._call_vision_llm(base64_image, prompt)

            # Return different response formats based on task type
            if task_type == "analyze_image":
                return AgentResponse(
                    success=True,
                    data={
                        "detected_features": analysis_result,
                        "analysis_confidence": analysis_result.get("confidence_metrics", {}).get("overall", 0.7),
                        "image_processed": True
                    },
                    agent_name=self.name
                )
            elif task_type == "detect_facial_features":
                return AgentResponse(
                    success=True,
                    data={
                        "facial_analysis": analysis_result.get("facial_analysis", {}),
                        "features_detected": True
                    },
                    agent_name=self.name
                )
            elif task_type == "analyze_hair_characteristics":
                return AgentResponse(
                    success=True,
                    data={
                        "hair_analysis": analysis_result.get("hair_analysis", {}),
                        "hair_characteristics_detected": True
                    },
                    agent_name=self.name
                )

        return self._process_image_with_validation(image_path, process_comprehensive_analysis)
