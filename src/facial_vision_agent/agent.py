from agent_core_framework import BaseAgent, AgentTask, AgentResponse
from typing import Dict, Any, Optional
import os
import base64
import re
import json
from .llm_client import VisionLLMClient
from .utils import ImageUtils
import logging


class FacialVisionAgent(BaseAgent):
    """
    Specialized agent for analyzing facial features and hair from images.
    Only performs visual analysis - does NOT make style recommendations.
    """

    def __init__(self, openrouter_api_key: str = None):
        super().__init__("FacialVision", "1.0.0")
        self.supported_tasks = [
            "analyze_image",
            "detect_facial_features",
            "analyze_hair_characteristics",
        ]

        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable."
            )

        # Initialize components
        self.llm_client = VisionLLMClient(self.api_key)
        self.image_utils = ImageUtils()
        self.logger = logging.getLogger(__name__)

    def process(self, task: AgentTask) -> AgentResponse:
        """Process analysis tasks. Only performs visual analysis."""
        try:
            task_type = task.type
            payload = task.payload

            if task_type in self.supported_tasks:
                return self._analyze_image_comprehensive(payload, task_type)

            return AgentResponse(
                success=False,
                error=f"Unsupported task type: {task_type}",
                agent_name=self.name,
            )

        except Exception as e:
            return AgentResponse(
                success=False, error=f"Vision analysis error: {str(e)}", agent_name=self.name
            )

    def _call_vision_llm(self, base64_image: str) -> Dict[str, Any]:
        return self.llm_client.call_vision_llm(base64_image)

    def _validate_face_presence(self, base64_image: str) -> bool:
        return self.llm_client.validate_face_presence(base64_image)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        # Try to find JSON object in the text
        json_match = re.search(r'{.*}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def get_info(self) -> Dict[str, Any]:
        base_info = super().get_info()
        base_info.update(
            {
                "capabilities": [
                    "facial_feature_detection",
                    "hair_characteristic_analysis",
                    "image_analysis",
                ],
                "output_type": "feature_extraction",
                "does_recommendations": False,
            }
        )
        return base_info

    def _process_image_with_validation(self, image_path: Optional[str] = None, base64_image: Optional[str] = None, processing_function=None) -> AgentResponse:
        if not image_path and not base64_image:
            return AgentResponse(success=False, error="Image path is required", agent_name=self.name)

        try:
            if not base64_image:
                if not os.path.exists(image_path):
                    return AgentResponse(success=False, error=f"Image file not found: {image_path}", agent_name=self.name)

                base64_image = self.image_utils.encode_image_to_base64(image_path)
                if not base64_image:
                    return AgentResponse(success=False, error=f"Failed to encode image: {image_path}", agent_name=self.name)

            if not self._validate_face_presence(base64_image):
                return AgentResponse(
                    success=False, error="No human face detected in the image.", agent_name=self.name
                )

            return processing_function(base64_image)

        except Exception as e:
            return AgentResponse(success=False, error=f"Image processing failed: {str(e)}", agent_name=self.name)

    def _analyze_image_comprehensive(self, payload: Dict[str, Any], task_type: str) -> AgentResponse:
        # Support either 'image_path' or 'base64_image' in the payload
        image_path = payload.get("image_path")
        base64_image = payload.get("base64_image")

        def process_comprehensive_analysis(base64_image: str) -> AgentResponse:
            try:
                analysis_result = self._call_vision_llm(base64_image)
            except Exception as e:
                # Surface a clear error to the caller when LLM analysis fails
                return AgentResponse(success=False, error=f"Vision LLM analysis failed: {str(e)}", agent_name=self.name)

            if task_type == "analyze_image":
                return AgentResponse(
                    success=True,
                    data={
                        "detected_features": analysis_result,
                        "analysis_confidence": analysis_result.get("confidence_metrics", {}).get("overall", 0.7),
                        "image_processed": True,
                    },
                    agent_name=self.name,
                )
            elif task_type == "detect_facial_features":
                return AgentResponse(
                    success=True,
                    data={"facial_analysis": analysis_result.get("facial_analysis", {}), "features_detected": True},
                    agent_name=self.name,
                )
            elif task_type == "analyze_hair_characteristics":
                return AgentResponse(
                    success=True,
                    data={"hair_analysis": analysis_result.get("hair_analysis", {}), "hair_characteristics_detected": True},
                    agent_name=self.name,
                )

        return self._process_image_with_validation(image_path=image_path, base64_image=base64_image, processing_function=process_comprehensive_analysis)
