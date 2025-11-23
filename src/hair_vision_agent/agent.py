from agent_core_framework import BaseAgent, AgentTask, AgentResponse
from typing import Dict, Any
import base64
import requests
import json
import os


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
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")

    def process(self, task: AgentTask) -> AgentResponse:
        """
        Process analysis tasks. Only performs visual analysis, not recommendations.
        """
        try:
            task_type = task.type
            payload = task.payload

            if task_type == "analyze_image":
                return self._analyze_image(payload)
            elif task_type == "detect_facial_features":
                return self._detect_facial_features(payload)
            elif task_type == "analyze_hair_characteristics":
                return self._analyze_hair_characteristics(payload)
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

    def _analyze_image(self, payload: Dict[str, Any]) -> AgentResponse:
        """
        Comprehensive image analysis - extracts hair and facial features.
        Returns raw analysis data without style recommendations.
        """
        image_path = payload.get('image_path')

        if not image_path:
            return AgentResponse(
                success=False,
                error="Image path is required for analysis",
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

            # Build analysis prompt (focused on extraction, not recommendations)
            prompt = self._build_analysis_prompt()

            # Call vision LLM
            analysis_result = self._call_vision_llm(base64_image, prompt)

            return AgentResponse(
                success=True,
                data={
                    "detected_features": analysis_result,
                    "analysis_confidence": analysis_result.get("confidence_metrics", {}).get("overall", 0.7),
                    "image_processed": True
                },
                agent_name=self.name
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Image analysis failed: {str(e)}",
                agent_name=self.name
            )

    def _detect_facial_features(self, payload: Dict[str, Any]) -> AgentResponse:
        """
        Specifically detect facial features and shape.
        """
        image_path = payload.get('image_path')

        if not image_path:
            return AgentResponse(
                success=False,
                error="Image path is required for facial feature detection",
                agent_name=self.name
            )

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            prompt = """
            Analyze this person's facial features and determine:
            1. Face shape (oval, round, square, heart, oblong)
            2. Key facial proportions
            3. Prominent features (jawline, cheekbones, forehead)

            Return as JSON:
            {
                "face_shape": "detected shape",
                "facial_proportions": {
                    "face_width_to_height_ratio": 0.0,
                    "jaw_strength": "soft/medium/strong",
                    "forehead_height": "low/medium/high"
                },
                "prominent_features": ["feature1", "feature2"],
                "detection_confidence": 0.0-1.0
            }
            """

            result = self._call_vision_llm(base64_image, prompt)

            return AgentResponse(
                success=True,
                data={
                    "facial_analysis": result,
                    "features_detected": True
                },
                agent_name=self.name
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Facial feature detection failed: {str(e)}",
                agent_name=self.name
            )

    def _analyze_hair_characteristics(self, payload: Dict[str, Any]) -> AgentResponse:
        """
        Specifically analyze hair characteristics.
        """
        image_path = payload.get('image_path')

        if not image_path:
            return AgentResponse(
                success=False,
                error="Image path is required for hair analysis",
                agent_name=self.name
            )

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            prompt = """
            Analyze this person's hair characteristics:
            1. Hair type (straight, wavy, curly, coily)
            2. Hair length (short, medium, long)
            3. Hair color (natural description)
            4. Hair density (thin, medium, thick)
            5. Current style (if visible)

            Return as JSON:
            {
                "hair_type": "type",
                "hair_length": "length", 
                "hair_color": "color description",
                "hair_density": "density",
                "current_style": "description if visible",
                "analysis_confidence": 0.0-1.0
            }
            """

            result = self._call_vision_llm(base64_image, prompt)

            return AgentResponse(
                success=True,
                data={
                    "hair_analysis": result,
                    "hair_characteristics_detected": True
                },
                agent_name=self.name
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Hair analysis failed: {str(e)}",
                agent_name=self.name
            )

    def _build_analysis_prompt(self) -> str:
        """
        Build prompt focused on feature extraction, not recommendations.
        """
        return """
        Analyze this person's photo and extract visual characteristics.

        Focus on EXTRACTION only - do NOT make style recommendations.

        Extract and return:

        FACIAL FEATURES:
        - Face shape (oval, round, square, heart, oblong)
        - Facial proportions (width/height ratio, jawline, forehead)
        - Prominent facial features

        HAIR CHARACTERISTICS:
        - Hair type (straight, wavy, curly, coily)
        - Hair length (short, medium, long) 
        - Hair color (natural description)
        - Hair density (thin, medium, thick)
        - Current hair condition

        Return as JSON:
        {
            "facial_analysis": {
                "face_shape": "shape",
                "facial_proportions": {
                    "width_height_ratio": 0.0,
                    "jawline_strength": "soft/medium/strong",
                    "forehead_height": "low/medium/high"
                },
                "prominent_features": ["list", "of", "features"]
            },
            "hair_analysis": {
                "type": "hair type",
                "length": "hair length",
                "color": "color description", 
                "density": "density",
                "condition": "healthy/dry/etc"
            },
            "confidence_metrics": {
                "face_detection": 0.0-1.0,
                "hair_analysis": 0.0-1.0,
                "overall": 0.0-1.0
            }
        }

        IMPORTANT: Only return extracted features. No recommendations.
        """

    def _call_vision_llm(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """
        Call vision-capable LLM for analysis.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.1  # Low temperature for consistent analysis
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to structured analysis
                return self._get_fallback_analysis()

        except Exception as e:
            print(f"Vision LLM call failed: {e}")
            return self._get_fallback_analysis()

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