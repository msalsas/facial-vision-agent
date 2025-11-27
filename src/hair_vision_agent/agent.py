from agent_core_framework import BaseAgent, AgentTask, AgentResponse
from typing import Dict, Any, Optional
import base64
import requests
import json
import os
import re


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
        return """
        Analyze this person's facial physiognomy and features in detail, including hair characteristics.

        Focus on facial features: face shape, forehead, eyebrows, eyes, nose, cheeks, mouth, chin, jawline.

        Provide DETAILED PROPORTION MEASUREMENTS:
        - Face width to height ratio
        - Forehead height to total face height ratio
        - Eye width to face width ratio
        - Inter-eye distance to face width ratio
        - Nose width to face width ratio
        - Nose height to face height ratio
        - Mouth width to face width ratio
        - Chin width to face width ratio
        - Eye to nose distance ratio
        - Nose to mouth distance ratio

        Return as JSON:
        {
            "facial_analysis": {
                "face_shape": "oval/round/square/heart/oblong",
                "forehead": "high/medium/low",
                "eyebrows": "thick/thin/ arched/straight",
                "eyes": "large/small, round/almond shaped",
                "nose": "straight/curved, large/small",
                "cheeks": "high/prominent/flat",
                "mouth": "full/thin, wide/narrow",
                "chin": "pointed/rounded/square",
                "jawline": "strong/soft/defined",
                "facial_proportions": {
                    "face_width_to_height_ratio": 0.0,
                    "forehead_to_face_height_ratio": 0.0,
                    "eye_width_to_face_width_ratio": 0.0,
                    "inter_eye_to_face_width_ratio": 0.0,
                    "nose_width_to_face_width_ratio": 0.0,
                    "nose_height_to_face_height_ratio": 0.0,
                    "mouth_width_to_face_width_ratio": 0.0,
                    "chin_width_to_face_width_ratio": 0.0,
                    "eye_to_nose_distance_ratio": 0.0,
                    "nose_to_mouth_distance_ratio": 0.0
                },
                "prominent_features": ["feature1", "feature2", "feature3"]
            },
            "hair_analysis": {
                "type": "straight/wavy/curly/coily",
                "length": "short/medium/long",
                "color": "color description",
                "density": "thin/medium/thick",
                "condition": "healthy/dry/damaged"
            },
            "confidence_metrics": {
                "face_detection": 0.0,
                "hair_analysis": 0.0,
                "overall": 0.0
            }
        }
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
            "temperature": 0.3
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON response
            parsed = self._extract_json(content)
            if parsed is not None:
                return parsed
            else:
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

    def _validate_face_presence(self, base64_image: str) -> bool:
        """
        Validate that the image contains at least one face.
        Returns True if face detected, False otherwise.
        """
        prompt = """
        Analyze this image and determine if it contains a human face.
        
        Return only a JSON object:
        {
            "face_detected": true/false,
            "confidence": 0.0-1.0
        }
        
        Be strict: only return true if you can clearly see a human face.
        """

        try:
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
                "max_tokens": 200,
                "temperature": 0.1
            }

            response = requests.post(self.base_url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse validation response
            parsed = self._extract_json(content)
            if parsed and isinstance(parsed, dict):
                face_detected = parsed.get('face_detected', False)
                confidence = parsed.get('confidence', 0.0)

                # Require high confidence for face detection
                return face_detected and confidence > 0.7

            return False

        except Exception as e:
            print(f"Face validation failed: {e}")
            return False

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
