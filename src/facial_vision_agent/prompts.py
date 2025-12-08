"""
Prompt templates for facial and hair analysis.
"""


class AnalysisPrompts:
    """Collection of prompt templates for different analysis types."""

    @staticmethod
    def get_face_validation_prompt() -> str:
        return """
        Analyze this image and determine if it contains a human face.
        Respond with either 'Y' or 'N'.
        Return only the letter; do NOT add other explanation.
        """

    @staticmethod
    def get_comprehensive_analysis_system_prompt() -> str:
        return """"
        You are an assistant that MUST reply with a single JSON object only,\n
        and nothing else (no explanation, no markdown/code fences).\n
        The JSON MUST follow this minimal schema:\n
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
        All values must be set. Do not leave any fields empty.
        If you cannot produce valid JSON, return an empty JSON object: {}
        """

    @staticmethod
    def get_comprehensive_analysis_prompt() -> str:
        return """
        Analyze this person's facial physiognomy and features in detail, including hair characteristics.

        Return ONLY a JSON object with the system prompt schema provided.
        """

    @staticmethod
    def get_comprehensive_analysis_retry_prompt() -> str:
        return """
        Extract and RETURN ONLY the single JSON object that was embedded in your previous reply."
        Do NOT add any explanations or extra text. If there is no JSON, return {}.
        """

