"""
Prompt templates for facial and hair analysis.
"""


class AnalysisPrompts:
    """Collection of prompt templates for different analysis types."""

    @staticmethod
    def get_comprehensive_analysis_prompt() -> str:
        """
        Get the comprehensive facial and hair analysis prompt.
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

    @staticmethod
    def get_face_validation_prompt() -> str:
        """
        Get the face validation prompt.
        """
        return """
        Analyze this image and determine if it contains a human face.
        
        Return only a JSON object:
        {
            "face_detected": true/false,
            "confidence": 0.0-1.0
        }
        
        Be strict: only return true if you can clearly see a human face.
        """
