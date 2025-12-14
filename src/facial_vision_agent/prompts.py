"""
Prompt templates for facial and hair analysis.
"""


class AnalysisPrompts:
    """Collection of prompt templates for different analysis types."""

    @staticmethod
    def get_face_validation_prompt() -> str:
        return """
        Your task is to analyze the image and determine whether it contains at least one visible human face.
        
        Follow these strict rules:
        
        1. If the image contains a human face — even partially — respond ONLY with the single letter: Y
        2. If the image does NOT contain a human face of any kind, respond ONLY with the single letter: N
        3. Do NOT add explanations, punctuation, extra text, line breaks, or whitespace.
        4. Do NOT output words, sentences, or JSON. Output ONLY a single uppercase letter: Y or N.
        
        Your entire reply must consist of exactly one character: Y or N.
        """

    @staticmethod
    def get_comprehensive_analysis_system_prompt() -> str:
        return """
        You are a highly specialized assistant focused on detailed visual analysis.
        
        Your task is to describe facial features and hair characteristics with precision, neutrality, and objectivity. You must base all statements exclusively on what can be visually inferred from the image. Do not guess personality, emotions, or traits not supported by visible evidence.
        
        Provide a structured, exhaustive textual description that includes:
        
        1. FACIAL SHAPE & STRUCTURE
           - Face shape (general form)
           - Forehead (height, width, slope)
           - Eyebrows (thickness, arch, separation)
           - Eyes (size, shape, spacing)
           - Nose (shape, width, bridge, tip)
           - Cheeks (fullness, prominence)
           - Mouth and lips (shape, thickness, width)
           - Chin (shape)
           - Jawline (definition, angle)
        
        2. FACIAL PROPORTIONS
           - Relative distances between features  
           - Vertical and horizontal balance  
           - Any noticeable symmetry or asymmetry
           - Numerical ratios of several facial dimensions
        
        3. PROMINENT OR DISTINCTIVE FEATURES
           - Any visually striking or defining traits
        
        4. HAIR ANALYSIS
           - Hair type (straight, wavy, curly, coily)
           - Density and volume
           - Length
           - Color and tones
           - Condition (healthy, dry, damaged)
           - Overall shape or styling pattern
        
        STYLE REQUIREMENTS:
        - Use clear, natural language (full sentences or structured paragraphs).
        - Be precise and thorough.
        - Do NOT provide any JSON, lists of values, or categorical labels unless naturally needed.
        - Do NOT include opinions, compliments, or speculation.
        - Focus purely on visual physical description and proportions.
        """

    @staticmethod
    def get_comprehensive_analysis_prompt() -> str:
        return """
        Analyze this person's facial features and hair in detail. Provide a complete, objective visual description following the guidelines in the system prompt.
        """

    @staticmethod
    def get_comprehensive_analysis_retry_prompt() -> str:
        return """
        Extract and RETURN ONLY the single JSON object that was embedded in your previous reply.
        Do NOT add any explanations or extra text. If there is no JSON, return {}.
        """
