import google.generativeai as genai
import base64
import json
import logging
from PIL import Image
import io
import os

class GeminiContextAnalyzer:
    def __init__(self, api_key=None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.model = None
        
        if api_key:
            self._initialize_gemini()
        else:
            self.logger.warning("No Gemini API key provided. Context analysis will be limited.")

    def _initialize_gemini(self):
        """Initialize Gemini 2.5 Flash model."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.logger.info("Gemini 2.5 Flash model initialized successfully!")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None

    def analyze_emotion_scene_correlation(self, image_path, facial_emotions, scene_analysis):
        """
        Use Gemini to analyze correlation between facial emotions and scene context.
        """
        if not self.model:
            return self._fallback_correlation_analysis(facial_emotions, scene_analysis)

        try:
            # Load and prepare image
            image = Image.open(image_path).convert('RGB')
            
            # Create analysis prompt
            prompt = f"""
            Analyze this image and provide a detailed correlation between the detected facial emotions and scene context.

            DETECTED FACIAL EMOTIONS:
            {json.dumps(facial_emotions, indent=2)}

            DETECTED SCENE ANALYSIS:
            {json.dumps(scene_analysis, indent=2)}

            Please provide:
            1. CORRELATION_SCORE (0-100): How well facial emotions match the scene context
            2. CONTEXT_EXPLANATION: Why the emotions match or don't match the scene
            3. ENHANCED_INTERPRETATION: A deeper interpretation combining both analyses
            4. AUTHENTICITY_ASSESSMENT: How natural/authentic the emotions appear in this context
            5. SUGGESTED_CORRECTIONS: If there are mismatches, suggest what might be more accurate

            Format your response as a JSON object with these exact keys.
            """

            response = self.model.generate_content([prompt, image])
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                # If not valid JSON, create structured response from text
                return self._parse_text_response(response.text, facial_emotions, scene_analysis)
                
        except Exception as e:
            self.logger.error(f"Gemini correlation analysis error: {e}")
            return self._fallback_correlation_analysis(facial_emotions, scene_analysis)

    def enhance_emotion_context(self, image_path, basic_analysis):
        """
        Use Gemini to provide enhanced context and deeper emotional insights.
        """
        if not self.model:
            return basic_analysis

        try:
            image = Image.open(image_path).convert('RGB')
            
            prompt = f"""
            Analyze this image for advanced emotional and contextual insights.

            CURRENT ANALYSIS:
            {json.dumps(basic_analysis, indent=2)}

            Please provide enhanced insights including:
            1. EMOTIONAL_SUBTEXT: Subtle emotions not captured by basic analysis
            2. CONTEXTUAL_CLUES: Environmental factors affecting emotions
            3. CULTURAL_CONTEXT: Any cultural or social context visible
            4. TEMPORAL_CONTEXT: Time of day, season, occasion indicators
            5. RELATIONSHIP_DYNAMICS: If multiple people, their relationship dynamics
            6. AUTHENTICITY_INDICATORS: Signs of genuine vs posed emotions
            7. ENHANCED_MOOD_DESCRIPTION: Rich, nuanced mood description

            Return as JSON with these exact keys.
            """

            response = self.model.generate_content([prompt, image])
            
            try:
                enhanced_insights = json.loads(response.text)
                return {**basic_analysis, "enhanced_insights": enhanced_insights}
            except json.JSONDecodeError:
                return {**basic_analysis, "enhanced_insights": {"description": response.text}}
                
        except Exception as e:
            self.logger.error(f"Gemini enhancement error: {e}")
            return basic_analysis

    def validate_scene_emotion_match(self, scene_type, facial_emotions):
        """
        Use Gemini to validate if scene type matches facial emotions.
        """
        if not self.model:
            return self._simple_validation(scene_type, facial_emotions)

        try:
            prompt = f"""
            Evaluate if the scene type "{scene_type}" logically matches these facial emotions:
            {json.dumps(facial_emotions, indent=2)}

            Consider:
            - Do people typically feel these emotions in this type of scene?
            - Are there any obvious mismatches?
            - What would be more typical emotions for this scene?

            Respond with:
            {{
                "match_score": 0-100,
                "is_logical_match": true/false,
                "explanation": "detailed explanation",
                "expected_emotions": ["list", "of", "more", "typical", "emotions"],
                "confidence": 0-100
            }}
            """

            response = self.model.generate_content(prompt)
            
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return self._simple_validation(scene_type, facial_emotions)
                
        except Exception as e:
            self.logger.error(f"Gemini validation error: {e}")
            return self._simple_validation(scene_type, facial_emotions)

    def generate_personality_insights(self, image_path, complete_analysis):
        """
        Use Gemini to generate personality and character insights from the image.
        """
        if not self.model:
            return {"personality_insights": "Gemini API not available"}

        try:
            image = Image.open(image_path).convert('RGB')
            
            prompt = f"""
            Based on this image and the analysis data, provide personality and character insights:

            ANALYSIS DATA:
            {json.dumps(complete_analysis, indent=2)}

            Generate insights about:
            1. PERSONALITY_TRAITS: Observable personality characteristics
            2. EMOTIONAL_INTELLIGENCE: Signs of emotional awareness and expression
            3. SOCIAL_STYLE: How they interact in social situations
            4. CONFIDENCE_LEVEL: Body language and expression confidence indicators
            5. AUTHENTICITY: How genuine the expressions appear
            6. LIFESTYLE_INDICATORS: What the image suggests about lifestyle/values
            7. COMMUNICATION_STYLE: How they likely communicate based on expressions

            Return as JSON with these exact keys and detailed descriptions.
            """

            response = self.model.generate_content([prompt, image])
            
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"personality_insights": response.text}
                
        except Exception as e:
            self.logger.error(f"Gemini personality analysis error: {e}")
            return {"personality_insights": "Analysis unavailable"}

    def _fallback_correlation_analysis(self, facial_emotions, scene_analysis):
        """Fallback analysis when Gemini is not available."""
        # Simple heuristic-based correlation
        scene_type = scene_analysis.get('scene_type', 'unknown')
        
        # Basic correlation mapping
        positive_scenes = ['bright_cheerful', 'sunset_sunrise', 'nature_peaceful', 'party_celebration']
        negative_scenes = ['dark_moody', 'stormy_weather', 'urban_gritty', 'abandoned_lonely']
        
        positive_emotions = facial_emotions.get('happiness', 0) + facial_emotions.get('surprise', 0)
        negative_emotions = facial_emotions.get('sadness', 0) + facial_emotions.get('anger', 0) + facial_emotions.get('fear', 0)
        
        correlation_score = 50  # Default neutral
        
        if scene_type in positive_scenes and positive_emotions > negative_emotions:
            correlation_score = 80
        elif scene_type in negative_scenes and negative_emotions > positive_emotions:
            correlation_score = 75
        elif scene_type in positive_scenes and negative_emotions > positive_emotions:
            correlation_score = 30
        elif scene_type in negative_scenes and positive_emotions > negative_emotions:
            correlation_score = 25
        
        return {
            "CORRELATION_SCORE": correlation_score,
            "CONTEXT_EXPLANATION": f"Basic correlation between {scene_type} scene and detected emotions",
            "ENHANCED_INTERPRETATION": "Limited analysis without Gemini API",
            "AUTHENTICITY_ASSESSMENT": "Cannot assess without advanced AI",
            "SUGGESTED_CORRECTIONS": "Gemini API needed for detailed suggestions"
        }

    def _simple_validation(self, scene_type, facial_emotions):
        """Simple validation without Gemini."""
        # Basic logic for scene-emotion matching
        positive_emotions = ['happiness', 'surprise']
        negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
        
        dominant_emotion = max(facial_emotions, key=facial_emotions.get)
        
        expected_match = False
        if 'bright' in scene_type or 'cheerful' in scene_type or 'sunset' in scene_type:
            expected_match = dominant_emotion in positive_emotions
        elif 'dark' in scene_type or 'moody' in scene_type or 'stormy' in scene_type:
            expected_match = dominant_emotion in negative_emotions
        else:
            expected_match = True  # Neutral scenes can have any emotion
        
        return {
            "match_score": 80 if expected_match else 40,
            "is_logical_match": expected_match,
            "explanation": f"Scene type '{scene_type}' with dominant emotion '{dominant_emotion}'",
            "expected_emotions": ["happiness", "neutral"] if 'bright' in scene_type else ["neutral"],
            "confidence": 60
        }

    def _parse_text_response(self, text, facial_emotions, scene_analysis):
        """Parse non-JSON text response from Gemini."""
        try:
            # Extract key information from text response
            return {
                "CORRELATION_SCORE": 75,  # Default reasonable score
                "CONTEXT_EXPLANATION": text[:200] + "..." if len(text) > 200 else text,
                "ENHANCED_INTERPRETATION": "Detailed analysis available in context explanation",
                "AUTHENTICITY_ASSESSMENT": "Refer to context explanation for authenticity details",
                "SUGGESTED_CORRECTIONS": "See enhanced interpretation for suggestions"
            }
        except:
            return self._fallback_correlation_analysis(facial_emotions, scene_analysis)