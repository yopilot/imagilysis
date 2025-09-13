import numpy as np
import logging
from .enhanced_facial_emotion_analysis import EnhancedFacialEmotionAnalysis
from .facial_emotion_analysis import FacialEmotionAnalysis  # Keep as fallback
from .scene_emotion_analysis import SceneEmotionAnalysis
from .gemini_context_analyzer import GeminiContextAnalyzer

class IntelligentEmotionAnalysis:
    """
    Enhanced intelligent emotion analysis with GPU acceleration and Gemini integration.
    """
    
    def __init__(self, gemini_api_key=None):
        self.logger = logging.getLogger(__name__)
        
        # Try enhanced facial analysis first, fallback to basic if needed
        try:
            self.facial_analyzer = EnhancedFacialEmotionAnalysis()
            self.enhanced_mode = True
            self.logger.info("Using Enhanced Facial Emotion Analysis with GPU support")
        except Exception as e:
            self.logger.warning(f"Enhanced analysis failed, falling back to basic: {e}")
            self.facial_analyzer = FacialEmotionAnalysis()
            self.enhanced_mode = False
        
        self.scene_analyzer = SceneEmotionAnalysis()
        
        # Initialize Gemini for advanced context analysis
        self.gemini_analyzer = GeminiContextAnalyzer(gemini_api_key) if gemini_api_key else None
        if self.gemini_analyzer:
            self.logger.info("Gemini context analyzer enabled")
        else:
            self.logger.info("Gemini API not available - using heuristic correlation analysis")
    
    def analyze(self, image_path):
        """
        Enhanced intelligent analysis with improved accuracy and context correlation.
        """
        try:
            self.logger.info(f"Starting enhanced intelligent analysis for: {image_path}")
            
            # Step 1: Enhanced face detection and emotion analysis
            facial_result = self.facial_analyzer.analyze(image_path)
            num_faces = facial_result.get('number_of_faces', 0)
            has_emotions = bool(facial_result.get('emotion_distribution'))
            
            if num_faces > 0 and has_emotions:
                self.logger.info(f"Detected {num_faces} face(s) - using enhanced facial emotion analysis")
                
                # Get scene context for correlation
                try:
                    scene_result = self.scene_analyzer.analyze(image_path)
                    
                    # Enhanced analysis with Gemini correlation
                    if self.gemini_analyzer and self.enhanced_mode:
                        try:
                            # Correlate emotions with scene context using Gemini
                            correlation = self.gemini_analyzer.analyze_emotion_scene_correlation(
                                image_path, 
                                facial_result.get('emotion_distribution', {}),
                                scene_result
                            )
                            
                            # Validate emotion-scene match
                            validation = self.gemini_analyzer.validate_scene_emotion_match(
                                scene_result.get('scene_type', 'unknown'),
                                facial_result.get('emotion_distribution', {})
                            )
                            
                            # Add correlation analysis to results
                            facial_result['gemini_correlation'] = correlation
                            facial_result['scene_validation'] = validation
                            
                            self.logger.info(f"Gemini correlation score: {validation.get('match_score', 'N/A')}")
                            
                        except Exception as e:
                            self.logger.warning(f"Gemini analysis failed: {e}")
                    
                    # Assess correlation using heuristics as backup
                    correlation_assessment = self._assess_emotion_scene_correlation(
                        facial_result.get('emotion_spectrum', {}),
                        scene_result
                    )
                    facial_result['correlation_assessment'] = correlation_assessment
                    
                    return self._combine_face_and_scene_enhanced(facial_result, scene_result)
                    
                except Exception as e:
                    self.logger.warning(f"Scene analysis failed: {e}")
                    # Fall back to just facial analysis
                    facial_result['analysis_type'] = 'face_only'
                    return facial_result
            else:
                self.logger.info("No faces detected - using scene-only analysis")
                try:
                    scene_result = self.scene_analyzer.analyze(image_path)
                    return self._format_scene_only_enhanced(scene_result)
                except Exception as e:
                    return self._create_fallback_result(str(e))
                    
        except Exception as e:
            self.logger.error(f"Enhanced intelligent emotion analysis error: {e}")
            return self._create_fallback_result(str(e))

    def _assess_emotion_scene_correlation(self, emotion_spectrum, scene_results):
        """
        Assess correlation between detected emotions and scene context.
        """
        try:
            scene_type = scene_results.get('scene_type', 'unknown')
            positivity_score = emotion_spectrum.get('positivity_score', 0)
            dominant_emotion = emotion_spectrum.get('dominant_emotion', 'neutral')
            
            # Scene-emotion correlation mapping (improved)
            positive_scenes = [
                'bright_cheerful', 'sunset_sunrise', 'nature_peaceful', 
                'party_celebration', 'beach_vacation', 'garden_flowers',
                'sunny_outdoor', 'festive_occasion'
            ]
            
            negative_scenes = [
                'dark_moody', 'stormy_weather', 'urban_gritty', 
                'abandoned_lonely', 'rainy_day', 'foggy_mysterious',
                'cemetery_somber', 'industrial_harsh'
            ]
            
            neutral_scenes = [
                'indoor_casual', 'office_professional', 'home_comfortable',
                'street_everyday', 'transport_travel'
            ]
            
            correlation_score = 50  # Default neutral
            correlation_status = "neutral"
            explanation = ""
            
            # Assess correlation
            if scene_type in positive_scenes:
                if positivity_score > 0.6 and dominant_emotion in ['happiness', 'surprise']:
                    correlation_score = 90
                    correlation_status = "excellent_match"
                    explanation = f"Facial emotions ({dominant_emotion}, {positivity_score:.1%} positivity) perfectly match the {scene_type} scene context"
                elif positivity_score > 0.3:
                    correlation_score = 70
                    correlation_status = "good_match" 
                    explanation = f"Facial emotions moderately align with {scene_type} scene"
                else:
                    correlation_score = 30
                    correlation_status = "poor_match"
                    explanation = f"Low positivity ({positivity_score:.1%}) doesn't match cheerful {scene_type} scene - possible forced smile or contextual mismatch"
                    
            elif scene_type in negative_scenes:
                if positivity_score < 0.4 and dominant_emotion in ['sadness', 'fear', 'anger']:
                    correlation_score = 85
                    correlation_status = "excellent_match"
                    explanation = f"Facial emotions appropriately reflect the {scene_type} scene mood"
                elif positivity_score < 0.6:
                    correlation_score = 65
                    correlation_status = "acceptable_match"
                    explanation = f"Emotions reasonably align with {scene_type} context"
                else:
                    correlation_score = 25
                    correlation_status = "mismatch"
                    explanation = f"High positivity ({positivity_score:.1%}) seems inconsistent with {scene_type} scene"
                    
            else:  # Neutral scenes
                correlation_score = 75  # Most emotions acceptable in neutral contexts
                correlation_status = "context_appropriate"
                explanation = f"Emotions are contextually appropriate for {scene_type} setting"
            
            # Provide recommendations for mismatches
            recommendations = []
            if correlation_score < 50:
                if scene_type in positive_scenes and positivity_score < 0.4:
                    recommendations.append("Consider if the photo captured a candid moment, forced expression, or if scene classification needs adjustment")
                    recommendations.append("Low facial positivity in bright scene may indicate authentic candid emotion vs posed happiness")
                elif scene_type in negative_scenes and positivity_score > 0.6:
                    recommendations.append("High positivity in moody scene may indicate resilience, defiance, or scene misclassification")
            
            return {
                "correlation_score": correlation_score,
                "correlation_status": correlation_status,
                "explanation": explanation,
                "recommendations": recommendations,
                "scene_type": scene_type,
                "facial_positivity": positivity_score,
                "dominant_emotion": dominant_emotion
            }
            
        except Exception as e:
            self.logger.error(f"Correlation assessment error: {e}")
            return {
                "correlation_score": 50,
                "correlation_status": "assessment_failed",
                "explanation": "Unable to assess emotion-scene correlation",
                "recommendations": [],
                "error": str(e)
            }

    def _combine_face_and_scene_enhanced(self, facial_result, scene_result):
        """
        Enhanced combination of facial emotion analysis with scene context.
        """
        result = facial_result.copy()
        result['analysis_type'] = 'face_enhanced_with_scene'
        
        # Add enhanced scene context
        scene_emotion = scene_result.get('emotion_spectrum', {})
        scene_type = scene_result.get('scene_type', 'unknown')
        
        # Enhanced emotion spectrum with scene context
        if 'emotion_spectrum' in result:
            original_spectrum = result['emotion_spectrum'].copy()
            
            # Intelligent scene-face correlation
            scene_positivity = scene_emotion.get('positivity_score', 0.5)
            face_positivity = original_spectrum.get('positivity_score', 0.5)
            
            # Ensure we have valid positivity scores
            if face_positivity == 0:
                face_positivity = 0.5  # Default neutral
            if scene_positivity == 0:
                scene_positivity = 0.5  # Default neutral
            
            # Weighted combination based on correlation strength
            correlation_assessment = result.get('correlation_assessment', {})
            correlation_score = correlation_assessment.get('correlation_score', 50) / 100
            
            # Higher correlation = more scene influence
            scene_weight = 0.2 + (correlation_score * 0.3)  # 0.2-0.5 scene influence
            face_weight = 1 - scene_weight
            
            combined_positivity = (face_positivity * face_weight) + (scene_positivity * scene_weight)
            
            result['emotion_spectrum']['combined_positivity'] = round(combined_positivity, 3)
            result['emotion_spectrum']['context_enhanced_positivity'] = round(combined_positivity, 3)
            result['emotion_spectrum']['scene_influence_weight'] = round(scene_weight, 3)
            result['emotion_spectrum']['correlation_strength'] = correlation_score
        
        # Enhanced scene context information with proper contextual match
        correlation_assessment = result.get('correlation_assessment', {})
        contextual_match = self._determine_contextual_match(correlation_assessment)
        
        result['scene_context'] = {
            'scene_type': scene_type,
            'scene_emotions': scene_emotion.get('scene_emotions', []),
            'scene_confidence': scene_result.get('scene_confidence', 0),
            'color_analysis': scene_result.get('color_analysis', {}),
            'lighting_mood': scene_result.get('lighting_mood', 'unknown'),
            'contextual_match_score': correlation_assessment.get('correlation_score', 50),
            'contextual_match': contextual_match
        }
        
        return result

    def _determine_contextual_match(self, correlation_assessment):
        """
        Determine contextual match description based on correlation assessment.
        """
        correlation_score = correlation_assessment.get('correlation_score', 50)
        correlation_status = correlation_assessment.get('correlation_status', 'unknown')
        
        if correlation_score >= 85:
            return "highly_congruent"
        elif correlation_score >= 70:
            return "moderately_congruent"
        elif correlation_score >= 50:
            return "somewhat_congruent"
        elif correlation_score >= 30:
            return "weakly_congruent"
        else:
            return "incongruent"

    def _format_scene_only_enhanced(self, scene_result):
        """
        Enhanced formatting for scene-only analysis with better emotion mapping.
        """
        scene_emotion = scene_result.get('emotion_spectrum', {})
        scene_emotions = scene_emotion.get('scene_emotions', [])
        scene_type = scene_result.get('scene_type', 'unknown')
        
        # Enhanced scene-to-emotion mapping
        emotion_distribution = self._map_scene_to_emotions(scene_emotions, scene_type)
        
        # Calculate enhanced emotion spectrum
        emotion_spectrum = self._calculate_scene_emotion_spectrum(emotion_distribution, scene_result)
        
        return {
            'analysis_type': 'scene_only',
            'number_of_faces': 0,
            'emotion_distribution': emotion_distribution,
            'emotion_spectrum': emotion_spectrum,
            'scene_details': {
                'scene_type': scene_type,
                'scene_confidence': scene_result.get('scene_confidence', 0),
                'detected_emotions': scene_emotions,
                'color_mood': scene_result.get('color_mood', 'unknown'),
                'lighting_assessment': scene_result.get('lighting_mood', 'unknown')
            }
        }

    def _map_scene_to_emotions(self, scene_emotions, scene_type):
        """
        Enhanced mapping of scene emotions to standard emotion categories.
        """
        emotion_distribution = {
            'happiness': 0.0, 'sadness': 0.0, 'fear': 0.0,
            'anger': 0.0, 'disgust': 0.0, 'surprise': 0.0, 'neutral': 0.1
        }
        
        # Enhanced emotion mapping
        emotion_mapping = {
            'cheerful': {'happiness': 0.8, 'surprise': 0.2},
            'joyful': {'happiness': 0.9, 'surprise': 0.1},
            'bright': {'happiness': 0.6, 'surprise': 0.3, 'neutral': 0.1},
            'uplifting': {'happiness': 0.7, 'surprise': 0.2, 'neutral': 0.1},
            'peaceful': {'neutral': 0.7, 'happiness': 0.3},
            'calm': {'neutral': 0.8, 'happiness': 0.2},
            'serene': {'neutral': 0.6, 'happiness': 0.4},
            'natural': {'neutral': 0.9, 'happiness': 0.1},
            'romantic': {'happiness': 0.6, 'surprise': 0.2, 'neutral': 0.2},
            'inspiring': {'happiness': 0.5, 'surprise': 0.4, 'neutral': 0.1},
            'melancholic': {'sadness': 0.7, 'neutral': 0.3},
            'somber': {'sadness': 0.8, 'fear': 0.1, 'neutral': 0.1},
            'moody': {'sadness': 0.4, 'anger': 0.3, 'fear': 0.2, 'neutral': 0.1},
            'dramatic': {'anger': 0.4, 'fear': 0.3, 'surprise': 0.2, 'neutral': 0.1},
            'mysterious': {'fear': 0.5, 'surprise': 0.3, 'neutral': 0.2},
            'eerie': {'fear': 0.7, 'disgust': 0.2, 'neutral': 0.1},
            'spooky': {'fear': 0.8, 'surprise': 0.1, 'disgust': 0.1}
        }
        
        # Apply scene type bonuses
        scene_bonuses = {
            'bright_cheerful': {'happiness': 0.3, 'surprise': 0.2},
            'sunset_sunrise': {'happiness': 0.2, 'neutral': 0.2},
            'nature_peaceful': {'neutral': 0.3, 'happiness': 0.1},
            'dark_moody': {'sadness': 0.2, 'fear': 0.1},
            'stormy_weather': {'fear': 0.3, 'anger': 0.2}
        }
        
        # Map scene emotions to standard emotions
        total_weight = 0
        for scene_emotion in scene_emotions:
            scene_lower = scene_emotion.lower()
            if scene_lower in emotion_mapping:
                weight = 1.0
                for emotion, score in emotion_mapping[scene_lower].items():
                    emotion_distribution[emotion] += score * weight
                total_weight += weight
        
        # Apply scene type bonuses
        if scene_type in scene_bonuses:
            for emotion, bonus in scene_bonuses[scene_type].items():
                emotion_distribution[emotion] += bonus
                total_weight += bonus
        
        # Normalize
        if total_weight > 0:
            for emotion in emotion_distribution:
                emotion_distribution[emotion] /= total_weight
        
        # Ensure minimum neutral emotion
        if sum(emotion_distribution.values()) < 0.1:
            emotion_distribution['neutral'] = 1.0
        
        return emotion_distribution

    def _calculate_scene_emotion_spectrum(self, emotion_distribution, scene_result):
        """
        Calculate emotion spectrum for scene-only analysis.
        """
        positivity_weights = {
            'happiness': 1.0, 'surprise': 0.7, 'neutral': 0.0,
            'sadness': -0.9, 'anger': -1.0, 'fear': -0.8, 'disgust': -0.95
        }
        
        enthusiasm_weights = {
            'happiness': 0.95, 'surprise': 1.0, 'anger': 0.2,
            'neutral': 0.0, 'sadness': -0.6, 'fear': -0.4, 'disgust': -0.3
        }
        
        # Calculate scores
        positivity_score = sum(emotion_distribution[emotion] * positivity_weights.get(emotion, 0)
                              for emotion in emotion_distribution)
        positivity_score = max(0, min(1, (positivity_score + 1) / 2))
        
        enthusiasm_score = sum(emotion_distribution[emotion] * enthusiasm_weights.get(emotion, 0)
                              for emotion in emotion_distribution)
        enthusiasm_score = max(0, min(1, (enthusiasm_score + 1) / 2))
        
        emotional_intensity = 1 - emotion_distribution.get('neutral', 0)
        dominant_emotion = max(emotion_distribution, key=emotion_distribution.get)
        
        # Core emotions for chart
        core_emotions = {
            emotion: emotion_distribution.get(emotion, 0.0)
            for emotion in ['happiness', 'sadness', 'fear', 'anger', 'disgust', 'surprise']
        }
        
        return {
            'positivity_score': round(positivity_score, 3),
            'negativity_score': round(1 - positivity_score, 3),
            'enthusiasm_score': round(enthusiasm_score, 3),
            'emotional_intensity': round(emotional_intensity, 3),
            'dominant_emotion': dominant_emotion,
            'core_emotions': {k: round(v, 3) for k, v in core_emotions.items()},
            'scene_emotions': scene_result.get('emotion_spectrum', {}).get('scene_emotions', [])
        }

    def _create_fallback_result(self, error_message):
        """
        Create a fallback result when analysis fails.
        """
        return {
            'analysis_type': 'failed',
            'error': error_message,
            'number_of_faces': 0,
            'emotion_distribution': {'neutral': 1.0},
            'emotion_spectrum': {
                'positivity_score': 0.5,
                'negativity_score': 0.5,
                'enthusiasm_score': 0.0,
                'emotional_intensity': 0.0,
                'dominant_emotion': 'neutral',
                'core_emotions': {
                    'happiness': 0.0, 'sadness': 0.0, 'fear': 0.0,
                    'anger': 0.0, 'disgust': 0.0, 'surprise': 0.0
                }
            }
        }