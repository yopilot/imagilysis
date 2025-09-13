import numpy as np

class VisualSentimentAnalysis:
    def __init__(self):
        pass

    def aggregate_mood(self, color_mood, emotion_results):
        """
        Aggregate overall mood from color and emotion, handling intelligent emotion analysis.
        """
        analysis_type = emotion_results.get('analysis_type', 'unknown')
        
        # Handle different analysis types from intelligent emotion analysis
        if analysis_type == 'face_enhanced_with_scene':
            # Face-based with scene context
            emotion_spectrum = emotion_results.get('emotion_spectrum', {})
            scene_context = emotion_results.get('scene_context', {})
            
            # Use enhanced positivity scores
            pos_score = emotion_spectrum.get('context_enhanced_positivity', 
                                           emotion_spectrum.get('combined_positivity', 
                                           emotion_spectrum.get('positivity_score', 0.5)))
            neg_score = 1 - pos_score
            
            if pos_score > 0.6:
                overall_mood = 'positive'
            elif neg_score > 0.6:
                overall_mood = 'negative'
            else:
                overall_mood = 'neutral'
                
            # Enhanced mood description with scene context
            scene_type = scene_context.get('scene_type', '')
            contextual_match = scene_context.get('contextual_match', '')
            mood_description = f"{overall_mood} (face+scene: {scene_type}, {contextual_match})"
            
        elif analysis_type == 'scene_based':
            # Scene-only analysis
            emotion_spectrum = emotion_results.get('emotion_spectrum', {})
            scene_details = emotion_results.get('scene_details', {})
            
            pos_score = emotion_spectrum.get('positivity_score', 0.5)
            neg_score = emotion_spectrum.get('negativity_score', 0.5)
            
            if pos_score > 0.6:
                overall_mood = 'positive'
            elif neg_score > 0.6:
                overall_mood = 'negative'
            else:
                overall_mood = 'neutral'
                
            scene_type = scene_details.get('scene_type', 'general')
            scene_emotions = emotion_spectrum.get('scene_emotions', [])
            mood_description = f"{overall_mood} (scene: {scene_type}, {', '.join(scene_emotions[:2])})"
            
        elif analysis_type == 'face_only':
            # Traditional face-based analysis
            emotion_dist = emotion_results.get('emotion_distribution', {})
            positive_emotions = ['happy', 'surprised']
            negative_emotions = ['sad', 'angry', 'fear', 'disgust']
            
            pos_score = sum(emotion_dist.get(e, 0) for e in positive_emotions)
            neg_score = sum(emotion_dist.get(e, 0) for e in negative_emotions)
            
            if pos_score > neg_score:
                overall_mood = 'positive'
            elif neg_score > pos_score:
                overall_mood = 'negative'
            else:
                overall_mood = 'neutral'
            
            dominant = emotion_results.get('emotion_spectrum', {}).get('dominant_emotion', 'unknown')
            mood_description = f"{overall_mood} (face-only: {dominant})"
            
        else:
            # Fallback for unknown or error cases
            overall_mood = 'neutral'
            mood_description = f"neutral ({analysis_type})"
        
        # Color sentiment
        color_positive = ['happy', 'energetic', 'calm', 'serene']
        color_sentiment = 'positive' if color_mood in color_positive else 'neutral'
        
        return overall_mood, color_sentiment, mood_description

    def analyze_consistency(self, results_list):
        """
        Analyze consistency across images.
        """
        scores = []
        for res in results_list:
            if 'overall_aesthetic_score' in res:
                scores.append(res['overall_aesthetic_score'])
            elif 'aesthetic_score' in res:
                scores.append(res['aesthetic_score'])
        
        if not scores:
            return 0.5
        variance = np.var(scores)
        consistency = max(0, 1 - variance)  # Normalize
        return consistency

    def detect_curation(self, consistency):
        """
        Detect if images are curated or authentic.
        """
        if consistency > 0.7:
            return 'curated'
        else:
            return 'authentic'

    def analyze(self, color_results, emotion_results, composition_results, other_results=None):
        """
        Full sentiment analysis with enhanced scene emotion support.
        """
        overall_mood, color_sentiment, mood_description = self.aggregate_mood(
            color_results.get('mood', 'neutral'), 
            emotion_results
        )
        
        consistency = self.analyze_consistency([composition_results] + (other_results or []))
        
        curation = self.detect_curation(consistency)
        
        # Add intelligent analysis insights if available
        analysis_insights = {}
        analysis_type = emotion_results.get('analysis_type', 'unknown')
        
        if analysis_type == 'face_enhanced_with_scene':
            scene_context = emotion_results.get('scene_context', {})
            analysis_insights = {
                'analysis_type': 'face_with_scene_context',
                'scene_type': scene_context.get('scene_type', 'unknown'),
                'contextual_match': scene_context.get('contextual_match', 'unknown'),
                'scene_influence': emotion_results.get('emotion_spectrum', {}).get('scene_influence', 0)
            }
        elif analysis_type == 'scene_based':
            scene_details = emotion_results.get('scene_details', {})
            analysis_insights = {
                'analysis_type': 'scene_only',
                'scene_type': scene_details.get('scene_type', 'unknown'),
                'scene_confidence': scene_details.get('scene_confidence', 0)
            }
        elif analysis_type == 'face_only':
            analysis_insights = {
                'analysis_type': 'face_only',
                'faces_detected': emotion_results.get('number_of_faces', 0)
            }
        
        return {
            "overall_mood": overall_mood,
            "mood_description": mood_description,
            "color_sentiment": color_sentiment,
            "consistency_score": consistency,
            "curation_type": curation,
            "analysis_insights": analysis_insights
        }