from .color_analysis import ColorAnalysis
from .intelligent_emotion_analysis import IntelligentEmotionAnalysis
from .object_recognition import ObjectRecognition
from .composition_analysis import CompositionAnalysis
from .visual_sentiment_analysis import VisualSentimentAnalysis
from .pdf_report_generator import PDFReportGenerator
import json
import os
import logging

class VisualPersonaGenerator:
    def __init__(self, gemini_api_key=None):
        self.logger = logging.getLogger(__name__)
        self.color_analyzer = ColorAnalysis()
        self.emotion_analyzer = IntelligentEmotionAnalysis(gemini_api_key=gemini_api_key)
        self.object_analyzer = ObjectRecognition()
        self.composition_analyzer = CompositionAnalysis()
        self.sentiment_analyzer = VisualSentimentAnalysis()
        self.pdf_generator = PDFReportGenerator()
        
        if gemini_api_key:
            self.logger.info("Visual Persona Generator initialized with Gemini API integration")
        else:
            self.logger.info("Visual Persona Generator initialized without Gemini API")

    def analyze_single_image(self, image_path):
        """
        Analyze a single image with enhanced error handling and scene emotion fallback.
        """
        try:
            color_res = self.color_analyzer.analyze(image_path)
        except Exception as e:
            color_res = {"error": f"Color analysis failed: {str(e)}"}
        
        try:
            emotion_res = self.emotion_analyzer.analyze(image_path)
        except Exception as e:
            emotion_res = {"error": f"Emotion analysis failed: {str(e)}"}
        
        try:
            object_res = self.object_analyzer.analyze(image_path)
        except Exception as e:
            object_res = {"error": f"Object recognition failed: {str(e)}"}
        
        try:
            comp_res = self.composition_analyzer.analyze(image_path)
        except Exception as e:
            comp_res = {"error": f"Composition analysis failed: {str(e)}"}
        
        try:
            sent_res = self.sentiment_analyzer.analyze(color_res, emotion_res, comp_res)
        except Exception as e:
            sent_res = {"error": f"Sentiment analysis failed: {str(e)}"}
        
        return {
            "color_analysis": color_res,
            "facial_emotion": emotion_res,
            "object_recognition": object_res,
            "composition": comp_res,
            "sentiment": sent_res
        }

    def analyze_batch(self, image_paths):
        """
        Analyze a batch of images with progress tracking.
        """
        results = []
        for i, path in enumerate(image_paths):
            try:
                res = self.analyze_single_image(path)
                res["image_path"] = path
                res["image_index"] = i + 1
                results.append(res)
            except Exception as e:
                # Include failed analysis in results
                results.append({
                    "image_path": path,
                    "image_index": i + 1,
                    "error": f"Complete analysis failed: {str(e)}"
                })
        
        return results

    def generate_report(self, results, output_format="json"):
        """
        Generate a report from analysis results.
        """
        if output_format == "json":
            report_path = os.path.join("data", "report.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=4)
            return report_path
        elif output_format == "pdf":
            return self.pdf_generator.generate_pdf_report(results)
        
        return results

    def cleanup(self, image_paths):
        """
        Clean up uploaded images after analysis.
        """
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Warning: Could not delete {path}: {str(e)}")

    def get_summary_statistics(self, results):
        """
        Generate summary statistics across all analyzed images.
        """
        if not results:
            return {}
        
        total_images = len(results)
        successful_analyses = len([r for r in results if "error" not in r])
        
        # Aggregate emotion metrics
        total_positivity = 0
        total_enthusiasm = 0
        total_aesthetic = 0
        
        emotion_counts = {}
        social_tendencies = {}
        
        for result in results:
            if "error" in result:
                continue
                
            emotion_data = result.get('facial_emotion', {})
            if 'emotion_spectrum' in emotion_data:
                spectrum = emotion_data['emotion_spectrum']
                total_positivity += spectrum.get('positivity_score', 0)
                total_enthusiasm += spectrum.get('enthusiasm_score', 0)
                dominant_emotion = spectrum.get('dominant_emotion', 'unknown')
                emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
            
            if 'social_dynamics' in emotion_data:
                tendency = emotion_data['social_dynamics'].get('social_tendency', 'unknown')
                social_tendencies[tendency] = social_tendencies.get(tendency, 0) + 1
            
            comp_data = result.get('composition', {})
            if 'overall_aesthetic_score' in comp_data:
                total_aesthetic += comp_data['overall_aesthetic_score']
        
        if successful_analyses > 0:
            return {
                "total_images": total_images,
                "successful_analyses": successful_analyses,
                "analysis_success_rate": round(successful_analyses / total_images, 3),
                "average_positivity": round(total_positivity / successful_analyses, 3),
                "average_enthusiasm": round(total_enthusiasm / successful_analyses, 3),
                "average_aesthetic_quality": round(total_aesthetic / successful_analyses, 3),
                "dominant_emotions": emotion_counts,
                "social_preferences": social_tendencies
            }
        
        return {"total_images": total_images, "successful_analyses": 0}