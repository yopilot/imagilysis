import unittest
import os
import tempfile
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.facial_emotion_analysis import FacialEmotionAnalysis
from src.color_analysis import ColorAnalysis
from src.composition_analysis import CompositionAnalysis
from src.object_recognition import ObjectRecognition
from src.visual_sentiment_analysis import VisualSentimentAnalysis
from src.visual_persona_generator import VisualPersonaGenerator

class TestAnalysisModules(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with sample images."""
        self.test_image = self.create_test_image()
        self.emotion_analyzer = FacialEmotionAnalysis()
        self.color_analyzer = ColorAnalysis()
        self.composition_analyzer = CompositionAnalysis()
        self.sentiment_analyzer = VisualSentimentAnalysis()
        self.generator = VisualPersonaGenerator()
    
    def create_test_image(self, size=(224, 224), color=(128, 128, 128)):
        """Create a test image for analysis."""
        img = Image.new('RGB', size, color)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        img.save(temp_file.name)
        return temp_file.name
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_image):
            os.unlink(self.test_image)
    
    def test_emotion_analysis_structure(self):
        """Test facial emotion analysis returns expected structure."""
        result = self.emotion_analyzer.analyze(self.test_image)
        
        # Check main structure
        self.assertIn('emotion_distribution', result)
        self.assertIn('emotion_spectrum', result)
        self.assertIn('social_dynamics', result)
        self.assertIn('authenticity_score', result)
        
        # Check emotion spectrum structure
        spectrum = result['emotion_spectrum']
        expected_keys = ['positivity_score', 'negativity_score', 'enthusiasm_score', 
                        'emotional_intensity', 'dominant_emotion']
        for key in expected_keys:
            self.assertIn(key, spectrum)
        
        # Check score ranges
        self.assertGreaterEqual(spectrum['positivity_score'], 0)
        self.assertLessEqual(spectrum['positivity_score'], 1)
        self.assertGreaterEqual(spectrum['enthusiasm_score'], 0)
        self.assertLessEqual(spectrum['enthusiasm_score'], 1)
    
    def test_color_analysis_structure(self):
        """Test color analysis returns expected structure."""
        result = self.color_analyzer.analyze(self.test_image)
        
        expected_keys = ['dominant_colors', 'mood', 'personality_traits', 'palette_visual']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check data types
        self.assertIsInstance(result['dominant_colors'], list)
        self.assertIsInstance(result['mood'], str)
        self.assertIsInstance(result['personality_traits'], list)
    
    def test_composition_analysis_structure(self):
        """Test composition analysis returns expected structure."""
        result = self.composition_analyzer.analyze(self.test_image)
        
        expected_keys = ['symmetry_score', 'balance_score', 'rule_of_thirds_score', 
                        'overall_aesthetic_score']
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float))
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 1)
    
    def test_sentiment_analysis_structure(self):
        """Test visual sentiment analysis returns expected structure."""
        # Create mock inputs
        color_results = {'mood': 'happy'}
        emotion_results = {'emotion_distribution': {'happy': 0.6, 'neutral': 0.4}}
        composition_results = {'overall_aesthetic_score': 0.75}
        
        result = self.sentiment_analyzer.analyze(color_results, emotion_results, composition_results)
        
        expected_keys = ['overall_mood', 'color_sentiment', 'consistency_score', 'curation_type']
        for key in expected_keys:
            self.assertIn(key, result)
    
    def test_generator_single_image_analysis(self):
        """Test the main generator analyzes a single image correctly."""
        result = self.generator.analyze_single_image(self.test_image)
        
        expected_modules = ['color_analysis', 'facial_emotion', 'object_recognition', 
                          'composition', 'sentiment']
        for module in expected_modules:
            self.assertIn(module, result)
    
    def test_generator_batch_analysis(self):
        """Test batch analysis functionality."""
        # Create multiple test images
        test_images = [self.create_test_image() for _ in range(3)]
        
        try:
            results = self.generator.analyze_batch(test_images)
            
            # Check we get results for all images
            self.assertEqual(len(results), 3)
            
            # Check each result has the expected structure
            for result in results:
                self.assertIn('image_path', result)
                self.assertIn('image_index', result)
                self.assertIn('color_analysis', result)
        
        finally:
            # Clean up test images
            for img_path in test_images:
                if os.path.exists(img_path):
                    os.unlink(img_path)
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        # Test with non-existent file
        result = self.emotion_analyzer.analyze("nonexistent_file.jpg")
        self.assertIn('error', result)
        
        # Test with invalid image path for color analysis
        result = self.color_analyzer.analyze("invalid_path.jpg")
        # Should handle gracefully and not crash
        self.assertIsInstance(result, dict)
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        # Create mock results
        mock_results = [
            {
                'facial_emotion': {
                    'emotion_spectrum': {
                        'positivity_score': 0.8,
                        'enthusiasm_score': 0.6,
                        'dominant_emotion': 'happy'
                    },
                    'authenticity_score': 0.7,
                    'social_dynamics': {'social_tendency': 'solo'}
                },
                'composition': {'overall_aesthetic_score': 0.75}
            },
            {
                'facial_emotion': {
                    'emotion_spectrum': {
                        'positivity_score': 0.6,
                        'enthusiasm_score': 0.4,
                        'dominant_emotion': 'neutral'
                    },
                    'authenticity_score': 0.8,
                    'social_dynamics': {'social_tendency': 'group'}
                },
                'composition': {'overall_aesthetic_score': 0.65}
            }
        ]
        
        summary = self.generator.get_summary_statistics(mock_results)
        
        # Check summary structure
        expected_keys = ['total_images', 'successful_analyses', 'analysis_success_rate',
                        'average_positivity', 'average_enthusiasm', 'average_authenticity',
                        'average_aesthetic_quality', 'dominant_emotions', 'social_preferences']
        
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check calculations
        self.assertEqual(summary['total_images'], 2)
        self.assertEqual(summary['successful_analyses'], 2)
        self.assertEqual(summary['average_positivity'], 0.7)  # (0.8 + 0.6) / 2

class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""
    
    def test_emotion_spectrum_calculation(self):
        """Test emotion spectrum calculations with edge cases."""
        analyzer = FacialEmotionAnalysis()
        
        # Test with empty emotion distribution
        spectrum = analyzer.calculate_emotion_spectrum({})
        self.assertEqual(spectrum['positivity_score'], 0.0)
        self.assertEqual(spectrum['dominant_emotion'], 'neutral')
        
        # Test with all happy emotions
        all_happy = {'happy': 1.0, 'sad': 0.0, 'angry': 0.0, 'surprised': 0.0, 
                    'neutral': 0.0, 'fear': 0.0, 'disgust': 0.0}
        spectrum = analyzer.calculate_emotion_spectrum(all_happy)
        self.assertGreater(spectrum['positivity_score'], 0.5)
        self.assertEqual(spectrum['dominant_emotion'], 'happy')
    
    def test_composition_score_ranges(self):
        """Test composition scores are within valid ranges."""
        analyzer = CompositionAnalysis()
        
        # Create test image
        test_img = Image.new('RGB', (100, 100), (255, 255, 255))
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        test_img.save(temp_file.name)
        
        try:
            result = analyzer.analyze(temp_file.name)
            
            # Check all scores are in valid range [0, 1]
            for score_name, score_value in result.items():
                self.assertGreaterEqual(score_value, 0, f"{score_name} below 0")
                self.assertLessEqual(score_value, 1, f"{score_name} above 1")
        
        finally:
            os.unlink(temp_file.name)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)