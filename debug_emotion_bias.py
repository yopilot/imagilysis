#!/usr/bin/env python3
"""
Emotion Detection Debug Script - Test with sample images
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.enhanced_facial_emotion_analysis import EnhancedFacialEmotionAnalysis
import logging
from PIL import Image
import numpy as np

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_emotion_detection_bias():
    """Test emotion detection to identify bias towards anger."""
    print("🧪 Testing Emotion Detection Bias...")
    
    # Initialize the emotion analyzer
    analyzer = EnhancedFacialEmotionAnalysis()
    
    # Create a simple test image (white square - should be neutral)
    print("\n📊 Test 1: Neutral white image")
    white_image = Image.new('RGB', (224, 224), color='white')
    
    # Test raw analyze_face_emotions (before enhancement)
    print("Testing raw emotion analysis...")
    raw_emotions = analyzer.analyze_face_emotions(white_image)
    print(f"Raw emotions: {raw_emotions}")
    
    # Test enhanced emotions
    print("Testing enhanced emotion analysis...")
    enhanced_emotions = analyzer.sophisticated_emotion_analysis(white_image)
    print(f"Enhanced emotions: {enhanced_emotions}")
    
    # Check which emotion is dominant
    if enhanced_emotions:
        dominant_emotion = max(enhanced_emotions, key=enhanced_emotions.get)
        dominant_score = enhanced_emotions[dominant_emotion]
        print(f"🎯 Dominant emotion: {dominant_emotion} ({dominant_score:.1%})")
        
        if dominant_emotion == 'anger' and dominant_score > 0.5:
            print("❌ PROBLEM: Neutral image showing anger as dominant!")
        else:
            print("✅ Result looks reasonable")
    
    # Test 2: Create a "happy" pattern (bright colors)
    print("\n📊 Test 2: Bright colorful image (should suggest happiness)")
    
    # Create a simple bright image
    bright_array = np.full((224, 224, 3), [255, 255, 100], dtype=np.uint8)  # Bright yellow
    bright_image = Image.fromarray(bright_array)
    
    bright_emotions = analyzer.sophisticated_emotion_analysis(bright_image)
    print(f"Bright image emotions: {bright_emotions}")
    
    if bright_emotions:
        bright_dominant = max(bright_emotions, key=bright_emotions.get)
        bright_score = bright_emotions[bright_dominant]
        print(f"🎯 Dominant emotion: {bright_dominant} ({bright_score:.1%})")
    
    # Test 3: Manual emotion score enhancement test
    print("\n📊 Test 3: Testing enhancement function directly")
    
    # Simulate what should be a happy result
    test_happy_scores = {
        'happiness': 0.45,  # Should be dominant
        'sadness': 0.15,
        'anger': 0.10,
        'fear': 0.12,
        'disgust': 0.08,
        'surprise': 0.10,
        'neutral': 0.0
    }
    
    print(f"Input happy scores: {test_happy_scores}")
    enhanced_happy = analyzer.enhance_emotion_scores(test_happy_scores)
    print(f"Enhanced happy scores: {enhanced_happy}")
    
    happy_dominant = max(enhanced_happy, key=enhanced_happy.get)
    print(f"🎯 Should be happiness, got: {happy_dominant} ({enhanced_happy[happy_dominant]:.1%})")
    
    if happy_dominant != 'happiness':
        print("❌ PROBLEM: Enhancement function not preserving happiness dominance!")
    
    print("\n🔍 Diagnosis complete!")

if __name__ == "__main__":
    test_emotion_detection_bias()