import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans

class SceneEmotionAnalysis:
    """
    Analyzes emotional content of scenes without relying on faces.
    Can differentiate between different scene types and moods.
    """
    
    def __init__(self):
        # Color emotion mappings based on color psychology
        self.color_emotions = {
            'warm_colors': {
                'emotions': ['happy', 'energetic', 'excited', 'passionate'],
                'positivity': 0.8,
                'ranges': [(0, 30), (300, 360)]  # Red-orange range in HSV
            },
            'cool_colors': {
                'emotions': ['calm', 'serene', 'peaceful', 'melancholic'],
                'positivity': 0.6,
                'ranges': [(180, 270)]  # Blue-cyan range
            },
            'dark_colors': {
                'emotions': ['mysterious', 'somber', 'dramatic', 'spooky'],
                'positivity': 0.2,
                'ranges': None  # Based on brightness/value
            },
            'bright_colors': {
                'emotions': ['joyful', 'vibrant', 'optimistic', 'cheerful'],
                'positivity': 0.9,
                'ranges': None  # Based on saturation/value
            }
        }
        
        # Scene type patterns based on color combinations and brightness
        self.scene_patterns = {
            'sunrise_sunset': {
                'colors': ['orange', 'pink', 'yellow', 'red'],
                'brightness_range': (0.3, 0.9),
                'warmth_threshold': 0.7,
                'emotions': ['peaceful', 'hopeful', 'romantic', 'inspiring'],
                'positivity': 0.85
            },
            'forest_nature': {
                'colors': ['green', 'brown'],
                'brightness_range': (0.2, 0.8),
                'green_dominance': 0.4,
                'emotions': ['natural', 'fresh', 'alive', 'grounded'],
                'positivity': 0.7
            },
            'dark_spooky': {
                'colors': ['black', 'dark_brown', 'dark_purple'],
                'brightness_range': (0.0, 0.3),
                'low_saturation': True,
                'emotions': ['spooky', 'mysterious', 'eerie', 'dramatic'],
                'positivity': 0.15
            },
            'bright_cheerful': {
                'colors': ['yellow', 'light_blue', 'pink', 'white'],
                'brightness_range': (0.6, 1.0),
                'high_saturation': True,
                'emotions': ['cheerful', 'happy', 'bright', 'uplifting'],
                'positivity': 0.9
            },
            'urban_cityscape': {
                'colors': ['gray', 'blue', 'metal'],
                'contrast_high': True,
                'emotions': ['modern', 'busy', 'dynamic', 'urban'],
                'positivity': 0.5
            }
        }

    def extract_scene_features(self, image_path):
        """Extract comprehensive scene features for emotion analysis."""
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Resize for efficiency
        img_resized = img.resize((224, 224))
        img_array_small = np.array(img_resized)
        
        features = {}
        
        # Color analysis
        features.update(self._analyze_colors(img_array_small))
        
        # Brightness and contrast analysis
        features.update(self._analyze_brightness_contrast(img_array_small))
        
        # Texture and edge analysis
        features.update(self._analyze_texture_edges(img_array_small))
        
        return features

    def _analyze_colors(self, img_array):
        """Analyze color characteristics of the image."""
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Extract dominant colors using K-means
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_
        
        # Calculate color properties
        hues = hsv_img[:, :, 0].flatten()
        saturations = hsv_img[:, :, 1].flatten()
        values = hsv_img[:, :, 2].flatten()
        
        # Color temperature (warm vs cool)
        warm_mask = ((hues < 30) | (hues > 300)) | ((hues >= 30) & (hues <= 90))
        warmth_ratio = np.sum(warm_mask) / len(hues)
        
        # Color diversity
        color_variance = np.var(hues)
        
        return {
            'dominant_colors': dominant_colors,
            'avg_hue': np.mean(hues),
            'avg_saturation': np.mean(saturations) / 255.0,
            'avg_value': np.mean(values) / 255.0,
            'warmth_ratio': warmth_ratio,
            'color_variance': color_variance,
            'hue_distribution': np.histogram(hues, bins=12)[0]
        }

    def _analyze_brightness_contrast(self, img_array):
        """Analyze brightness and contrast characteristics."""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Brightness metrics
        avg_brightness = np.mean(gray) / 255.0
        brightness_variance = np.var(gray) / (255.0 ** 2)
        
        # Contrast metrics
        contrast = np.std(gray) / 255.0
        dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
        
        # Brightness distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / np.sum(hist)
        
        # Check for darkness (spooky indicator)
        dark_pixel_ratio = np.sum(gray < 64) / gray.size
        bright_pixel_ratio = np.sum(gray > 192) / gray.size
        
        return {
            'avg_brightness': avg_brightness,
            'brightness_variance': brightness_variance,
            'contrast': contrast,
            'dynamic_range': dynamic_range,
            'dark_pixel_ratio': dark_pixel_ratio,
            'bright_pixel_ratio': bright_pixel_ratio,
            'brightness_distribution': hist_normalized
        }

    def _analyze_texture_edges(self, img_array):
        """Analyze texture and edge characteristics."""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture analysis using local binary patterns (simplified)
        texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'edge_density': edge_density,
            'texture_variance': texture_variance / 10000.0  # Normalize
        }

    def classify_scene_type(self, features):
        """Classify the scene type based on extracted features."""
        scores = {}
        
        for scene_type, pattern in self.scene_patterns.items():
            score = 0
            
            # Brightness range check
            if 'brightness_range' in pattern:
                min_bright, max_bright = pattern['brightness_range']
                if min_bright <= features['avg_brightness'] <= max_bright:
                    score += 0.3
            
            # Warmth check for sunrise/sunset
            if scene_type == 'sunrise_sunset':
                if features['warmth_ratio'] > 0.6 and features['avg_saturation'] > 0.4:
                    score += 0.4
                # Check for warm color dominance
                warm_hues = features['hue_distribution'][:3].sum() + features['hue_distribution'][-3:].sum()
                if warm_hues > features['hue_distribution'][3:-3].sum():
                    score += 0.3
            
            # Forest/nature check
            elif scene_type == 'forest_nature':
                # Check for green dominance (hues 60-180 in HSV)
                green_range = features['hue_distribution'][2:6].sum()
                if green_range > 0.3 * features['hue_distribution'].sum():
                    score += 0.4
                if 0.3 <= features['avg_brightness'] <= 0.7:
                    score += 0.3
            
            # Dark/spooky check
            elif scene_type == 'dark_spooky':
                if features['avg_brightness'] < 0.3:
                    score += 0.4
                if features['dark_pixel_ratio'] > 0.5:
                    score += 0.3
                if features['avg_saturation'] < 0.3:  # Desaturated
                    score += 0.3
            
            # Bright/cheerful check
            elif scene_type == 'bright_cheerful':
                if features['avg_brightness'] > 0.6:
                    score += 0.4
                if features['bright_pixel_ratio'] > 0.3:
                    score += 0.3
                if features['avg_saturation'] > 0.5:
                    score += 0.3
            
            scores[scene_type] = score
        
        # Find the best matching scene type
        best_scene = max(scores, key=scores.get) if max(scores.values()) > 0.4 else 'general'
        confidence = scores.get(best_scene, 0)
        
        return best_scene, confidence, scores

    def calculate_scene_emotion(self, scene_type, features):
        """Calculate emotion metrics based on scene type and features."""
        if scene_type in self.scene_patterns:
            pattern = self.scene_patterns[scene_type]
            base_emotions = pattern['emotions']
            base_positivity = pattern['positivity']
        else:
            # General scene analysis based on color and brightness
            if features['avg_brightness'] > 0.6 and features['warmth_ratio'] > 0.5:
                base_emotions = ['pleasant', 'bright', 'positive']
                base_positivity = 0.7
            elif features['avg_brightness'] < 0.3 or features['dark_pixel_ratio'] > 0.6:
                base_emotions = ['dark', 'mysterious', 'somber']
                base_positivity = 0.25
            else:
                base_emotions = ['neutral', 'balanced']
                base_positivity = 0.5
        
        # Adjust positivity based on brightness and saturation
        brightness_factor = features['avg_brightness']
        saturation_factor = min(features['avg_saturation'], 0.8)  # Cap saturation effect
        
        adjusted_positivity = base_positivity * (0.7 + 0.3 * brightness_factor) * (0.8 + 0.2 * saturation_factor)
        adjusted_positivity = max(0, min(1, adjusted_positivity))
        
        # Calculate emotion spectrum similar to facial analysis
        emotion_spectrum = {
            'positivity_score': round(adjusted_positivity, 3),
            'negativity_score': round(1 - adjusted_positivity, 3),
            'brightness_mood': 'bright' if features['avg_brightness'] > 0.6 else 'dark' if features['avg_brightness'] < 0.3 else 'moderate',
            'color_warmth': 'warm' if features['warmth_ratio'] > 0.6 else 'cool' if features['warmth_ratio'] < 0.4 else 'neutral',
            'emotional_intensity': round(features['brightness_variance'] + features['color_variance'] / 1000, 3),
            'scene_emotions': base_emotions
        }
        
        return emotion_spectrum

    def analyze(self, image_path):
        """
        Complete scene emotion analysis for images without faces.
        """
        try:
            # Extract scene features
            features = self.extract_scene_features(image_path)
            
            # Classify scene type
            scene_type, confidence, scene_scores = self.classify_scene_type(features)
            
            # Calculate emotion metrics
            emotion_spectrum = self.calculate_scene_emotion(scene_type, features)
            
            return {
                'scene_type': scene_type,
                'scene_confidence': round(confidence, 3),
                'scene_type_scores': {k: round(v, 3) for k, v in scene_scores.items()},
                'emotion_spectrum': emotion_spectrum,
                'scene_features': {
                    'brightness': round(features['avg_brightness'], 3),
                    'saturation': round(features['avg_saturation'], 3),
                    'warmth_ratio': round(features['warmth_ratio'], 3),
                    'contrast': round(features['contrast'], 3),
                    'dark_pixel_ratio': round(features['dark_pixel_ratio'], 3),
                    'bright_pixel_ratio': round(features['bright_pixel_ratio'], 3)
                }
            }
            
        except Exception as e:
            return {"error": f"Scene emotion analysis failed: {str(e)}"}