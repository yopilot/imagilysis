import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

class ObjectRecognition:
    def __init__(self):
        # Lazy loading - models will be loaded when first used
        self.processor = None
        self.model = None
    
    def _load_models(self):
        """Load models only when needed (lazy loading)."""
        if self.processor is None or self.model is None:
            print("Loading DETR model for object detection...")
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            print("DETR model loaded successfully!")

    def detect_objects(self, image_path):
        """
        Detect objects using DETR.
        """
        self._load_models()  # Load models if not already loaded
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)
        return results[0]  # For single image

    def categorize_lifestyle(self, detections):
        """
        Categorize detected objects into lifestyle patterns.
        """
        labels = detections['labels']
        scores = detections['scores']
        
        # Filter by confidence
        keep = scores > 0.7
        labels = labels[keep]
        
        # Lifestyle mapping (simplified)
        lifestyle_objects = {
            'person': 'social',
            'car': 'active',
            'bicycle': 'active',
            'motorcycle': 'active',
            'airplane': 'travel',
            'bus': 'travel',
            'train': 'travel',
            'boat': 'leisure',
            'bird': 'nature',
            'cat': 'pet',
            'dog': 'pet',
            'backpack': 'travel',
            'handbag': 'fashion',
            'suitcase': 'travel',
            'frisbee': 'leisure',
            'skis': 'sports',
            'snowboard': 'sports',
            'sports ball': 'sports',
            'kite': 'leisure',
            'baseball bat': 'sports',
            'skateboard': 'sports',
            'surfboard': 'sports',
            'tennis racket': 'sports',
            'bottle': 'everyday',
            'wine glass': 'social',
            'cup': 'everyday',
            'fork': 'food',
            'knife': 'food',
            'spoon': 'food',
            'bowl': 'food',
            'banana': 'food',
            'apple': 'food',
            'sandwich': 'food',
            'pizza': 'food',
            'cake': 'food',
            'chair': 'furniture',
            'couch': 'furniture',
            'potted plant': 'home',
            'bed': 'home',
            'dining table': 'home',
            'tv': 'entertainment',
            'laptop': 'work',
            'cell phone': 'modern',
            'book': 'intellectual',
            'clock': 'practical',
            'vase': 'decor',
            'scissors': 'practical',
            'teddy bear': 'childish',
            'hair drier': 'personal',
            'toothbrush': 'personal'
        }
        
        categories = {}
        detected = []
        for label_id in labels:
            label = self.model.config.id2label[label_id.item()]
            detected.append(label)
            category = lifestyle_objects.get(label, 'other')
            categories[category] = categories.get(category, 0) + 1
        
        # Calculate scores
        activity_level = categories.get('active', 0) + categories.get('sports', 0) + categories.get('travel', 0)
        sophistication = len([c for c in categories if c in ['fashion', 'decor', 'intellectual', 'formal']])
        
        return {
            "detected_objects": detected,
            "lifestyle_categories": categories,
            "activity_level": activity_level,
            "sophistication_score": sophistication
        }

    def analyze(self, image_path):
        """
        Enhanced object recognition and lifestyle analysis with better activity calculation.
        """
        try:
            detections = self.detect_objects(image_path)
            lifestyle = self.categorize_lifestyle(detections)
            
            # Enhanced activity level calculation
            base_activity = lifestyle.get('activity_level', 0)
            
            # If no activity detected, analyze image properties
            if base_activity == 0:
                enhanced_activity = self._calculate_image_activity(image_path)
                lifestyle['activity_level'] = enhanced_activity
            
            return lifestyle
        except Exception as e:
            # Enhanced fallback with image analysis
            return self._analyze_image_fallback(image_path)

    def _calculate_image_activity(self, image_path):
        """
        Calculate activity level from image properties when object detection fails.
        """
        try:
            import cv2
            import numpy as np
            
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for scene complexity
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Color variety analysis
            b, g, r = cv2.split(img)
            color_std = (np.std(r) + np.std(g) + np.std(b)) / (3 * 255)
            
            # Brightness variation
            brightness = np.mean(gray) / 255.0
            brightness_var = np.std(gray) / 255.0
            
            # Calculate activity score
            activity_score = (
                edge_ratio * 4 +      # Scene complexity
                color_std * 3 +       # Visual diversity
                brightness_var * 2 +  # Dynamic lighting
                (1 if brightness > 0.6 else 0.5)  # Brightness bonus
            )
            
            # Scale to 0-5 range
            return round(min(5, max(0.5, activity_score * 2)), 1)
            
        except Exception:
            return 1.5  # Default moderate activity

    def _analyze_image_fallback(self, image_path):
        """
        Complete fallback analysis when object detection fails.
        """
        try:
            activity_level = self._calculate_image_activity(image_path)
            
            return {
                "detected_objects": ["image_analysis"],
                "lifestyle_categories": {"general": 1},
                "activity_level": activity_level,
                "sophistication_score": 0.5,
                "analysis_method": "fallback_image_analysis"
            }
        except Exception as e:
            return {
                "detected_objects": [],
                "lifestyle_categories": {"unknown": 1},
                "activity_level": 1.0,
                "sophistication_score": 0.3,
                "error": f"Complete analysis failed: {str(e)}"
            }