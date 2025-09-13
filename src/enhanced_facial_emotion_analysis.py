import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import logging
from PIL import Image
import timm
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class EnhancedFacialEmotionAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.emotion_pipeline = None
        self.face_cascade = None
        
        # Core 6 emotions mapping (industry standard)
        self.core_emotions = ['happiness', 'sadness', 'fear', 'anger', 'disgust', 'surprise']
        
        # Enhanced emotion weights for better accuracy
        self.positivity_weights = {
            'happiness': 1.0, 'surprise': 0.7, 'neutral': 0.0, 
            'sadness': -0.9, 'anger': -1.0, 'fear': -0.8, 'disgust': -0.95
        }
        self.enthusiasm_weights = {
            'happiness': 0.95, 'surprise': 1.0, 'anger': 0.2, 
            'neutral': 0.0, 'sadness': -0.6, 'fear': -0.4, 'disgust': -0.3
        }
        
        self._setup_device()
        self.log_system_info()

    def _setup_device(self):
        """Setup GPU/CPU device with proper optimization."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            self.torch_dtype = torch.float16
            self.logger.info(f"GPU detected! Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32
            self.logger.info("No GPU detected. Using CPU for inference.")

    def log_system_info(self):
        """Log detailed system information."""
        if torch.cuda.is_available():
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                self.logger.info(f"GPU {i}: {gpu_name} ({memory_total:.1f}GB total, "
                               f"{memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved)")

    def _load_models(self):
        """Load enhanced face detection and emotion recognition models."""
        if self.face_cascade is None:
            self.logger.info("Loading enhanced OpenCV face detection...")
            # Use more accurate DNN face detection when available
            try:
                # Try to load DNN face detector (more accurate)
                self.face_net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 
                                                             'opencv_face_detector.pbtxt')
                self.use_dnn = True
                self.logger.info("DNN face detector loaded successfully!")
            except:
                # Fallback to Haar cascades but with better parameters
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.use_dnn = False
                self.logger.info("Using enhanced Haar cascade face detection")

        if self.emotion_pipeline is None:
            self.logger.info("Loading emotion recognition pipeline...")
            try:
                # Using HuggingFace's emotion recognition with GPU support
                device_id = 0 if torch.cuda.is_available() else -1
                self.emotion_pipeline = pipeline(
                    "image-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=device_id
                )
                self.logger.info("HuggingFace emotion recognition pipeline loaded successfully!")
            except Exception as e:
                self.logger.warning(f"Failed to load HuggingFace emotion pipeline: {e}")
                self.logger.info("Using enhanced heuristic emotion analysis")
                self.emotion_pipeline = None

    def detect_faces_enhanced(self, image_path):
        """
        Enhanced face detection using available methods.
        """
        self._load_models()
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0, [], [], []
            
            if self.use_dnn and hasattr(self, 'face_net'):
                # Use DNN detection (more accurate)
                return self._detect_faces_dnn(image)
            else:
                # Use enhanced Haar cascades
                return self._detect_faces_haar(image)
                
        except Exception as e:
            self.logger.error(f"Enhanced face detection error: {e}")
            return 0, [], [], []

    def _detect_faces_dnn(self, image):
        """DNN-based face detection (more accurate than Haar)."""
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        boxes = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.7:  # High confidence threshold
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Extract face crop
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    faces.append(face_pil)
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(confidence)
        
        return len(faces), faces, boxes, confidences

    def _detect_faces_haar(self, image):
        """Enhanced Haar cascade face detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced parameters for better detection
        faces_coords = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # More sensitive
            minNeighbors=6,    # Better filtering
            minSize=(30, 30),  # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        boxes = []
        confidences = []
        
        for (x, y, w, h) in faces_coords:
            # Extract face crop
            face_crop = image[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            faces.append(face_pil)
            boxes.append([x, y, x+w, y+h])
            confidences.append(0.8)  # Default confidence for Haar
        
        return len(faces), faces, boxes, confidences

    def analyze_face_emotions(self, face_image):
        """
        Analyze emotions in a single face using available models.
        """
        try:
            if self.emotion_pipeline:
                # Use HuggingFace pipeline for emotion detection
                results = self.emotion_pipeline(face_image)
                
                # Convert to our standard emotion format
                emotion_scores = {}
                for result in results:
                    label = result['label'].lower()
                    score = result['score']
                    
                    # Map to core emotions
                    if 'joy' in label or 'happy' in label:
                        emotion_scores['happiness'] = score
                    elif 'sad' in label:
                        emotion_scores['sadness'] = score
                    elif 'anger' in label or 'angry' in label:
                        emotion_scores['anger'] = score
                    elif 'fear' in label:
                        emotion_scores['fear'] = score
                    elif 'disgust' in label:
                        emotion_scores['disgust'] = score
                    elif 'surprise' in label:
                        emotion_scores['surprise'] = score
                    else:
                        emotion_scores['neutral'] = score
                
                return emotion_scores
                
            else:
                # Enhanced heuristic analysis based on image properties
                return self._analyze_face_heuristic(face_image)
                
        except Exception as e:
            self.logger.error(f"Emotion analysis error: {e}")
            return self._analyze_face_heuristic(face_image)

    def _analyze_face_heuristic(self, face_image):
        """
        Enhanced heuristic emotion analysis as fallback.
        """
        try:
            # Convert to numpy array for analysis
            face_array = np.array(face_image)
            
            # Analyze brightness and color distribution
            brightness = np.mean(face_array)
            color_variance = np.var(face_array, axis=(0, 1))
            
            # Analyze face regions (very basic)
            h, w = face_array.shape[:2]
            
            # Upper face (eyes/forehead area)
            upper_region = face_array[:h//2, :]
            upper_brightness = np.mean(upper_region)
            
            # Lower face (mouth area)
            lower_region = face_array[h//2:, :]
            lower_brightness = np.mean(lower_region)
            
            # Simple heuristics for emotion estimation
            emotion_scores = {}
            
            # Brightness-based emotion estimation
            if brightness > 120:  # Bright face
                if lower_brightness > upper_brightness * 1.1:  # Bright mouth area
                    emotion_scores['happiness'] = 0.7
                    emotion_scores['surprise'] = 0.2
                    emotion_scores['neutral'] = 0.1
                else:
                    emotion_scores['neutral'] = 0.6
                    emotion_scores['happiness'] = 0.3
                    emotion_scores['surprise'] = 0.1
            else:  # Darker face
                emotion_scores['sadness'] = 0.4
                emotion_scores['neutral'] = 0.4
                emotion_scores['fear'] = 0.2
            
            # Ensure all core emotions are present
            for emotion in self.core_emotions:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.0
            
            # Add neutral if not present
            if 'neutral' not in emotion_scores:
                emotion_scores['neutral'] = 0.1
            
            # Normalize scores
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v/total for k, v in emotion_scores.items()}
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Heuristic emotion analysis error: {e}")
            # Return neutral emotion as ultimate fallback
            return {
                'happiness': 0.0, 'sadness': 0.0, 'fear': 0.0,
                'anger': 0.0, 'disgust': 0.0, 'surprise': 0.0, 'neutral': 1.0
            }

    def aggregate_face_emotions(self, all_face_emotions):
        """
        Aggregate emotions from multiple faces with advanced weighting.
        """
        if not all_face_emotions:
            return {}
        
        # Initialize aggregated scores
        aggregated = {emotion: 0.0 for emotion in self.core_emotions + ['neutral']}
        
        # Weight faces by their emotion intensity
        total_weight = 0
        
        for face_emotions in all_face_emotions:
            # Calculate emotion intensity for this face
            intensity = max(face_emotions.values()) - min(face_emotions.values())
            weight = 1.0 + intensity  # Higher intensity = higher weight
            
            for emotion, score in face_emotions.items():
                if emotion in aggregated:
                    aggregated[emotion] += score * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for emotion in aggregated:
                aggregated[emotion] /= total_weight
        
        # Ensure probabilities sum to 1
        total_prob = sum(aggregated.values())
        if total_prob > 0:
            for emotion in aggregated:
                aggregated[emotion] /= total_prob
        
        return aggregated

    def calculate_enhanced_emotion_spectrum(self, emotion_dist):
        """
        Calculate advanced emotion spectrum with better accuracy.
        """
        if not emotion_dist:
            return {
                'positivity_score': 0.0,
                'negativity_score': 0.0,
                'enthusiasm_score': 0.0,
                'emotional_intensity': 0.0,
                'dominant_emotion': 'neutral',
                'core_emotions': {emotion: 0.0 for emotion in self.core_emotions}
            }
        
        # Calculate weighted scores with better mapping
        positivity_score = 0.0
        for emotion, score in emotion_dist.items():
            if emotion in self.positivity_weights:
                positivity_score += score * self.positivity_weights[emotion]
        
        # Ensure scores are in proper range
        positivity_score = max(0, min(1, (positivity_score + 1) / 2))  # Normalize to 0-1
        negativity_score = 1 - positivity_score
        
        # Calculate enthusiasm
        enthusiasm_score = 0.0
        for emotion, score in emotion_dist.items():
            if emotion in self.enthusiasm_weights:
                enthusiasm_score += score * self.enthusiasm_weights[emotion]
        
        enthusiasm_score = max(0, min(1, (enthusiasm_score + 1) / 2))
        
        # Emotional intensity (how far from neutral)
        emotional_intensity = 1 - emotion_dist.get('neutral', 0)
        
        # Dominant emotion
        dominant_emotion = max(emotion_dist, key=emotion_dist.get)
        
        # Core emotions for visualization
        core_emotions = {}
        for emotion in self.core_emotions:
            core_emotions[emotion] = emotion_dist.get(emotion, 0.0)
        
        return {
            'positivity_score': round(positivity_score, 3),
            'negativity_score': round(negativity_score, 3),
            'enthusiasm_score': round(enthusiasm_score, 3),
            'emotional_intensity': round(emotional_intensity, 3),
            'dominant_emotion': dominant_emotion,
            'core_emotions': {k: round(v, 3) for k, v in core_emotions.items()}
        }

    def assess_enhanced_social_dynamics(self, num_faces, emotion_dist, face_confidences=None):
        """
        Enhanced social dynamics analysis with confidence weighting.
        """
        if num_faces == 0:
            return {
                'social_setting': 'no_faces',
                'group_harmony': 0.0,
                'social_confidence': 0.0,
                'social_energy': 0.0
            }
        
        # Social setting classification
        if num_faces == 1:
            social_setting = 'portrait'
        elif num_faces == 2:
            social_setting = 'couple'
        elif num_faces <= 5:
            social_setting = 'small_group'
        else:
            social_setting = 'large_group'
        
        # Group harmony based on positive emotions
        positive_emotions = (emotion_dist.get('happiness', 0) + 
                           emotion_dist.get('surprise', 0) * 0.7)
        negative_emotions = (emotion_dist.get('sadness', 0) + 
                           emotion_dist.get('anger', 0) + 
                           emotion_dist.get('fear', 0) + 
                           emotion_dist.get('disgust', 0))
        
        group_harmony = max(0, positive_emotions - negative_emotions * 0.5)
        
        # Social confidence
        confidence_emotions = (emotion_dist.get('happiness', 0) * 0.9 + 
                             emotion_dist.get('surprise', 0) * 0.6 + 
                             emotion_dist.get('neutral', 0) * 0.4)
        social_confidence = confidence_emotions
        
        # Social energy (overall emotional activation)
        high_energy = (emotion_dist.get('happiness', 0) + 
                      emotion_dist.get('surprise', 0) + 
                      emotion_dist.get('anger', 0) * 0.5)
        social_energy = high_energy
        
        return {
            'social_setting': social_setting,
            'group_harmony': round(group_harmony, 3),
            'social_confidence': round(social_confidence, 3),
            'social_energy': round(social_energy, 3)
        }

    def analyze(self, image_path):
        """
        Comprehensive enhanced facial emotion analysis with GPU acceleration.
        """
        try:
            # Enhanced face detection
            num_faces, face_crops, boxes, confidences = self.detect_faces_enhanced(image_path)
            
            if num_faces == 0:
                return {
                    "emotion_distribution": {},
                    "emotion_spectrum": self.calculate_enhanced_emotion_spectrum({}),
                    "number_of_faces": 0,
                    "social_dynamics": self.assess_enhanced_social_dynamics(0, {}),
                    "face_quality_score": 0.0,
                    "analysis_confidence": 0.0
                }
            
            # Analyze emotions for each face
            all_face_emotions = []
            total_confidence = 0
            
            for i, face_crop in enumerate(face_crops):
                face_emotions = self.analyze_face_emotions(face_crop)
                all_face_emotions.append(face_emotions)
                if i < len(confidences):
                    total_confidence += confidences[i]
            
            # Aggregate emotions across all faces
            aggregated_emotions = self.aggregate_face_emotions(all_face_emotions)
            
            # Calculate advanced metrics
            emotion_spectrum = self.calculate_enhanced_emotion_spectrum(aggregated_emotions)
            social_dynamics = self.assess_enhanced_social_dynamics(num_faces, aggregated_emotions, confidences)
            
            # Face quality score (based on detection confidence)
            face_quality_score = (total_confidence / num_faces) if num_faces > 0 else 0.0
            
            # Analysis confidence (how certain we are about the results)
            analysis_confidence = min(1.0, face_quality_score * emotion_spectrum['emotional_intensity'])
            
            return {
                "emotion_distribution": aggregated_emotions,
                "emotion_spectrum": emotion_spectrum,
                "number_of_faces": num_faces,
                "social_dynamics": social_dynamics,
                "face_quality_score": round(face_quality_score, 3),
                "analysis_confidence": round(analysis_confidence, 3),
                "individual_faces": all_face_emotions  # For detailed analysis
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced facial emotion analysis error: {e}")
            return {"error": str(e)}

    def cleanup_gpu_memory(self):
        """Clean up GPU memory after analysis."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cleaned up")