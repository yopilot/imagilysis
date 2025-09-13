from PIL import Image
import numpy as np
from scipy.cluster.vq import kmeans
import matplotlib.pyplot as plt
import os

class ColorAnalysis:
    def __init__(self):
        pass

    def extract_dominant_colors(self, image_path, n_colors=5):
        """
        Extract dominant colors using K-means clustering.
        """
        img = Image.open(image_path)
        img = img.resize((150, 150))  # Resize for efficiency
        img_array = np.array(img)
        if img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA to RGB
            img_array = img_array[:, :, :3]
        pixels = img_array.reshape(-1, 3).astype(float)
        
        centroids, _ = kmeans(pixels, n_colors)
        dominant_colors = centroids.astype(int)
        return dominant_colors

    def interpret_mood_and_traits(self, colors):
        """
        Simple mood interpretation based on color hues.
        This is a placeholder; in a real implementation, use more sophisticated mapping.
        """
        # Calculate average hue
        hsv_colors = []
        for color in colors:
            r, g, b = color / 255.0
            hsv = self._rgb_to_hsv(r, g, b)
            hsv_colors.append(hsv)
        
        avg_hue = np.mean([h for h, s, v in hsv_colors])
        
        if avg_hue < 0.1 or avg_hue > 0.9:  # Red
            mood = "energetic"
            traits = ["passionate", "bold"]
        elif 0.1 <= avg_hue < 0.3:  # Yellow/Orange
            mood = "happy"
            traits = ["optimistic", "friendly"]
        elif 0.3 <= avg_hue < 0.5:  # Green
            mood = "calm"
            traits = ["balanced", "nature-loving"]
        elif 0.5 <= avg_hue < 0.7:  # Cyan
            mood = "serene"
            traits = ["peaceful", "introspective"]
        elif 0.7 <= avg_hue < 0.9:  # Blue
            mood = "sad"
            traits = ["melancholic", "thoughtful"]
        else:
            mood = "neutral"
            traits = ["practical", "reliable"]
        
        return {"mood": mood, "traits": traits}

    def _rgb_to_hsv(self, r, g, b):
        """
        Convert RGB to HSV.
        """
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        if max_c == min_c:
            h = 0
        elif max_c == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_c == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        elif max_c == b:
            h = (60 * ((r - g) / diff) + 240) % 360
        h /= 360  # Normalize to 0-1
        s = 0 if max_c == 0 else diff / max_c
        v = max_c
        return h, s, v

    def generate_palette_visual(self, colors, output_path):
        """
        Generate a visual palette image.
        """
        fig, ax = plt.subplots(1, len(colors), figsize=(len(colors)*2, 2))
        if len(colors) == 1:
            ax = [ax]
        for i, color in enumerate(colors):
            ax[i].imshow([[color / 255.0]])
            ax[i].axis('off')
        plt.savefig(output_path)
        plt.close()

    def analyze(self, image_path):
        """
        Full analysis: extract colors, interpret mood, generate visual.
        """
        colors = self.extract_dominant_colors(image_path)
        mood_traits = self.interpret_mood_and_traits(colors)
        palette_path = os.path.join("data", "palette.png")
        self.generate_palette_visual(colors, palette_path)
        return {
            "dominant_colors": colors.tolist(),
            "mood": mood_traits["mood"],
            "personality_traits": mood_traits["traits"],
            "palette_visual": palette_path
        }