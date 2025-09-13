import numpy as np
from PIL import Image

class CompositionAnalysis:
    def __init__(self):
        pass

    def analyze_symmetry(self, img_array):
        """
        Analyze horizontal symmetry.
        """
        height, width = img_array.shape[:2]
        mid = width // 2
        left = img_array[:, :mid]
        right = img_array[:, -mid:]
        if width % 2 == 1:
            right = right[:, 1:]
        right_flipped = np.fliplr(right)
        
        diff = np.mean(np.abs(left.astype(float) - right_flipped.astype(float)))
        max_diff = 255 * 3  # RGB
        symmetry_score = 1 - (diff / max_diff)
        return max(0, min(1, symmetry_score))

    def analyze_balance(self, img_array):
        """
        Analyze balance using center of mass.
        """
        gray = np.mean(img_array.astype(float), axis=2)
        height, width = gray.shape
        total_mass = np.sum(gray)
        if total_mass == 0:
            return 0.5
        
        y_coords, x_coords = np.mgrid[:height, :width]
        center_y = np.sum(y_coords * gray) / total_mass
        center_x = np.sum(x_coords * gray) / total_mass
        
        dist_from_center = abs(center_y - height/2) + abs(center_x - width/2)
        max_dist = height/2 + width/2
        balance_score = 1 - (dist_from_center / max_dist)
        return max(0, min(1, balance_score))

    def analyze_rule_of_thirds(self, img_array):
        """
        Placeholder for rule of thirds analysis.
        """
        # In a real implementation, detect focal points and check alignment
        return 0.8  # Dummy score

    def analyze(self, image_path):
        """
        Full composition analysis.
        """
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        symmetry_score = self.analyze_symmetry(img_array)
        balance_score = self.analyze_balance(img_array)
        rule_of_thirds_score = self.analyze_rule_of_thirds(img_array)
        
        overall_aesthetic_score = (symmetry_score + balance_score + rule_of_thirds_score) / 3
        
        return {
            "symmetry_score": symmetry_score,
            "balance_score": balance_score,
            "rule_of_thirds_score": rule_of_thirds_score,
            "overall_aesthetic_score": overall_aesthetic_score
        }