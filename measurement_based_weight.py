#!/usr/bin/env python3
"""
Measurement-Based Weight Estimation
Uses computer vision to extract girth and length measurements,
then applies the formula: Weight = (Girth² × Length) ÷ 300
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

class CattleMeasurementExtractor:
    """Extract girth and length measurements from cattle images"""
    
    def __init__(self):
        self.reference_object = None  # For scale reference
        self.pixels_per_cm = None     # Scale factor
        
    def detect_cattle_contour(self, image):
        """Detect cattle contour in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the cattle)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour
        return None
    
    def measure_girth(self, contour, image):
        """Estimate girth (circumference) from contour"""
        if contour is None:
            return None
        
        # Get contour moments
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        
        # Find centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Find the widest part (girth)
        # This is a simplified approach - in reality, you'd need more sophisticated methods
        hull = cv2.convexHull(contour)
        
        # Calculate approximate girth using perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Convert to real-world measurements (this is where you need scale reference)
        if self.pixels_per_cm:
            girth_cm = perimeter / self.pixels_per_cm
        else:
            # Without scale reference, return pixel measurements
            girth_cm = perimeter
        
        return girth_cm
    
    def measure_length(self, contour, image):
        """Estimate length from contour"""
        if contour is None:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Length is typically the longer dimension
        length_pixels = max(w, h)
        
        # Convert to real-world measurements
        if self.pixels_per_cm:
            length_cm = length_pixels / self.pixels_per_cm
        else:
            length_cm = length_pixels
        
        return length_cm
    
    def set_scale_reference(self, reference_length_cm, reference_pixels):
        """Set scale reference for real-world measurements"""
        self.pixels_per_cm = reference_pixels / reference_length_cm
        print(f"Scale set: {self.pixels_per_cm:.2f} pixels per cm")
    
    def calculate_weight(self, girth_cm, length_cm, k=300):
        """Calculate weight using the formula: Weight = (Girth² × Length) ÷ K"""
        if girth_cm is None or length_cm is None:
            return None
        
        weight = (girth_cm ** 2 * length_cm) / k
        return weight
    
    def process_image(self, image_path):
        """Process a single cattle image to extract measurements and calculate weight"""
        print(f"🖼️ Processing: {Path(image_path).name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, "Could not load image"
        
        # Detect cattle contour
        contour = self.detect_cattle_contour(image)
        if contour is None:
            return None, "Could not detect cattle in image"
        
        # Extract measurements
        girth = self.measure_girth(contour, image)
        length = self.measure_length(contour, image)
        
        if girth is None or length is None:
            return None, "Could not extract measurements"
        
        # Calculate weight using formula
        weight = self.calculate_weight(girth, length)
        
        return {
            'image_name': Path(image_path).name,
            'girth_cm': round(girth, 1),
            'length_cm': round(length, 1),
            'calculated_weight_kg': round(weight, 1) if weight else None,
            'formula': f"({girth:.1f}² × {length:.1f}) ÷ 300 = {weight:.1f} kg"
        }, None

def main():
    """Demonstrate measurement-based weight estimation"""
    print("🐄 MEASUREMENT-BASED WEIGHT ESTIMATION")
    print("=" * 50)
    print("Formula: Weight = (Girth² × Length) ÷ 300")
    print()
    
    # Initialize extractor
    extractor = CattleMeasurementExtractor()
    
    # Example: Set scale reference (you need to provide this)
    # This is crucial for real-world measurements
    print("⚠️ IMPORTANT: You need to set scale reference for accurate measurements!")
    print("Example: If you know a 10cm object appears as 100 pixels in your image:")
    print("extractor.set_scale_reference(10, 100)")
    print()
    
    # For demonstration, let's assume a scale (you would measure this in reality)
    # extractor.set_scale_reference(10, 100)  # 10cm = 100 pixels
    
    # Process sample images
    sample_images = [
        "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images/1_side_450_M.jpg",
        "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images/2_side_380_F.jpg"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            result, error = extractor.process_image(image_path)
            
            if error:
                print(f"❌ {Path(image_path).name}: {error}")
            else:
                print(f"✅ {result['image_name']}:")
                print(f"   Girth: {result['girth_cm']} cm")
                print(f"   Length: {result['length_cm']} cm")
                print(f"   Weight: {result['calculated_weight_kg']} kg")
                print(f"   Formula: {result['formula']}")
                print()
        else:
            print(f"❌ Image not found: {image_path}")
    
    print("💡 NOTE: This is a simplified example.")
    print("For accurate measurements, you need:")
    print("1. Scale reference in the image")
    print("2. Proper cattle pose (side view)")
    print("3. Good lighting and contrast")
    print("4. More sophisticated measurement algorithms")

if __name__ == "__main__":
    main()
