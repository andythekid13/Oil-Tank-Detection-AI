import cv2
import numpy as np
from PIL import Image
import logging
import math
import torch
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForSequenceClassification

class SatelliteImageAnalyzer:
    def __init__(self):
        self.min_diameter = 20
        self.max_diameter = 200
        self.height_to_diameter_ratio = 0.6
        self.pixels_to_meters = 0.5

        # Initialize without the vision model for now
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        
        logging.info("Initialized analyzer with basic capabilities")

    def calculate_confidence(self, roi):
        """Calculate confidence score based on image features"""
        try:
            if roi.size == 0:
                return 0.5
                
            contrast = (np.max(roi) - np.min(roi)) / 255
            std_dev = np.std(roi) / 255
            confidence = (contrast * 0.6 + std_dev * 0.4)
            
            return f"{min(max(confidence, 0.3), 0.95):.2%}"
            
        except Exception as e:
            logging.error(f"Error calculating confidence: {str(e)}")
            return "50.00%"

    def calculate_tank_volume(self, diameter_pixels):
        """Calculate the approximate volume of a cylindrical tank"""
        try:
            diameter_meters = diameter_pixels * self.pixels_to_meters
            height_meters = diameter_meters * self.height_to_diameter_ratio
            radius_meters = diameter_meters / 2
            volume = math.pi * (radius_meters ** 2) * height_meters
            volume_barrels = volume * 6.28981
            
            return {
                "volume_cubic_meters": round(volume, 2),
                "volume_barrels": round(volume_barrels, 2),
                "diameter_meters": round(diameter_meters, 2),
                "height_meters": round(height_meters, 2)
            }
        except Exception as e:
            logging.error(f"Error calculating volume: {str(e)}")
            return {
                "volume_cubic_meters": 0,
                "volume_barrels": 0,
                "diameter_meters": 0,
                "height_meters": 0
            }

    def detect_tanks(self, image):
        """Detect circular tanks in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=self.min_diameter // 2,
            maxRadius=self.max_diameter // 2
        )
        
        detected_tanks = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y, r = int(i[0]), int(i[1]), int(i[2])
                roi = gray[max(0, y-r):min(gray.shape[0], y+r), 
                         max(0, x-r):min(gray.shape[1], x+r)]
                
                tank = {
                    'x': x,
                    'y': y,
                    'radius': r,
                    'diameter_pixels': r * 2,
                    'confidence': self.calculate_confidence(roi),
                    'position': f"x: {x}, y: {y}",
                    'volume_info': self.calculate_tank_volume(r * 2)
                }
                detected_tanks.append(tank)
        
        return detected_tanks

    def analyze_image(self, image_path):
        """Analyze image for oil tanks"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            max_dimension = 800
            height, width = image.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
            else:
                scale = 1.0
            
            tanks = self.detect_tanks(image)
            confidences = [float(tank['confidence'].strip('%')) / 100 for tank in tanks]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                "tanks": tanks,
                "image_scale": scale,
                "confidence_score": avg_confidence
            }
            
        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}")
            return None

    def detect_industrial_features(self, image_path):
        """Main detection method with structured output including oil value calculations"""
        try:
            results = self.analyze_image(image_path)
            if not results:
                return {"error": "Analysis failed"}
            
            results["image_path"] = image_path
            tanks_data = []
            total_value = 0
            current_oil_price = 80.73
            
            for i, tank in enumerate(results["tanks"], 1):
                volume_barrels = tank["volume_info"]["volume_barrels"]
                total_value += volume_barrels * current_oil_price
                
                tanks_data.append({
                    "tank_number": i,
                    "diameter_pixels": tank["diameter_pixels"],
                    "diameter_meters": tank["volume_info"]["diameter_meters"],
                    "volume_barrels": volume_barrels,
                    "position": tank["position"],
                    "confidence": tank["confidence"],
                    "oil_value_usd": volume_barrels * current_oil_price
                })
            
            total_volume = sum(tank["volume_barrels"] for tank in tanks_data)
            
            return {
                "analysis_type": "Oil Tank Detection",
                "tanks_data": tanks_data,
                "image_path": image_path,
                "metrics": {
                    "total_tanks": len(tanks_data),
                    "total_capacity_barrels": total_volume,
                    "total_oil_value": total_value,
                    "price_per_barrel": current_oil_price,
                    "confidence_score": f"{results['confidence_score']:.2%}"
                }
            }
            
        except Exception as e:
            logging.error(f"Error in feature detection: {str(e)}")
            return {"error": str(e)}

    def visualize_detections(self, image_path):
        """Create visualization of detected tanks"""
        try:
            image = cv2.imread(image_path)
            results = self.analyze_image(image_path)
            
            if results and "tanks" in results:
                for tank in results["tanks"]:
                    # Draw the circle
                    cv2.circle(image, (tank['x'], tank['y']), tank['radius'], (0, 255, 0), 2)
                    # Draw center point
                    cv2.circle(image, (tank['x'], tank['y']), 2, (0, 0, 255), 3)
                    # Add volume information
                    volume_text = f"{tank['volume_info']['volume_barrels']:.0f} bbl"
                    value_text = f"${tank['volume_info']['volume_barrels'] * 80.73:,.0f}"
                    
                    # Add text annotations
                    cv2.putText(image, 
                              volume_text,
                              (tank['x'] - 30, tank['y'] + tank['radius'] + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(image,
                              value_text,
                              (tank['x'] - 30, tank['y'] + tank['radius'] + 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(image,
                              tank['confidence'],
                              (tank['x'] - 20, tank['y'] - tank['radius'] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            output_path = image_path.replace('.', '_analyzed.')
            cv2.imwrite(output_path, image)
            return output_path
            
        except Exception as e:
            logging.error(f"Error creating visualization: {str(e)}")
            return None