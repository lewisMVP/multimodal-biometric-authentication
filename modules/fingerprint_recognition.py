"""
Fingerprint Recognition Module
Handles fingerprint enrollment, verification and identification
"""

import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, List


class FingerprintRecognition:
    """
    Fingerprint recognition system using minutiae-based matching
    or deep learning features
    """
    
    def __init__(self, database_path: str = "data/database/fingerprints"):
        """
        Initialize fingerprint recognition system
        
        Args:
            database_path: Path to store fingerprint templates
        """
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.templates = {}
        self.optimal_threshold = 0.3  # Can be updated based on EER analysis
        self.load_templates()
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess fingerprint image using advanced CLAHE method
        - Convert to grayscale
        - CLAHE for adaptive contrast enhancement
        - Normalize
        - Denoise
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Better than regular histogram equalization for fingerprints
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Final normalization
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract fingerprint features
        
        Options:
        1. Minutiae-based: Extract ridge endings and bifurcations
        2. Deep learning: Use pre-trained CNN
        3. SIFT/ORB descriptors
        
        Args:
            image: Preprocessed fingerprint image
            
        Returns:
            Dictionary containing feature descriptors and keypoints
        """
        # Method: Using SIFT (Scale-Invariant Feature Transform)
        # SIFT provides better accuracy than ORB for fingerprint matching
        sift = cv2.SIFT_create(nfeatures=500)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        # Convert keypoints to serializable format
        keypoints_data = [
            {
                'pt': kp.pt,
                'size': kp.size,
                'angle': kp.angle,
                'response': kp.response,
                'octave': kp.octave
            }
            for kp in keypoints
        ]
        
        features = {
            'keypoints': keypoints_data,
            'descriptors': descriptors,
            'method': 'SIFT'
        }
        
        return features
    
    def match_features(self, features1: Dict, features2: Dict, 
                      threshold: float = 0.75) -> Tuple[float, int]:
        """
        Match two fingerprint feature sets
        
        Args:
            features1: First feature set
            features2: Second feature set
            threshold: Matching threshold (lower = stricter)
            
        Returns:
            Tuple of (match_score, number_of_matches)
        """
        desc1 = features1['descriptors']
        desc2 = features2['descriptors']
        
        if desc1 is None or desc2 is None:
            return 0.0, 0
        
        # Use FLANN matcher with L2 norm for SIFT (float descriptors)
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(desc1, desc2, k=2)
        except:
            return 0.0, 0
        
        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
        
        # Calculate match score
        num_matches = len(good_matches)
        if num_matches == 0:
            return 0.0, 0
        
        # Normalize score by total keypoints
        total_keypoints = min(len(desc1), len(desc2))
        match_score = num_matches / total_keypoints if total_keypoints > 0 else 0.0
        
        return match_score, num_matches
    
    def enroll(self, user_id: str, fingerprint_image: np.ndarray) -> bool:
        """
        Enroll a user's fingerprint into the system
        
        Args:
            user_id: Unique user identifier
            fingerprint_image: Fingerprint image (can be path or numpy array)
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            # Load image if path is provided
            if isinstance(fingerprint_image, str):
                fingerprint_image = cv2.imread(fingerprint_image, cv2.IMREAD_GRAYSCALE)
            
            if fingerprint_image is None:
                raise ValueError("Invalid fingerprint image")
            
            # Preprocess
            preprocessed = self.preprocess(fingerprint_image)
            
            # Extract features
            features = self.extract_features(preprocessed)
            
            # Store template
            self.templates[user_id] = {
                'features': features,
                'enrolled_date': np.datetime64('now')
            }
            
            # Save to disk
            self.save_templates()
            
            print(f"✓ User {user_id} enrolled successfully")
            return True
            
        except Exception as e:
            print(f"✗ Enrollment failed: {str(e)}")
            return False
    
    def verify(self, user_id: str, fingerprint_image: np.ndarray, 
               threshold: float = 0.3) -> Tuple[bool, float]:
        """
        Verify if fingerprint matches the claimed user (1:1 matching)
        
        Args:
            user_id: Claimed user ID
            fingerprint_image: Fingerprint to verify
            threshold: Verification threshold
            
        Returns:
            Tuple of (is_verified, confidence_score)
        """
        try:
            # Check if user exists
            if user_id not in self.templates:
                print(f"✗ User {user_id} not found in database")
                return False, 0.0
            
            # Load image if path is provided
            if isinstance(fingerprint_image, str):
                fingerprint_image = cv2.imread(fingerprint_image, cv2.IMREAD_GRAYSCALE)
            
            if fingerprint_image is None:
                raise ValueError("Invalid fingerprint image")
            
            # Preprocess and extract features
            preprocessed = self.preprocess(fingerprint_image)
            features = self.extract_features(preprocessed)
            
            # Match with stored template
            stored_features = self.templates[user_id]['features']
            match_score, num_matches = self.match_features(features, stored_features)
            
            is_verified = match_score >= threshold
            
            return is_verified, match_score
            
        except Exception as e:
            print(f"✗ Verification failed: {str(e)}")
            return False, 0.0
    
    def identify(self, fingerprint_image: np.ndarray, 
                 threshold: float = 0.3, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify user from fingerprint (1:N matching)
        
        Args:
            fingerprint_image: Fingerprint to identify
            threshold: Minimum matching threshold
            top_n: Return top N matches
            
        Returns:
            List of (user_id, confidence_score) tuples, sorted by score
        """
        try:
            # Load image if path is provided
            if isinstance(fingerprint_image, str):
                fingerprint_image = cv2.imread(fingerprint_image, cv2.IMREAD_GRAYSCALE)
            
            if fingerprint_image is None:
                raise ValueError("Invalid fingerprint image")
            
            # Preprocess and extract features
            preprocessed = self.preprocess(fingerprint_image)
            features = self.extract_features(preprocessed)
            
            # Match against all templates
            results = []
            for user_id, template in self.templates.items():
                stored_features = template['features']
                match_score, _ = self.match_features(features, stored_features)
                
                if match_score >= threshold:
                    results.append((user_id, match_score))
            
            # Sort by score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_n]
            
        except Exception as e:
            print(f"✗ Identification failed: {str(e)}")
            return []
    
    def save_templates(self):
        """Save fingerprint templates to disk"""
        template_file = self.database_path / "templates.pkl"
        with open(template_file, 'wb') as f:
            pickle.dump(self.templates, f)
    
    def load_templates(self):
        """Load fingerprint templates from disk"""
        template_file = self.database_path / "templates.pkl"
        if template_file.exists():
            with open(template_file, 'rb') as f:
                self.templates = pickle.load(f)
            print(f"✓ Loaded {len(self.templates)} fingerprint templates")
        else:
            self.templates = {}
            print("✓ Initialized empty template database")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            'total_users': len(self.templates),
            'database_path': str(self.database_path)
        }
    
    def list_enrolled_users(self) -> list:
        """Get list of all enrolled user IDs"""
        return list(self.templates.keys())
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user from the database
        
        Args:
            user_id: User identifier to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if user_id not in self.templates:
            print(f"⚠️ User {user_id} not found in database")
            return False
        
        # Remove from memory
        del self.templates[user_id]
        
        # Save updated database
        self.save_templates()
        
        print(f"✅ User {user_id} deleted successfully")
        return True
    
    def clear_database(self) -> bool:
        """
        Clear entire database (remove all users)
        
        Returns:
            True if cleared successfully
        """
        user_count = len(self.templates)
        self.templates = {}
        self.save_templates()
        
        print(f"✅ Database cleared. {user_count} users removed.")
        return True


