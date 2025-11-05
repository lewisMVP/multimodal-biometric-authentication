"""
Iris Recognition Module using Traditional CV
Implements Daugman's algorithm with Hough Transform and Gabor wavelets
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, List


class IrisRecognition:
    """
    Iris recognition system using traditional computer vision
    Based on Daugman's algorithm:
    1. Segmentation (Hough Circle Transform)
    2. Normalization (Rubber sheet model)
    3. Feature extraction (Gabor wavelets)
    4. Matching (Hamming distance)
    """
    
    def __init__(self, database_path: str = "data/database/iris"):
        """
        Initialize iris recognition system
        
        Args:
            database_path: Path to store iris templates
        """
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.templates = {}
        self.optimal_threshold = 0.35  # Hamming distance threshold (1-0.35=0.65 similarity)
        
        # Normalization parameters
        self.norm_height = 64
        self.norm_width = 512
        
        # Quality assessment thresholds
        self.min_quality_score = 60  # Minimum quality to accept
        
        self.load_templates()
        print(f"✓ Iris Recognition initialized (Daugman's algorithm + Quality Assessment)")
    
    def segment_iris(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Robust iris segmentation with multiple strategies and validation
        
        Args:
            image: Grayscale eye image
            
        Returns:
            Tuple of (segmented_iris, params) or (None, None) if failed
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Try multiple parameter combinations for robust detection
            # Optimized for MMU iris database (240x320, iris ~50-70px)
            params_list = [
                {'dp': 1, 'minDist': 80, 'param1': 100, 'param2': 30},
                {'dp': 1, 'minDist': 100, 'param1': 80, 'param2': 25},
                {'dp': 1.5, 'minDist': 70, 'param1': 60, 'param2': 20},
                {'dp': 2.0, 'minDist': 50, 'param1': 50, 'param2': 18}
            ]
            
            best_params = None
            best_validation_score = -1
            
            for hp in params_list:
                # Detect iris boundary (outer circle)
                # MMU dataset: iris radius ~50-70 pixels
                circles_iris = cv2.HoughCircles(
                    enhanced, cv2.HOUGH_GRADIENT, dp=hp['dp'], minDist=hp['minDist'],
                    param1=hp['param1'], param2=hp['param2'], 
                    minRadius=45, maxRadius=80
                )
                
                if circles_iris is None:
                    continue
                
                # Get the most prominent circle
                iris_circle = circles_iris[0][0]
                ix, iy, ir = int(iris_circle[0]), int(iris_circle[1]), int(iris_circle[2])
                
                # Detect pupil (inner circle) - darker region
                pupil_region = enhanced[max(0, iy-ir):min(enhanced.shape[0], iy+ir),
                                       max(0, ix-ir):min(enhanced.shape[1], ix+ir)]
                
                if pupil_region.size == 0:
                    continue
                
                circles_pupil = cv2.HoughCircles(
                    pupil_region, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                    param1=50, param2=15, minRadius=10, maxRadius=int(ir*0.55)
                )
                
                if circles_pupil is not None:
                    pupil_circle = circles_pupil[0][0]
                    px = int(pupil_circle[0]) + max(0, ix-ir)
                    py = int(pupil_circle[1]) + max(0, iy-ir)
                    pr = int(pupil_circle[2])
                else:
                    # Default pupil if not detected
                    px, py, pr = ix, iy, int(ir * 0.4)
                
                # Validate circles
                validation_score = self._validate_circles(ix, iy, ir, px, py, pr, enhanced)
                
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    best_params = {
                        'iris_center': (ix, iy),
                        'iris_radius': ir,
                        'pupil_center': (px, py),
                        'pupil_radius': pr,
                        'validation_score': validation_score
                    }
            
            if best_params is None or best_validation_score < 0.3:
                return None, None
            
            return gray, best_params
            
        except Exception as e:
            print(f"⚠️ Iris segmentation failed: {str(e)}")
            return None, None
    
    def _validate_circles(self, ix: int, iy: int, ir: int, 
                         px: int, py: int, pr: int,
                         image: np.ndarray) -> float:
        """
        Validate detected iris and pupil circles
        
        Returns:
            Validation score (0-1), higher is better
        """
        score = 0.0
        
        # Check 1: Radius ratio (typical: 1:2 to 1:6 for MMU dataset)
        ratio = ir / pr if pr > 0 else 0
        if 1.8 <= ratio <= 6.5:
            score += 0.3
        else:
            return 0.0  # Invalid ratio
        
        # Check 2: Centers should be close (concentric)
        dx = ix - px
        dy = iy - py
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < pr * 0.3:  # Centers very close
            score += 0.3
        elif dist < pr * 0.5:  # Acceptable
            score += 0.2
        else:
            score += 0.1
        
        # Check 3: Circles should be within image bounds
        h, w = image.shape[:2]
        if (ir < ix < w - ir) and (ir < iy < h - ir):
            score += 0.2
        
        # Check 4: Pupil should be darker than iris
        try:
            # Sample pupil region
            pupil_mask = np.zeros_like(image)
            cv2.circle(pupil_mask, (px, py), pr, 255, -1)
            pupil_mean = cv2.mean(image, mask=pupil_mask)[0]
            
            # Sample iris region
            iris_mask = np.zeros_like(image)
            cv2.circle(iris_mask, (ix, iy), ir, 255, -1)
            cv2.circle(iris_mask, (px, py), pr, 0, -1)  # Remove pupil
            iris_mean = cv2.mean(image, mask=iris_mask)[0]
            
            if pupil_mean < iris_mean:  # Pupil darker
                score += 0.2
        except:
            pass
        
        return score
    
    def normalize_iris(self, image: np.ndarray, params: Dict) -> Optional[np.ndarray]:
        """
        Normalize iris region using rubber sheet model with eyelid detection
        Unwrap circular iris into rectangular form
        
        Args:
            image: Segmented eye image
            params: Segmentation parameters
            
        Returns:
            Normalized iris image
        """
        try:
            ix, iy = params['iris_center']
            ir = params['iris_radius']
            px, py = params['pupil_center']
            pr = params['pupil_radius']
            
            # Create normalized iris image
            normalized = np.zeros((self.norm_height, self.norm_width), dtype=np.uint8)
            
            # Polar to Cartesian mapping
            for theta_idx in range(self.norm_width):
                theta = 2 * np.pi * theta_idx / self.norm_width
                
                for r_idx in range(self.norm_height):
                    # Radial position between pupil and iris boundary
                    r = pr + (ir - pr) * r_idx / self.norm_height
                    
                    # Calculate Cartesian coordinates
                    x = int(px + r * np.cos(theta))
                    y = int(py + r * np.sin(theta))
                    
                    # Check bounds
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        normalized[r_idx, theta_idx] = image[y, x]
            
            return normalized
            
        except Exception as e:
            print(f"⚠️ Iris normalization failed: {str(e)}")
            return None
    
    def detect_occlusions(self, normalized_iris: np.ndarray) -> np.ndarray:
        """
        Detect eyelid and eyelash occlusions in normalized iris
        
        Args:
            normalized_iris: Normalized iris image
            
        Returns:
            Binary mask where True = occluded
        """
        h, w = normalized_iris.shape
        occlusion_mask = np.zeros((h, w), dtype=bool)
        
        # 1. Detect eyelids (horizontal edges in upper/lower regions)
        sobelx = cv2.Sobel(normalized_iris, cv2.CV_64F, 1, 0, ksize=3)
        strong_edges = np.abs(sobelx) > 50
        
        # Upper eyelid (top 30% of normalized iris)
        upper_region = int(h * 0.3)
        occlusion_mask[:upper_region, :] |= strong_edges[:upper_region, :]
        
        # Lower eyelid (bottom 30%)
        lower_region = int(h * 0.7)
        occlusion_mask[lower_region:, :] |= strong_edges[lower_region:, :]
        
        # 2. Detect eyelashes (very dark pixels)
        eyelash_threshold = 30
        eyelash_mask = normalized_iris < eyelash_threshold
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eyelash_mask = cv2.morphologyEx(eyelash_mask.astype(np.uint8), 
                                        cv2.MORPH_CLOSE, kernel).astype(bool)
        
        occlusion_mask |= eyelash_mask
        
        return occlusion_mask
    
    def extract_features(self, normalized_iris: np.ndarray) -> Optional[Dict]:
        """
        Extract iris code using Gabor wavelets with occlusion handling
        
        Args:
            normalized_iris: Normalized iris image
            
        Returns:
            Dictionary with iris code, noise mask, and quality metrics
        """
        try:
            # Detect occlusions
            occlusion_mask = self.detect_occlusions(normalized_iris)
            noise_mask = ~occlusion_mask  # Valid regions (not occluded)
            
            # Apply Gabor filter bank
            iris_code = []
            
            # Multiple orientations of Gabor filters
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21), sigma=3, theta=theta,
                    lambd=10, gamma=0.5, psi=0, ktype=cv2.CV_32F
                )
                
                # Filter normalized iris
                filtered = cv2.filter2D(normalized_iris, cv2.CV_32F, kernel)
                
                # Quantize to binary using MEDIAN (not zero!)
                # This properly encodes phase information
                threshold = np.median(filtered)
                binary_code = (filtered > threshold).astype(np.uint8)
                iris_code.append(binary_code)
            
            # Concatenate all orientations
            iris_code = np.concatenate(iris_code, axis=0)
            
            # Calculate valid ratio
            valid_ratio = noise_mask.sum() / noise_mask.size
            
            features = {
                'iris_code': iris_code,
                'noise_mask': np.tile(noise_mask, (4, 1)),  # Replicate for each orientation
                'code_shape': iris_code.shape,
                'valid_ratio': valid_ratio,
                'method': 'Gabor + Occlusion Handling'
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {str(e)}")
            return None
    
    def match_features(self, features1: Dict, features2: Dict, 
                      max_shift: int = 20) -> Tuple[float, float, int]:
        """
        Match two iris codes using Hamming distance with rotation handling
        
        Args:
            features1: First iris features
            features2: Second iris features
            max_shift: Maximum circular shift to try (handles rotation)
            
        Returns:
            Tuple of (similarity_score, hamming_distance, best_shift)
            Lower Hamming distance = more similar
        """
        code1 = features1['iris_code']
        code2 = features2['iris_code']
        
        # Get noise masks (if available)
        mask1 = features1.get('noise_mask', np.ones_like(code1, dtype=bool))
        mask2 = features2.get('noise_mask', np.ones_like(code2, dtype=bool))
        
        min_hamming = 1.0
        best_shift = 0
        
        # Try different circular shifts to handle rotation
        for shift in range(-max_shift, max_shift + 1):
            # Shift code2 and its mask
            shifted_code2 = np.roll(code2, shift, axis=1)
            shifted_mask2 = np.roll(mask2, shift, axis=1)
            
            # Combined valid mask (both must be valid)
            valid_mask = mask1 & shifted_mask2
            
            # Check if enough valid bits
            if valid_mask.sum() < code1.size * 0.1:  # Need at least 10% valid
                continue
            
            # Calculate Hamming distance on valid bits only
            xor = code1 ^ shifted_code2
            hamming = (xor & valid_mask).sum() / valid_mask.sum()
            
            if hamming < min_hamming:
                min_hamming = hamming
                best_shift = shift
        
        similarity = 1 - min_hamming
        
        return similarity, min_hamming, best_shift
    
    def assess_iris_quality(self, image: np.ndarray, 
                           segmentation_params: Optional[Dict] = None) -> Dict:
        """
        Comprehensive iris quality assessment
        
        Args:
            image: Original iris/eye image
            segmentation_params: Segmentation results (if already computed)
            
        Returns:
            Dictionary with quality scores and recommendation
        """
        quality_result = {
            'overall': 0,
            'sharpness': 0,
            'contrast': 0,
            'illumination': 0,
            'occlusion': 0,
            'segmentation': 0,
            'recommendation': 'Reject',
            'reason': ''
        }
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # If segmentation not provided, try to segment
            if segmentation_params is None:
                gray, segmentation_params = self.segment_iris(image)
                if segmentation_params is None:
                    quality_result['reason'] = 'Segmentation failed'
                    return quality_result
            
            scores = []
            weights = [0.20, 0.25, 0.20, 0.20, 0.15]
            
            # 1. Segmentation quality
            seg_score = segmentation_params.get('validation_score', 0.5) * 100
            scores.append(seg_score)
            quality_result['segmentation'] = seg_score
            
            # 2. Focus/Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_score = min(100, (sharpness / 500) * 100)
            scores.append(sharpness_score)
            quality_result['sharpness'] = sharpness_score
            
            # 3. Contrast (iris region vs pupil)
            ix, iy = segmentation_params['iris_center']
            ir = segmentation_params['iris_radius']
            px, py = segmentation_params['pupil_center']
            pr = segmentation_params['pupil_radius']
            
            # Extract iris region
            iris_mask = np.zeros_like(gray)
            cv2.circle(iris_mask, (ix, iy), ir, 255, -1)
            cv2.circle(iris_mask, (px, py), pr, 0, -1)
            iris_mean = cv2.mean(gray, mask=iris_mask)[0]
            
            # Extract pupil region
            pupil_mask = np.zeros_like(gray)
            cv2.circle(pupil_mask, (px, py), pr, 255, -1)
            pupil_mean = cv2.mean(gray, mask=pupil_mask)[0]
            
            contrast = abs(iris_mean - pupil_mean)
            contrast_score = min(100, (contrast / 80) * 100)
            scores.append(contrast_score)
            quality_result['contrast'] = contrast_score
            
            # 4. Illumination (check for specular reflections)
            bright_pixels = (gray > 240).sum()
            total_pixels = gray.size
            reflection_ratio = bright_pixels / total_pixels
            illumination_score = max(0, 100 - (reflection_ratio * 1000))
            scores.append(illumination_score)
            quality_result['illumination'] = illumination_score
            
            # 5. Occlusion analysis
            normalized = self.normalize_iris(gray, segmentation_params)
            if normalized is not None:
                occlusion_mask = self.detect_occlusions(normalized)
                occlusion_ratio = occlusion_mask.sum() / occlusion_mask.size
                occlusion_score = max(0, 100 - (occlusion_ratio * 100))
                scores.append(occlusion_score)
                quality_result['occlusion'] = occlusion_score
            else:
                scores.append(0)
            
            # Calculate overall quality
            overall = sum(s * w for s, w in zip(scores, weights))
            quality_result['overall'] = overall
            
            # Recommendation
            if overall >= 80:
                quality_result['recommendation'] = 'Excellent'
            elif overall >= 60:
                quality_result['recommendation'] = 'Accept'
            elif overall >= 40:
                quality_result['recommendation'] = 'Poor - Retry'
                quality_result['reason'] = self._get_quality_issues(quality_result)
            else:
                quality_result['recommendation'] = 'Reject'
                quality_result['reason'] = self._get_quality_issues(quality_result)
            
        except Exception as e:
            quality_result['reason'] = f'Quality assessment failed: {str(e)}'
        
        return quality_result
    
    def _get_quality_issues(self, quality_result: Dict) -> str:
        """Identify main quality issues"""
        issues = []
        if quality_result['sharpness'] < 50:
            issues.append('blurry')
        if quality_result['contrast'] < 50:
            issues.append('low contrast')
        if quality_result['illumination'] < 50:
            issues.append('poor lighting/reflections')
        if quality_result['occlusion'] < 50:
            issues.append('heavy occlusion')
        if quality_result['segmentation'] < 50:
            issues.append('poor segmentation')
        
        return ', '.join(issues) if issues else 'multiple issues'
    
    def detect_eye_side(self, iris_image: np.ndarray, params: Dict) -> str:
        """
        Detect which eye (left/right) based on pupil position relative to iris center
        
        Logic (when looking at the image):
        - If pupil is to the LEFT of iris center → LEFT eye (pupil closer to nose on left)
        - If pupil is to the RIGHT of iris center → RIGHT eye (pupil closer to nose on right)
        - If centered (within threshold) → UNKNOWN
        
        Args:
            iris_image: Eye image
            params: Segmentation parameters with iris_center and pupil_center
            
        Returns:
            "left", "right", or "unknown"
        """
        try:
            iris_x = params['iris_center'][0]
            pupil_x = params['pupil_center'][0]
            iris_radius = params['iris_radius']
            
            # Calculate horizontal offset as percentage of iris radius
            offset = pupil_x - iris_x
            offset_ratio = offset / iris_radius
            
            # Threshold for determining eye side (10% of iris radius)
            threshold = 0.1
            
            if offset_ratio < -threshold:
                # Pupil is significantly LEFT of iris center → LEFT eye
                return "left"
            elif offset_ratio > threshold:
                # Pupil is significantly RIGHT of iris center → RIGHT eye
                return "right"
            else:
                # Pupil is centered → Cannot determine
                return "unknown"
                
        except Exception as e:
            print(f"⚠️ Eye side detection failed: {e}")
            return "unknown"
    
    def enroll(self, user_id: str, iris_image: np.ndarray,
               eye_side: str = "unknown", check_quality: bool = True) -> bool:
        """
        Enroll a user's iris into the system with quality check
        Supports enrolling left and right eye separately
        
        Args:
            user_id: Unique user identifier
            iris_image: Iris/eye image
            eye_side: "left", "right", or "unknown" to specify which eye
            check_quality: Whether to perform quality assessment
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            # Validate eye_side parameter
            eye_side = eye_side.lower()
            if eye_side not in ["left", "right", "unknown"]:
                print(f"⚠️ Invalid eye_side '{eye_side}', using 'unknown'")
                eye_side = "unknown"
            
            # Load image if path is provided
            if isinstance(iris_image, str):
                iris_image = cv2.imread(iris_image, cv2.IMREAD_GRAYSCALE)
            
            if iris_image is None:
                raise ValueError("Invalid iris image")
            
            # Segment iris
            segmented, params = self.segment_iris(iris_image)
            if segmented is None:
                print("✗ Iris segmentation failed")
                return False
            
            # Detect actual eye side from image
            detected_eye = self.detect_eye_side(iris_image, params)
            
            # Validate eye side if user specified left or right
            if eye_side in ["left", "right"]:
                if detected_eye in ["left", "right"] and detected_eye != eye_side:
                    # Mismatch detected - but don't reject, just warn
                    print(f"⚠️ Eye side mismatch detected!")
                    print(f"   User selected: {eye_side.upper()} eye")
                    print(f"   Auto-detected: {detected_eye.upper()} eye")
                    print(f"   Note: Detection may be inaccurate. Proceeding with user selection.")
                elif detected_eye == "unknown":
                    print(f"ℹ️ Cannot auto-detect eye side (pupil centered)")
                    print(f"   Proceeding with user selection: {eye_side.upper()} eye")
                elif detected_eye == eye_side:
                    print(f"✓ Eye side confirmed: {eye_side.upper()} eye")
            else:
                # User selected "unknown", use detected value if available
                if detected_eye in ["left", "right"]:
                    eye_side = detected_eye
                    print(f"ℹ️ Auto-detected: {eye_side.upper()} eye")
            
            # Quality assessment
            quality = None
            if check_quality:
                quality = self.assess_iris_quality(segmented, params)
                print(f"   Quality: {quality['overall']:.1f}/100 ({quality['recommendation']})")
                
                if quality['overall'] < self.min_quality_score:
                    print(f"✗ Poor quality iris - {quality['reason']}")
                    return False
            
            # Normalize
            normalized = self.normalize_iris(segmented, params)
            if normalized is None:
                print("✗ Iris normalization failed")
                return False
            
            # Extract features
            features = self.extract_features(normalized)
            if features is None:
                print("✗ Feature extraction failed")
                return False
            
            # Check if enough valid bits
            if features['valid_ratio'] < 0.3:
                print(f"✗ Too much occlusion ({features['valid_ratio']*100:.1f}% valid)")
                return False
            
            # Get or create user template
            if user_id not in self.templates:
                self.templates[user_id] = {
                    'eyes': {},
                    'enrolled_date': str(np.datetime64('now'))
                }
            
            # Store template for specific eye
            self.templates[user_id]['eyes'][eye_side] = {
                'features': features,
                'params': params,
                'quality': quality,
                'enrolled_date': str(np.datetime64('now'))
            }
            
            # Save to disk
            self.save_templates()
            
            eye_label = eye_side.capitalize() if eye_side != "unknown" else "Unknown"
            enrolled_eyes = list(self.templates[user_id]['eyes'].keys())
            print(f"✓ User {user_id} enrolled successfully ({eye_label} eye, {features['valid_ratio']*100:.1f}% valid bits)")
            print(f"   Enrolled eyes: {', '.join([e.capitalize() for e in enrolled_eyes])}")
            return True
            
        except Exception as e:
            print(f"✗ Iris enrollment failed: {str(e)}")
            return False
    
    def verify(self, user_id: str, iris_image: np.ndarray,
               threshold: Optional[float] = None,
               eye_side: str = "auto") -> Tuple[bool, float]:
        """
        Verify if iris matches the claimed user (1:1 matching)
        Supports verification with any enrolled eye (auto-detect best match)
        
        Args:
            user_id: Claimed user ID
            iris_image: Iris to verify
            threshold: Verification threshold
            eye_side: "left", "right", "auto" (checks all enrolled eyes)
            
        Returns:
            Tuple of (is_verified, confidence_score)
        """
        if threshold is None:
            threshold = 1 - self.optimal_threshold  # Convert to similarity
        
        try:
            # Check if user exists
            if user_id not in self.templates:
                print(f"✗ User {user_id} not found in database")
                return False, 0.0
            
            # Check if user has any enrolled eyes
            if not self.templates[user_id]['eyes']:
                print(f"✗ User {user_id} has no enrolled eyes")
                return False, 0.0
            
            # Load image if path is provided
            if isinstance(iris_image, str):
                iris_image = cv2.imread(iris_image, cv2.IMREAD_GRAYSCALE)
            
            if iris_image is None:
                raise ValueError("Invalid iris image")
            
            # Process iris
            segmented, params = self.segment_iris(iris_image)
            if segmented is None:
                return False, 0.0
            
            normalized = self.normalize_iris(segmented, params)
            if normalized is None:
                return False, 0.0
            
            features = self.extract_features(normalized)
            if features is None:
                return False, 0.0
            
            # Validate eye_side parameter
            eye_side = eye_side.lower()
            enrolled_eyes = list(self.templates[user_id]['eyes'].keys())
            
            # Determine which eyes to check
            if eye_side == "auto":
                eyes_to_check = enrolled_eyes
            elif eye_side in enrolled_eyes:
                eyes_to_check = [eye_side]
            else:
                print(f"✗ {eye_side.capitalize()} eye not enrolled for {user_id}")
                print(f"   Available: {', '.join([e.capitalize() for e in enrolled_eyes])}")
                return False, 0.0
            
            # Compare with all specified eyes, take best match
            best_similarity = 0.0
            best_eye = None
            
            for eye in eyes_to_check:
                stored_features = self.templates[user_id]['eyes'][eye]['features']
                similarity, hamming_dist, shift = self.match_features(features, stored_features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_eye = eye
            
            is_verified = best_similarity >= threshold
            
            if is_verified:
                print(f"✓ Verified as {user_id} ({best_eye} eye, similarity: {best_similarity:.3f})")
            else:
                print(f"✗ Not verified (best: {best_similarity:.3f} < {threshold:.3f})")
            
            return is_verified, best_similarity
            
        except Exception as e:
            print(f"✗ Iris verification failed: {str(e)}")
            return False, 0.0
    
    def identify(self, iris_image: np.ndarray,
                 threshold: Optional[float] = None,
                 top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Identify user from iris (1:N matching)
        Compares against all enrolled eyes from all users
        
        Args:
            iris_image: Iris to identify
            threshold: Minimum matching threshold
            top_n: Return top N matches
            
        Returns:
            List of (user_id, confidence_score) tuples, sorted by score
        """
        if threshold is None:
            threshold = 1 - self.optimal_threshold
        
        try:
            # Load image if path is provided
            if isinstance(iris_image, str):
                iris_image = cv2.imread(iris_image, cv2.IMREAD_GRAYSCALE)
            
            if iris_image is None:
                raise ValueError("Invalid iris image")
            
            # Process iris
            segmented, params = self.segment_iris(iris_image)
            if segmented is None:
                return []
            
            normalized = self.normalize_iris(segmented, params)
            if normalized is None:
                return []
            
            features = self.extract_features(normalized)
            if features is None:
                return []
            
            # Match against all templates (all eyes from all users)
            results = []
            for user_id, user_data in self.templates.items():
                # Check all enrolled eyes for this user
                best_similarity = 0.0
                best_eye = None
                
                for eye_side, eye_data in user_data['eyes'].items():
                    stored_features = eye_data['features']
                    similarity, _, _ = self.match_features(features, stored_features)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_eye = eye_side
                
                # Only add if above threshold
                if best_similarity >= threshold:
                    results.append((user_id, best_similarity))
                    print(f"   {user_id}: {best_similarity:.3f} ({best_eye} eye)")
            
            # Sort by score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_n]
            
        except Exception as e:
            print(f"✗ Iris identification failed: {str(e)}")
            return []
    
    def save_templates(self):
        """Save iris templates to disk"""
        template_file = self.database_path / "templates.pkl"
        with open(template_file, 'wb') as f:
            pickle.dump(self.templates, f)
    
    def load_templates(self):
        """Load iris templates from disk with migration support"""
        template_file = self.database_path / "templates.pkl"
        if template_file.exists():
            try:
                with open(template_file, 'rb') as f:
                    loaded_templates = pickle.load(f)
                
                # Migrate old format to new format
                migrated = False
                for user_id, template in loaded_templates.items():
                    # Check if old format (has 'features' key directly)
                    if 'features' in template and 'eyes' not in template:
                        # Migrate to new format
                        loaded_templates[user_id] = {
                            'eyes': {
                                'unknown': {
                                    'features': template['features'],
                                    'params': template.get('params'),
                                    'quality': template.get('quality'),
                                    'enrolled_date': template.get('enrolled_date', str(np.datetime64('now')))
                                }
                            },
                            'enrolled_date': template.get('enrolled_date', str(np.datetime64('now')))
                        }
                        migrated = True
                
                self.templates = loaded_templates
                
                # Count total eyes
                total_eyes = sum(len(user['eyes']) for user in self.templates.values())
                
                if migrated:
                    print(f"✓ Migrated {len(self.templates)} users to new format (multi-eye support)")
                    self.save_templates()  # Save migrated format
                
                print(f"✓ Loaded {len(self.templates)} users with {total_eyes} eyes")
                
            except Exception as e:
                print(f"⚠️ Could not load templates: {e}")
                self.templates = {}
        else:
            self.templates = {}
            print("✓ Initialized empty iris template database")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        total_eyes = sum(len(user['eyes']) for user in self.templates.values())
        
        # Count by eye type
        eye_counts = {'left': 0, 'right': 0, 'unknown': 0}
        for user in self.templates.values():
            for eye_side in user['eyes'].keys():
                eye_counts[eye_side] = eye_counts.get(eye_side, 0) + 1
        
        return {
            'total_users': len(self.templates),
            'total_eyes': total_eyes,
            'eyes_breakdown': eye_counts,
            'method': 'Daugman (Hough + Gabor)',
            'database_path': str(self.database_path)
        }
    
    def list_enrolled_users(self) -> List[str]:
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


