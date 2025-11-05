"""
Face Recognition Module using DeepFace
Based on development from 02_face_development.ipynb
Uses DeepFace.verify() directly for simplicity and accuracy
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datetime import datetime

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not installed. Install with: pip install deepface retina-face")


class FaceRecognition:
    """
    Face recognition system using DeepFace library
    
    Simple approach: Store face images and use DeepFace.verify() for comparison
    This matches the notebook implementation exactly
    """
    
    def __init__(self, database_path: str = "data/database/faces", 
                 model_name: str = "VGG-Face",
                 detector_backend: str = "retinaface"):
        """
        Initialize face recognition system
        
        Args:
            database_path: Path to store face images
            model_name: Pretrained model to use
                Options: "VGG-Face", "Facenet", "ArcFace", "OpenFace", "Facenet512"
                Default: "VGG-Face" (as used in notebook)
            detector_backend: Face detector to use
                Options: "opencv", "retinaface", "mtcnn"
                Default: "retinaface" (as used in notebook)
        """
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.users = {}  # Store user info: {user_id: {'image_path': path, 'enrolled_date': date}}
        
        # Model-specific thresholds (from DeepFace documentation)
        self.model_thresholds = {
            'VGG-Face': 0.68,
            'Facenet': 0.40,
            'Facenet512': 0.30,
            'ArcFace': 0.68,
            'OpenFace': 0.10,
        }
        self.optimal_threshold = self.model_thresholds.get(model_name, 0.68)
        
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace is required. Install with: pip install deepface retina-face")
        
        self.load_database()
        print(f"✓ Face Recognition initialized")
        print(f"   Model: {model_name}")
        print(f"   Detector: {detector_backend}")
        print(f"   Threshold: {self.optimal_threshold}")
    
    def extract_faces(self, image, enforce_detection: bool = True) -> Optional[List[Dict]]:
        """
        Extract faces from image using DeepFace (as in notebook)
        
        Args:
            image: Input image (numpy array or file path)
            enforce_detection: Raise error if no face detected
            
        Returns:
            List of face objects with 'face', 'facial_area', 'confidence'
        """
        try:
            # DeepFace expects BGR for numpy arrays, but Streamlit/PIL gives RGB
            # Convert RGB to BGR if needed
            if isinstance(image, np.ndarray):
                # Check if it's likely RGB (from PIL/Streamlit upload)
                # DeepFace will convert it back internally, but face crops will be in original color space
                # So we need to keep it as RGB
                input_image = image
            else:
                input_image = image
            
            face_objs = DeepFace.extract_faces(
                img_path=input_image,
                detector_backend=self.detector_backend,
                enforce_detection=enforce_detection
            )
            
            # CRITICAL FIX: DeepFace.extract_faces returns face in BGR when input is numpy array!
            # Convert BGR back to RGB for display
            if face_objs and isinstance(image, np.ndarray):
                for face_obj in face_objs:
                    if 'face' in face_obj:
                        face_bgr = face_obj['face']
                        
                        # Ensure it's uint8 before conversion
                        if face_bgr.dtype != np.uint8:
                            # If normalized (0-1), scale to 0-255
                            if face_bgr.max() <= 1.0:
                                face_bgr = (face_bgr * 255).astype(np.uint8)
                            else:
                                face_bgr = face_bgr.astype(np.uint8)
                        
                        # Convert BGR to RGB
                        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Ensure output is uint8
                        if face_rgb.dtype != np.uint8:
                            face_rgb = face_rgb.astype(np.uint8)
                        
                        face_obj['face'] = face_rgb
            
            return face_objs if face_objs else None
        except Exception as e:
            if enforce_detection:
                print(f"⚠️ Face detection failed: {str(e)}")
                raise
            return None
    
    def enroll(self, user_id: str, face_image) -> bool:
        """
        Enroll a user by saving their face image
        
        Args:
            user_id: Unique user identifier
            face_image: Face image (numpy array or file path)
            
        Returns:
            True if enrollment successful
        """
        try:
            # Load image if path provided
            if isinstance(face_image, str):
                if not os.path.exists(face_image):
                    raise ValueError(f"Image file not found: {face_image}")
                img = cv2.imread(face_image)
            else:
                img = face_image
            
            if img is None:
                raise ValueError("Invalid face image")
            
            # CRITICAL: Ensure image is uint8 (not float64)
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    # If normalized (0-1), scale to 0-255
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                    print(f"   ⚠️ Converted image dtype to uint8 (was {face_image.dtype if hasattr(face_image, 'dtype') else 'unknown'})")
            
            # Verify face can be detected
            face_objs = self.extract_faces(img, enforce_detection=True)
            if not face_objs:
                print(f"✗ No face detected in image")
                return False
            
            print(f"   ✓ Detected {len(face_objs)} face(s)")
            
            # Create user directory
            user_folder = self.database_path / user_id
            user_folder.mkdir(exist_ok=True)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{user_id}_{timestamp}.jpg"
            image_path = user_folder / image_filename
            cv2.imwrite(str(image_path), img)
            
            # Store user info
            self.users[user_id] = {
                'image_path': str(image_path),
                'enrolled_date': str(datetime.now()),
                'model': self.model_name,
                'detector': self.detector_backend
            }
            
            # Save database
            self.save_database()
            
            print(f"✓ User '{user_id}' enrolled successfully")
            print(f"   Image saved: {image_path}")
            return True
            
        except Exception as e:
            print(f"✗ Enrollment failed: {str(e)}")
            return False
    
    def verify(self, user_id: str, face_image, 
               threshold: Optional[float] = None) -> Tuple[bool, float, Dict]:
        """
        Verify if face matches the enrolled user (using DeepFace.verify as in notebook)
        
        Args:
            user_id: User ID to verify against
            face_image: Face image to verify (numpy array or path)
            threshold: Custom threshold (optional)
            
        Returns:
            Tuple of (is_verified, confidence, details_dict)
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        try:
            # Check if user exists
            if user_id not in self.users:
                print(f"✗ User '{user_id}' not found in database")
                return False, 0.0, {'error': 'User not found'}
            
            # Get stored image path
            stored_image_path = self.users[user_id]['image_path']
            
            if not os.path.exists(stored_image_path):
                print(f"✗ Stored image not found: {stored_image_path}")
                return False, 0.0, {'error': 'Stored image not found'}
            
            # Use DeepFace.verify() directly (as in notebook)
            result = DeepFace.verify(
                img1_path=face_image,
                img2_path=stored_image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            # Extract results
            verified = result.get('verified', False)
            distance = result.get('distance', 1.0)
            model_threshold = result.get('threshold', threshold)
            metric = result.get('similarity_metric', 'cosine')
            
            # Calculate confidence (inverse of distance, normalized)
            confidence = 1 / (1 + distance) if distance >= 0 else 0.0
            
            details = {
                'verified': verified,
                'distance': distance,
                'threshold': model_threshold,
                'metric': metric,
                'model': self.model_name,
                'detector': self.detector_backend,
                'user_id': user_id
            }
            
            return verified, confidence, details
            
        except Exception as e:
            print(f"✗ Verification failed: {str(e)}")
            return False, 0.0, {'error': str(e)}
    
    def identify(self, face_image, threshold: Optional[float] = None,
                 top_n: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Identify user from face (1:N matching)
        
        Args:
            face_image: Face image to identify
            threshold: Minimum confidence threshold
            top_n: Return top N matches
            
        Returns:
            List of (user_id, confidence, details) tuples
        """
        if threshold is None:
            threshold = self.optimal_threshold
        
        try:
            results = []
            
            # Compare against all enrolled users
            for user_id in self.users.keys():
                verified, confidence, details = self.verify(user_id, face_image, threshold)
                
                if verified:
                    results.append((user_id, confidence, details))
            
            # Sort by confidence (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_n]
            
        except Exception as e:
            print(f"✗ Identification failed: {str(e)}")
            return []
    
    def capture_from_webcam(self, window_name: str = "Face Capture", low_res: bool = False) -> Optional[np.ndarray]:
        """
        Capture face image from webcam (optimized for real-time performance)
        
        Args:
            window_name: OpenCV window name
            low_res: Use lower resolution (320x240) for better performance
            
        Returns:
            Captured image or None
        """
        import threading
        import time
        
        # Prevent multiple simultaneous webcam access
        lock_file = self.database_path / ".camera_lock"
        if lock_file.exists():
            print("⚠️ Camera is already in use! Please wait...")
            return None
        
        # Create lock file
        lock_file.touch()
        
        try:
            return self._capture_webcam_internal(window_name, low_res)
        finally:
            # Always remove lock file
            if lock_file.exists():
                lock_file.unlink()
    
    def _capture_webcam_internal(self, window_name: str, low_res: bool) -> Optional[np.ndarray]:
        """Internal webcam capture implementation"""
        import uuid
        import time
        
        # EMERGENCY: Force release all cameras before starting
        print("🔧 Pre-cleanup: Releasing any stuck cameras...")
        for idx in range(3):
            try:
                temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if temp_cap.isOpened():
                    temp_cap.release()
            except:
                pass
        
        cv2.destroyAllWindows()
        time.sleep(0.5)
        print("   ✓ Pre-cleanup done\n")
        
        session_id = str(uuid.uuid4())[:8]
        print(f"{'='*60}")
        print(f"📸 WEBCAM SESSION START [{session_id}]")
        print(f"{'='*60}")
        print("   Press SPACE to capture")
        print("   Press ESC to cancel")
        print("")
        
        # Resolution based on performance mode
        width = 320 if low_res else 640
        height = 240 if low_res else 480
        
        print("🔍 Trying to open camera with DirectShow backend...")
        print(f"   Session: [{session_id}]")
        print("")
        
        # Force DirectShow backend (more stable on Windows than MSMF)
        cap = None
        camera_index = 0
        
        # Try camera indices with DirectShow
        for idx in [0, 1, 2]:
            print(f"   Testing camera {idx}...")
            try:
                # CORRECT syntax: cv2.VideoCapture(index, apiPreference)
                # NOT: cv2.VideoCapture(index + apiPreference)
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                
                # Configure camera BEFORE testing
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Now test if it works
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        camera_index = idx
                        print(f"✓ Camera {idx} opened successfully with DirectShow!")
                        break
                    else:
                        print(f"   Camera {idx} opened but can't read frames")
                        cap.release()
                        cap = None
                else:
                    print(f"   Camera {idx} failed to open")
                    if cap is not None:
                        cap.release()
                    cap = None
            except Exception as e:
                print(f"   Camera {idx} error: {e}")
                if cap is not None:
                    cap.release()
                cap = None
        
        if cap is None or not cap.isOpened():
            print("\n❌ Error: Could not open any webcam with DirectShow")
            print("\n💡 Troubleshooting:")
            print("   1. Open Windows Camera app to verify camera works")
            print("   2. Close ALL apps that might use camera:")
            print("      - Microsoft Teams, Zoom, Skype, Discord")
            print("      - Any browser tabs (Google Meet, etc.)")
            print("   3. Check Windows Privacy Settings:")
            print("      Settings > Privacy > Camera > Allow apps to access camera")
            print("   4. Restart your computer if camera is stuck")
            return None
        
        # Camera is already configured above, just add MJPEG codec
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG codec (faster)
        except:
            pass  # Ignore if codec setting fails
        
        # Warm up camera (skip first few frames which are often black/corrupted)
        print("🔥 Warming up camera...")
        for i in range(10):
            ret, _ = cap.read()
            if not ret:
                print(f"   Warm-up frame {i+1} failed, continuing...")
        
        # Verify camera is actually working
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("❌ Camera opened but cannot read frames reliably")
            cap.release()
            return None
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"✓ Camera ready: {actual_width}x{actual_height} @ {actual_fps} FPS")
        print(f"✓ Backend: DirectShow (no MSMF errors)")
        print("")
        
        captured_frame = None
        successful_reads = 0
        failed_reads = 0
        
        print("📹 Live preview started (face detection disabled for performance)")
        print("   Press SPACE to capture image")
        print("   Press ESC to cancel")
        
        try:
            while True:
                try:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        failed_reads += 1
                        if failed_reads > 10:
                            print(f"\n❌ Too many failed reads ({failed_reads}). Camera disconnected?")
                            break
                        continue
                    
                    successful_reads += 1
                    failed_reads = 0  # Reset counter on successful read
                    
                    # Simple display without face detection (for maximum performance)
                    display_frame = frame.copy()
                    
                    # Add instructions
                    cv2.putText(display_frame, "Press SPACE to capture, ESC to cancel", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show frame counter for feedback
                    cv2.putText(display_frame, f"Frames: {successful_reads}", 
                                (10, display_frame.shape[0] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow(window_name, display_frame)
                    
                    # Faster key check (waitKey(1) can cause lag if set too high)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 32:  # SPACE
                        # Convert BGR to RGB before returning (OpenCV uses BGR, but we need RGB)
                        # Ensure frame is uint8 before conversion
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        
                        captured_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Double check output is uint8
                        if captured_frame.dtype != np.uint8:
                            captured_frame = captured_frame.astype(np.uint8)
                        
                        print(f"\n✅ Image captured! [{session_id}] (Total frames: {successful_reads})")
                        print(f"✓ Converted BGR to RGB (dtype: {captured_frame.dtype})")
                        break
                    elif key == 27:  # ESC
                        print(f"\n❌ Capture cancelled [{session_id}]")
                        break
                
                except cv2.error as e:
                    # OpenCV specific errors (e.g., window closed manually)
                    print(f"\n⚠️ OpenCV error in loop: {e}")
                    break
                except Exception as e:
                    # Other errors - log but continue
                    print(f"\n⚠️ Error in frame loop: {e}")
                    failed_reads += 1
                    if failed_reads > 5:
                        break
        
        except Exception as e:
            print(f"❌ Webcam error: {e}")
        
        finally:
            # Always release camera and clean up
            print(f"\n🧹 Cleaning up camera resources... [{session_id}]")
            if cap is not None:
                cap.release()
                print(f"   ✓ Camera released [{session_id}]")
            
            # Destroy all windows multiple times to ensure cleanup
            for i in range(5):  # Increased from 3 to 5
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            print("   ✓ Windows destroyed")
            
            # Give OS more time to release camera (increased from 0.5 to 1.5 seconds)
            import time
            print("   ⏳ Waiting for OS to release camera...")
            time.sleep(1.5)
            
            # Final verification - try to open and close camera to confirm release
            try:
                test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if test_cap.isOpened():
                    test_cap.release()
                    print("   ✓ Camera release verified")
            except:
                pass
            
            print(f"✅ WEBCAM SESSION END [{session_id}]")
            print(f"{'='*60}\n")
        
        return captured_frame
    
    @staticmethod
    def release_all_cameras():
        """Force release all camera resources (utility function)"""
        print("🧹 Force releasing all cameras...")
        # Try to release cameras 0, 1, 2
        for idx in range(3):
            try:
                cap = cv2.VideoCapture(idx)
                cap.release()
            except:
                pass
        
        # Destroy all OpenCV windows
        for _ in range(5):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        
        import time
        time.sleep(1)
        print("✓ All cameras released")
    
    def list_enrolled_users(self) -> List[str]:
        """Get list of enrolled user IDs"""
        return list(self.users.keys())
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user from database
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if deleted successfully
        """
        if user_id in self.users:
            # Delete image file
            image_path = self.users[user_id]['image_path']
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Delete user folder if empty
            user_folder = self.database_path / user_id
            if user_folder.exists():
                try:
                    user_folder.rmdir()  # Only removes if empty
                    print(f"✓ Deleted folder: {user_folder}")
                except OSError:
                    # Folder not empty, that's okay
                    pass
            
            # Remove from database
            del self.users[user_id]
            self.save_database()
            
            print(f"✓ User '{user_id}' deleted")
            return True
        else:
            print(f"✗ User '{user_id}' not found")
            return False
    
    def clear_database(self) -> bool:
        """Clear all enrolled users"""
        import shutil
        
        # Delete all user folders
        for user_id in list(self.users.keys()):
            user_folder = self.database_path / user_id
            if user_folder.exists() and user_folder.is_dir():
                shutil.rmtree(user_folder)
                print(f"✓ Deleted folder: {user_folder}")
        
        self.users = {}
        self.save_database()
        print("✓ Face database cleared")
        return True
    
    def save_database(self):
        """Save user database to disk"""
        db_file = self.database_path / "database.pkl"
        with open(db_file, 'wb') as f:
            pickle.dump(self.users, f)
    
    def load_database(self):
        """Load user database from disk"""
        db_file = self.database_path / "database.pkl"
        if db_file.exists():
            try:
                with open(db_file, 'rb') as f:
                    self.users = pickle.load(f)
                print(f"✓ Loaded {len(self.users)} face users")
            except Exception as e:
                print(f"⚠️ Could not load database: {e}")
                self.users = {}
        else:
            self.users = {}
            print("✓ Initialized empty face database")
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            'total_users': len(self.users),
            'model_name': self.model_name,
            'detector_backend': self.detector_backend,
            'database_path': str(self.database_path)
        }
