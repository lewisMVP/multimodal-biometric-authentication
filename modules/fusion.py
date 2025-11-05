"""
Multimodal Fusion Module using Machine Learning
Combines scores from multiple biometric modalities using trained ML models
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SK_LEARN_AVAILABLE = True
except ImportError:
    SK_LEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")


class MultimodalFusion:
    """
    Fusion system for combining multiple biometric modalities
    Supports multiple fusion strategies:
    1. Score-level fusion (weighted sum)
    2. ML-based fusion (Random Forest, SVM)
    3. Decision-level fusion (majority voting, 3/4 rule)
    """
    
    def __init__(self, model_path: str = "models/fusion_model.pkl",
                 fusion_method: str = "random_forest"):
        """
        Initialize fusion system
        
        Args:
            model_path: Path to save/load trained fusion model
            fusion_method: 'random_forest', 'svm', 'weighted_sum', or 'voting'
        """
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.fusion_method = fusion_method
        
        # Fusion model and scaler
        self.model = None
        self.scaler = None
        
        # Default weights for weighted sum fusion
        self.weights = {
            'fingerprint': 0.3,
            'face': 0.3,
            'iris': 0.2,
            'voice': 0.2
        }
        
        # Decision thresholds for each modality
        self.thresholds = {
            'fingerprint': 0.3,
            'face': 0.4,
            'iris': 0.65,
            'voice': 0.6
        }
        
        if not SK_LEARN_AVAILABLE and fusion_method in ['random_forest', 'svm']:
            raise ImportError("scikit-learn required for ML fusion. Install with: pip install scikit-learn")
        
        # Try to load existing model
        self.load_model()
        
        print(f"✓ Fusion system initialized ({fusion_method})")
    
    def weighted_sum_fusion(self, scores: Dict[str, float]) -> float:
        """
        Simple weighted sum fusion
        
        Args:
            scores: Dictionary of {modality: score}
            
        Returns:
            Fused score (0-1)
        """
        fused_score = 0.0
        total_weight = 0.0
        
        for modality, score in scores.items():
            if score is not None and modality in self.weights:
                fused_score += self.weights[modality] * score
                total_weight += self.weights[modality]
        
        if total_weight > 0:
            fused_score /= total_weight
        
        return fused_score
    
    def voting_fusion(self, scores: Dict[str, float], 
                     min_required: int = 3) -> Tuple[bool, float]:
        """
        Decision-level fusion using voting
        Requires at least min_required modalities to accept
        
        Args:
            scores: Dictionary of {modality: score}
            min_required: Minimum number of modalities that must pass (default: 3/4)
            
        Returns:
            Tuple of (is_accepted, confidence)
        """
        accepted_count = 0
        total_score = 0.0
        valid_count = 0
        
        for modality, score in scores.items():
            if score is not None:
                valid_count += 1
                total_score += score
                
                # Check if this modality accepts
                if modality in self.thresholds and score >= self.thresholds[modality]:
                    accepted_count += 1
        
        # Calculate average confidence
        confidence = total_score / valid_count if valid_count > 0 else 0.0
        
        # Decision: at least min_required modalities must accept
        is_accepted = accepted_count >= min_required
        
        return is_accepted, confidence
    
    def ml_fusion_predict(self, scores: Dict[str, float]) -> Tuple[bool, float]:
        """
        ML-based fusion using trained Random Forest or SVM
        
        Args:
            scores: Dictionary of {modality: score}
            
        Returns:
            Tuple of (is_verified, confidence)
        """
        if self.model is None:
            print("⚠️ ML model not trained. Using weighted sum fusion.")
            fused_score = self.weighted_sum_fusion(scores)
            return fused_score >= 0.5, fused_score
        
        # Convert scores to feature vector
        feature_vector = self._scores_to_features(scores)
        
        # Normalize
        if self.scaler is not None:
            feature_vector = self.scaler.transform([feature_vector])
        else:
            feature_vector = [feature_vector]
        
        # Predict
        try:
            prediction = self.model.predict(feature_vector)[0]
            
            # Get confidence (probability)
            if hasattr(self.model, 'predict_proba'):
                confidence = self.model.predict_proba(feature_vector)[0][1]
            else:
                # For models without predict_proba, use decision function
                decision = self.model.decision_function(feature_vector)[0]
                confidence = 1 / (1 + np.exp(-decision))  # Sigmoid
            
            is_verified = bool(prediction == 1)
            
            return is_verified, float(confidence)
            
        except Exception as e:
            print(f"⚠️ ML prediction failed: {e}. Using weighted sum.")
            fused_score = self.weighted_sum_fusion(scores)
            return fused_score >= 0.5, fused_score
    
    def fuse(self, scores: Dict[str, float]) -> Tuple[bool, float]:
        """
        Main fusion function - routes to appropriate fusion method
        
        Args:
            scores: Dictionary of {modality: score}
                Example: {'fingerprint': 0.85, 'face': 0.72, 'iris': 0.90, 'voice': 0.65}
            
        Returns:
            Tuple of (is_verified, confidence_score)
        """
        if self.fusion_method == 'weighted_sum':
            fused_score = self.weighted_sum_fusion(scores)
            return fused_score >= 0.5, fused_score
        
        elif self.fusion_method == 'voting':
            return self.voting_fusion(scores, min_required=3)
        
        elif self.fusion_method in ['random_forest', 'svm']:
            return self.ml_fusion_predict(scores)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def train_fusion_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: Optional[np.ndarray] = None,
                           y_test: Optional[np.ndarray] = None):
        """
        Train ML fusion model
        
        Args:
            X_train: Training features (N x 4) - [fp_score, face_score, iris_score, voice_score]
            y_train: Training labels (N,) - [1: genuine, 0: impostor]
            X_test: Test features (optional)
            y_test: Test labels (optional)
        """
        if not SK_LEARN_AVAILABLE:
            raise ImportError("scikit-learn required for training")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize model based on fusion method
        if self.fusion_method == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.fusion_method == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Cannot train with fusion_method: {self.fusion_method}")
        
        # Train
        print(f"Training {self.fusion_method} fusion model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"  Training accuracy: {train_acc:.2%}")
        
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            test_pred = self.model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)
            print(f"  Test accuracy: {test_acc:.2%}")
            print("\nClassification Report:")
            print(classification_report(y_test, test_pred, 
                                       target_names=['Impostor', 'Genuine']))
        
        # Save model
        self.save_model()
        print(f"✓ Fusion model trained and saved")
    
    def _scores_to_features(self, scores: Dict[str, float]) -> List[float]:
        """
        Convert score dictionary to feature vector
        
        Args:
            scores: Dictionary of {modality: score}
            
        Returns:
            Feature vector [fp, face, iris, voice]
        """
        modalities = ['fingerprint', 'face', 'iris', 'voice']
        features = []
        
        for modality in modalities:
            if modality in scores and scores[modality] is not None:
                features.append(scores[modality])
            else:
                features.append(0.0)  # Missing modality
        
        return features
    
    def save_model(self):
        """Save fusion model and scaler to disk"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'weights': self.weights,
                'thresholds': self.thresholds,
                'fusion_method': self.fusion_method
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✓ Fusion model saved to {self.model_path}")
    
    def load_model(self):
        """Load fusion model from disk"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.weights = model_data.get('weights', self.weights)
                self.thresholds = model_data.get('thresholds', self.thresholds)
                
                print(f"✓ Loaded fusion model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Could not load fusion model: {e}")
                self.model = None
        else:
            print("ℹ️ No pre-trained fusion model found")
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set fusion weights for weighted sum method
        
        Args:
            weights: Dictionary of {modality: weight}
        """
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        print(f"✓ Fusion weights updated: {self.weights}")
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """
        Set decision thresholds for voting method
        
        Args:
            thresholds: Dictionary of {modality: threshold}
        """
        self.thresholds.update(thresholds)
        print(f"✓ Decision thresholds updated: {self.thresholds}")
    
    def get_statistics(self) -> Dict:
        """Get fusion system statistics"""
        return {
            'fusion_method': self.fusion_method,
            'model_trained': self.model is not None,
            'weights': self.weights,
            'thresholds': self.thresholds,
            'model_path': str(self.model_path)
        }


# Helper function to generate training data for fusion
def generate_fusion_training_data(genuine_scores_list: List[Dict[str, float]],
                                  impostor_scores_list: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for fusion model
    
    Args:
        genuine_scores_list: List of score dictionaries for genuine attempts
        impostor_scores_list: List of score dictionaries for impostor attempts
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    X = []
    y = []
    
    # Process genuine scores (label = 1)
    for scores in genuine_scores_list:
        features = [
            scores.get('fingerprint', 0.0),
            scores.get('face', 0.0),
            scores.get('iris', 0.0),
            scores.get('voice', 0.0)
        ]
        X.append(features)
        y.append(1)
    
    # Process impostor scores (label = 0)
    for scores in impostor_scores_list:
        features = [
            scores.get('fingerprint', 0.0),
            scores.get('face', 0.0),
            scores.get('iris', 0.0),
            scores.get('voice', 0.0)
        ]
        X.append(features)
        y.append(0)
    
    return np.array(X), np.array(y)
