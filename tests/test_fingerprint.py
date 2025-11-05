"""
Unit tests for Fingerprint Recognition Module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append('..')
from modules.fingerprint_recognition import FingerprintRecognition


class TestFingerprintRecognition:
    """Test cases for FingerprintRecognition class"""
    
    @pytest.fixture
    def fp_system(self, tmp_path):
        """Fixture to create a FingerprintRecognition instance"""
        return FingerprintRecognition(database_path=str(tmp_path / "test_db"))
    
    @pytest.fixture
    def sample_fingerprint(self):
        """Fixture to create a sample fingerprint image"""
        # Create a synthetic fingerprint (random grayscale image for testing)
        return np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    
    def test_initialization(self, tmp_path):
        """Test system initialization"""
        fp_system = FingerprintRecognition(database_path=str(tmp_path / "test_db"))
        assert fp_system is not None
        assert (tmp_path / "test_db").exists()
    
    def test_preprocess(self, fp_system, sample_fingerprint):
        """Test fingerprint preprocessing"""
        preprocessed = fp_system.preprocess(sample_fingerprint)
        
        assert preprocessed is not None
        assert preprocessed.shape == sample_fingerprint.shape
        assert preprocessed.dtype == np.uint8
    
    def test_extract_features(self, fp_system, sample_fingerprint):
        """Test feature extraction"""
        preprocessed = fp_system.preprocess(sample_fingerprint)
        features = fp_system.extract_features(preprocessed)
        
        assert features is not None
        assert 'keypoints' in features
        assert 'descriptors' in features
        assert 'method' in features
        assert features['method'] == 'ORB'
    
    def test_enrollment(self, fp_system, sample_fingerprint):
        """Test user enrollment"""
        success = fp_system.enroll('test_user_001', sample_fingerprint)
        
        assert success is True
        assert 'test_user_001' in fp_system.templates
    
    def test_enrollment_invalid_image(self, fp_system):
        """Test enrollment with invalid image"""
        success = fp_system.enroll('test_user_002', None)
        assert success is False
    
    def test_verification_user_not_found(self, fp_system, sample_fingerprint):
        """Test verification with non-existent user"""
        is_verified, confidence = fp_system.verify('nonexistent_user', sample_fingerprint)
        
        assert is_verified is False
        assert confidence == 0.0
    
    def test_verification_same_fingerprint(self, fp_system, sample_fingerprint):
        """Test verification with same fingerprint"""
        # Enroll user
        fp_system.enroll('test_user_003', sample_fingerprint)
        
        # Verify with same fingerprint
        is_verified, confidence = fp_system.verify('test_user_003', sample_fingerprint)
        
        # Should have high confidence (though might not be perfect due to randomness)
        assert isinstance(is_verified, bool)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_identification_empty_database(self, fp_system, sample_fingerprint):
        """Test identification with empty database"""
        results = fp_system.identify(sample_fingerprint)
        assert results == []
    
    def test_identification_with_enrolled_users(self, fp_system, sample_fingerprint):
        """Test identification with enrolled users"""
        # Enroll multiple users
        fp_system.enroll('test_user_004', sample_fingerprint)
        fp_system.enroll('test_user_005', np.random.randint(0, 255, (300, 300), dtype=np.uint8))
        
        # Try to identify
        results = fp_system.identify(sample_fingerprint)
        
        assert isinstance(results, list)
        # Check result format
        for user_id, score in results:
            assert isinstance(user_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_match_features(self, fp_system, sample_fingerprint):
        """Test feature matching"""
        # Extract features from two fingerprints
        preprocessed1 = fp_system.preprocess(sample_fingerprint)
        features1 = fp_system.extract_features(preprocessed1)
        
        fingerprint2 = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        preprocessed2 = fp_system.preprocess(fingerprint2)
        features2 = fp_system.extract_features(preprocessed2)
        
        # Match features
        match_score, num_matches = fp_system.match_features(features1, features2)
        
        assert isinstance(match_score, float)
        assert isinstance(num_matches, int)
        assert 0.0 <= match_score <= 1.0
        assert num_matches >= 0
    
    def test_get_statistics(self, fp_system, sample_fingerprint):
        """Test statistics retrieval"""
        # Enroll some users
        fp_system.enroll('test_user_006', sample_fingerprint)
        fp_system.enroll('test_user_007', sample_fingerprint)
        
        stats = fp_system.get_statistics()
        
        assert 'total_users' in stats
        assert 'database_path' in stats
        assert stats['total_users'] == 2
    
    def test_save_and_load_templates(self, fp_system, sample_fingerprint, tmp_path):
        """Test template persistence"""
        # Enroll user
        fp_system.enroll('test_user_008', sample_fingerprint)
        
        # Save templates
        fp_system.save_templates()
        
        # Create new instance and load templates
        new_fp_system = FingerprintRecognition(database_path=str(tmp_path / "test_db"))
        
        assert 'test_user_008' in new_fp_system.templates


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

