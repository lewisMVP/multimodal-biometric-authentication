"""
Configuration settings for Multimodal Biometric Authentication System
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Biometric data subdirectories
FINGERPRINT_DIR = RAW_DATA_DIR / "fingerprints"
FACE_DIR = RAW_DATA_DIR / "faces"
VOICE_DIR = RAW_DATA_DIR / "voices"
IRIS_DIR = RAW_DATA_DIR / "iris"

# Database directories (for storing templates)
DB_DIR = DATA_DIR / "database"
FINGERPRINT_DB = DB_DIR / "fingerprints"
FACE_DB = DB_DIR / "faces"
VOICE_DB = DB_DIR / "voices"
IRIS_DB = DB_DIR / "iris"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# Fingerprint Recognition Settings
FINGERPRINT_CONFIG = {
    'feature_extractor': 'ORB',  # Options: ORB, SIFT, SURF
    'num_features': 500,
    'matching_threshold': 0.3,
    'verification_threshold': 0.3,
    'identification_threshold': 0.25,
}

# Face Recognition Settings
FACE_CONFIG = {
    'model': 'dlib',  # Options: dlib, facenet, arcface
    'verification_threshold': 0.6,
    'identification_threshold': 0.5,
}

# Voice Recognition Settings
VOICE_CONFIG = {
    'sample_rate': 16000,
    'feature_type': 'mfcc',  # Options: mfcc, mel-spectrogram
    'verification_threshold': 0.5,
    'identification_threshold': 0.4,
}

# Iris Recognition Settings
IRIS_CONFIG = {
    'segmentation_method': 'circular_hough',
    'verification_threshold': 0.7,
    'identification_threshold': 0.6,
}

# Fusion System Settings
FUSION_CONFIG = {
    'method': 'weighted_sum',  # Options: weighted_sum, svm, neural_network
    'weights': {
        'fingerprint': 0.25,
        'face': 0.25,
        'voice': 0.25,
        'iris': 0.25
    },
    'decision_threshold': 0.5,
    'require_min_modalities': 2,  # Minimum modalities required for authentication
}

# System Settings
SYSTEM_CONFIG = {
    'log_level': 'INFO',
    'save_raw_images': True,
    'save_processed_images': True,
    'save_logs': True,
}

# Create directories if they don't exist
def create_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        FINGERPRINT_DIR, FACE_DIR, VOICE_DIR, IRIS_DIR,
        DB_DIR, FINGERPRINT_DB, FACE_DB, VOICE_DB, IRIS_DB,
        MODELS_DIR, PRETRAINED_DIR,
        RESULTS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✓ All directories created successfully")

if __name__ == "__main__":
    create_directories()

