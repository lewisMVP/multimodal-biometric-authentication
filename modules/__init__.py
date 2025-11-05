"""
Multimodal Biometric Authentication System
Modules Package
"""

from .fingerprint_recognition import FingerprintRecognition
from .face_recognition import FaceRecognition
from .voice_recognition import VoiceRecognition
from .iris_recognition import IrisRecognition

__all__ = [
    'FingerprintRecognition',
    'FaceRecognition',
    'VoiceRecognition',
    'IrisRecognition'
]

