"""
Voice Recognition Module using SpeechBrain
Uses pretrained ECAPA-TDNN model for speaker verification
"""

import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False


class VoiceRecognition:
    def __init__(self, database_path: str = "data/database/voices"):
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.templates = {}
        # Updated threshold based on performance analysis:
        # Genuine pairs: ~0.80, Impostor pairs: ~0.17
        # Optimal threshold: midpoint between clusters = 0.50
        self.optimal_threshold = 0.50
        
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError("Install: pip install librosa soundfile")
        
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError("Install: pip install speechbrain")
        
        # Initialize SpeechBrain ECAPA-TDNN model
        # Note: First run will download ~80MB pretrained model
        # Use COPY strategy to avoid symlink permission issues on Windows
        self.encoder = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join("models", "pretrained", "speechbrain"),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY  # Avoid symlink issues on Windows
        )
        self.embedding_size = 192
        self.load_all_templates()
    
    def extract_embedding(self, audio_path: str):
        """Extract voice embedding from audio file using SpeechBrain ECAPA-TDNN"""
        try:
            # Load audio using librosa
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz
            
            print(f"\n🎧 Audio Info:")
            print(f"   File: {audio_path}")
            print(f"   Duration: {len(audio)/sr:.2f}s")
            print(f"   Sample rate: {sr} Hz")
            print(f"   Audio shape: {audio.shape}")
            print(f"   Audio min/max BEFORE: {audio.min():.4f} / {audio.max():.4f}")
            print(f"   Audio energy BEFORE: {np.sum(audio**2):.4f}")
            print(f"   Audio RMS BEFORE: {np.sqrt(np.mean(audio**2)):.4f}")
            
            # CRITICAL CHECK: Validate audio has actual content
            audio_max_abs = np.abs(audio).max()
            audio_rms = np.sqrt(np.mean(audio**2))
            
            if audio_max_abs < 0.001:
                print(f"   ❌ CRITICAL: Audio is essentially silent! Max amplitude: {audio_max_abs:.6f}")
                print(f"   ⚠️  Microphone not working or permissions denied!")
                return None
            
            if audio_rms < 0.0001:
                print(f"   ❌ CRITICAL: Audio RMS too low: {audio_rms:.6f}")
                print(f"   ⚠️  No actual speech detected - mic issue!")
                return None
            
            # Apply Voice Activity Detection to remove silence/noise
            # This prevents model from learning room noise instead of voice
            intervals = librosa.effects.split(audio, top_db=30, frame_length=2048, hop_length=512)
            
            if len(intervals) == 0:
                print(f"   ❌ No speech detected in audio!")
                print(f"   ⚠️  Audio might be too quiet or only contains noise")
                return None
            
            # Concatenate speech segments only
            speech_segments = []
            for start, end in intervals:
                speech_segments.append(audio[start:end])
            audio = np.concatenate(speech_segments)
            
            print(f"   🎙️ VAD: Extracted {len(intervals)} speech segments")
            print(f"   Duration AFTER VAD: {len(audio)/sr:.2f}s")
            
            # Normalize audio to improve quality for quiet recordings
            # This helps with Streamlit's st.audio_input() which records at low volume
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()  # Normalize to [-1, 1]
                audio = audio * 0.95  # Scale to 95% to prevent clipping
                print(f"   Audio min/max AFTER: {audio.min():.4f} / {audio.max():.4f}")
                print(f"   Audio energy AFTER: {np.sum(audio**2):.4f}")
            
            # Validate audio
            if len(audio) < sr * 0.5:  # Less than 0.5 seconds
                print(f"   ❌ Audio too short! ({len(audio)/sr:.2f}s < 0.5s)")
                return None
            
            if np.sum(audio**2) < 0.01:  # After normalization, should have decent energy
                print(f"   ❌ Audio energy too low even after normalization!")
                return None
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            
            # Extract embedding using SpeechBrain
            with torch.no_grad():
                embedding = self.encoder.encode_batch(audio_tensor)
                embedding_np = embedding.squeeze().cpu().numpy()
                
                print(f"   Embedding shape: {embedding_np.shape}")
                print(f"   Embedding norm: {np.linalg.norm(embedding_np):.4f}")
                print(f"   Embedding mean: {np.mean(embedding_np):.4f}")
                print(f"   Embedding std: {np.std(embedding_np):.4f}")
                
                # Check if embedding is valid
                if np.all(embedding_np == 0) or np.isnan(embedding_np).any():
                    print(f"   ❌ Invalid embedding (all zeros or NaN)!")
                    return None
                
                print(f"   ✅ Valid embedding extracted\n")
                return embedding_np
                
        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def enroll(self, user_id: str, audio_path: str):
        embedding = self.extract_embedding(audio_path)
        if embedding is None:
            return False
        
        self.templates[user_id] = {
            "user_id": user_id,
            "embedding": embedding,
            "model": "SpeechBrain"
        }
        self.save_template(user_id, self.templates[user_id])
        return True
    
    def verify(self, user_id: str, audio_path: str):
        if user_id not in self.templates:
            print(f"⚠️ User {user_id} not found in voice database")
            return False, 0.0
        
        test_emb = self.extract_embedding(audio_path)
        if test_emb is None:
            print(f"❌ Failed to extract embedding from {audio_path}")
            return False, 0.0
        
        similarity = self.cosine_similarity(test_emb, self.templates[user_id]["embedding"])
        
        # Debug logging
        print(f"\n🎤 Voice Verification Debug:")
        print(f"   User ID: {user_id}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Threshold: {self.optimal_threshold}")
        print(f"   Result: {'✅ VERIFIED' if similarity >= self.optimal_threshold else '❌ REJECTED'}\n")
        
        return similarity >= self.optimal_threshold, float(similarity)
    
    def identify(self, audio_path: str, top_n: int = 5):
        """
        Identify user from voice (1:N matching)
        
        Args:
            audio_path: Path to audio file
            top_n: Number of top matches to return
            
        Returns:
            List of (user_id, similarity) tuples sorted by similarity
        """
        test_emb = self.extract_embedding(audio_path)
        if test_emb is None:
            return []
        
        matches = []
        for uid, tmpl in self.templates.items():
            sim = self.cosine_similarity(test_emb, tmpl["embedding"])
            matches.append((uid, float(sim)))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_n]
    
    def save_template(self, user_id: str, template):
        path = self.database_path / f"{user_id}_voice.pkl"
        with open(path, "wb") as f:
            pickle.dump(template, f)
    
    def load_template(self, user_id: str):
        path = self.database_path / f"{user_id}_voice.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    
    def load_all_templates(self):
        for file in self.database_path.glob("*_voice.pkl"):
            uid = file.stem.replace("_voice", "")
            tmpl = self.load_template(uid)
            if tmpl:
                self.templates[uid] = tmpl
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user from the voice database
        
        Args:
            user_id: User identifier to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if user_id not in self.templates:
            print(f"⚠️ User {user_id} not found in voice database")
            return False
        
        # Remove from memory
        del self.templates[user_id]
        
        # Delete pickle file
        pkl_path = self.database_path / f"{user_id}_voice.pkl"
        if pkl_path.exists():
            pkl_path.unlink()
            print(f"✅ Deleted file: {pkl_path}")
        
        print(f"✅ User {user_id} deleted from voice database")
        return True
    
    def clear_database(self) -> bool:
        """
        Clear entire voice database (remove all users)
        
        Returns:
            True if cleared successfully
        """
        user_count = len(self.templates)
        
        # Delete all pickle files
        for file in self.database_path.glob("*_voice.pkl"):
            file.unlink()
        
        # Clear memory
        self.templates.clear()
        
        print(f"✅ Voice database cleared ({user_count} users deleted)")
        return True
    
    def list_enrolled_users(self):
        """Return list of enrolled user IDs"""
        return list(self.templates.keys())
    
    def get_statistics(self):
        return {
            "total_users": len(self.templates),
            "model": "SpeechBrain-ECAPA-TDNN",
            "threshold": self.optimal_threshold
        }
