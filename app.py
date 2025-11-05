"""
Multimodal Biometric Authentication System - Web Interface
Using Streamlit for interactive demo
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
from PIL import Image
import time

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.fingerprint_recognition import FingerprintRecognition
from modules.iris_recognition import IrisRecognition
from modules.face_recognition import FaceRecognition
from modules.voice_recognition import VoiceRecognition

# Page configuration
st.set_page_config(
    page_title="Multimodal Biometric Auth",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        background-color: #f44336;
        color: white;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #2196F3;
        color: white;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'fingerprint_system' not in st.session_state:
    st.session_state.fingerprint_system = FingerprintRecognition()
if 'iris_system' not in st.session_state:
    st.session_state.iris_system = IrisRecognition()
if 'face_system' not in st.session_state:
    st.session_state.face_system = FaceRecognition()
if 'voice_system' not in st.session_state:
    with st.spinner("🎤 Initializing Voice Recognition (first run downloads ~80MB model)..."):
        st.session_state.voice_system = VoiceRecognition()
if 'enrolled_user' not in st.session_state:
    st.session_state.enrolled_user = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "📊 Dashboard"

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/fingerprint.png", width=80)
        st.title("Navigation")
        
        mode = st.radio(
            "Select Mode:",
            ["📊 Dashboard", "✍️ Enrollment", "🔍 Verification", "🔎 Identification", "⚙️ Settings"],
            index=["📊 Dashboard", "✍️ Enrollment", "🔍 Verification", "🔎 Identification", "⚙️ Settings"].index(st.session_state.current_mode),
            key="mode_radio"
        )
        
        # Update current mode
        st.session_state.current_mode = mode
        
        st.markdown("---")
        
        # System stats
        fp_stats = st.session_state.fingerprint_system.get_statistics()
        iris_stats = st.session_state.iris_system.get_statistics()
        face_stats = st.session_state.face_system.get_statistics()
        voice_stats = st.session_state.voice_system.get_statistics()
        st.metric("👆 Fingerprint Users", fp_stats['total_users'])
        st.metric("👁️ Iris Users", iris_stats['total_users'])
        st.metric("😊 Face Users", face_stats['total_users'])
        st.metric("🎤 Voice Users", voice_stats['total_users'])
        
        st.markdown("---")
        st.caption("Multimodal Biometric Auth v1.0")
        st.caption("Developed by: Lewis Chu")
    
    # Main content based on mode
    if mode == "📊 Dashboard":
        show_dashboard()
    elif mode == "✍️ Enrollment":
        show_enrollment()
    elif mode == "🔍 Verification":
        show_verification()
    elif mode == "🔎 Identification":
        show_identification()
    elif mode == "⚙️ Settings":
        show_settings()

def show_dashboard():
    # Welcome banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
        <h2 style='margin: 0; color: white;'>🔐 Multimodal Biometric Authentication System</h2>
        <p style='margin-top: 0.5rem; opacity: 0.9;'>Advanced authentication with Fingerprint, Iris, Face, and Voice recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview metrics
    st.subheader("📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    fp_stats = st.session_state.fingerprint_system.get_statistics()
    iris_stats = st.session_state.iris_system.get_statistics()
    face_stats = st.session_state.face_system.get_statistics()
    voice_stats = st.session_state.voice_system.get_statistics()
    
    total_users = max(fp_stats['total_users'], iris_stats['total_users'], 
                     face_stats['total_users'], voice_stats['total_users'])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👆 Fingerprint Users", fp_stats['total_users'])
        st.caption("✓ SIFT Feature Extraction")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👁️ Iris Users", iris_stats['total_users'])
        if 'total_eyes' in iris_stats:
            st.caption(f"✓ {iris_stats['total_eyes']} eyes enrolled")
        else:
            st.caption("✓ Daugman's Algorithm")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("😊 Face Users", face_stats['total_users'])
        st.caption("✓ DeepFace (VGG-Face)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎤 Voice Users", voice_stats['total_users'])
        st.caption("✓ ECAPA-TDNN Speaker")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Active Biometric Modalities")
        st.success("""
        **✅ Fingerprint Recognition**
        - Algorithm: SIFT (Scale-Invariant Feature Transform)
        - Preprocessing: CLAHE + Gaussian Blur
        - Matching: FLANN-based KNN with Lowe's ratio test
        
        **✅ Iris Recognition**
        - Algorithm: Daugman's Rubber Sheet Model
        - Feature Extraction: Gabor Wavelets (4 orientations)
        - Matching: Hamming Distance with rotation handling
        
        **✅ Face Recognition**
        - Model: VGG-Face (DeepFace framework)
        - Detector: RetinaFace
        - Distance Metric: Cosine Similarity
        
        **✅ Voice Recognition**
        - Model: ECAPA-TDNN (SpeechBrain)
        - Embeddings: 192 dimensions
        - Matching: Cosine Similarity (threshold: 0.25)
        """)
    
    with col2:
        st.subheader("📈 System Capabilities")
        st.info("""
        **🔹 Enrollment**
        - Support for 4 biometric modalities
        - Image/audio upload or live capture
        - Quality assessment and validation
        
        **🔹 Verification (1:1 Matching)**
        - Verify claimed identity
        - Fast matching (< 500ms per modality)
        - Configurable thresholds
        
        **🔹 Identification (1:N Matching)**
        - Find user from database
        - Multi-modal fusion support
        - Top-N results ranking
        
        **🔹 Database Management**
        - User enrollment/deletion
        - Database statistics
        - Backup and restore
        """)
    
    # Quick actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("➕ Enroll New User", use_container_width=True, type="primary"):
            st.session_state.current_mode = "✍️ Enrollment"
            st.rerun()
    with col2:
        if st.button("🔍 Verify User", use_container_width=True):
            st.session_state.current_mode = "🔍 Verification"
            st.rerun()
    with col3:
        if st.button("🔎 Identify User", use_container_width=True):
            st.session_state.current_mode = "🔎 Identification"
            st.rerun()
    with col4:
        if st.button("⚙️ Manage Database", use_container_width=True):
            st.session_state.current_mode = "⚙️ Settings"
            st.rerun()

def show_enrollment():
    st.header("✍️ User Enrollment")
    
    st.info("📌 Enroll a new user by providing their biometric data")
    
    # User ID input
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.text_input("👤 User ID", placeholder="Enter unique user ID (e.g., USER001)")
    
    # Select modality
    st.subheader("🔐 Select Biometric Modality")
    modality = st.radio(
        "Choose modality to enroll:",
        ["👆 Fingerprint", "👁️ Iris", "😊 Face", "🎤 Voice"],
        horizontal=True
    )
    
    if modality == "👆 Fingerprint":
        enroll_fingerprint(user_id)
    elif modality == "👁️ Iris":
        enroll_iris(user_id)
    elif modality == "😊 Face":
        enroll_face(user_id)
    elif modality == "🎤 Voice":
        enroll_voice(user_id)

def enroll_fingerprint(user_id):
    """Fingerprint enrollment section"""
    st.subheader("👆 Fingerprint Capture")
    
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Capture"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload fingerprint image", 
                                        type=['png', 'jpg', 'jpeg', 'bmp'],
                                        help="Upload a clear fingerprint image")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                # Preprocess and show
                preprocessed = st.session_state.fingerprint_system.preprocess(img_gray)
                st.image(preprocessed, caption="Preprocessed (CLAHE)", 
                        use_container_width=True, clamp=True)
            
            # Enrollment button
            if st.button("✅ Enroll User", type="primary", use_container_width=True):
                if not user_id:
                    st.error("❌ Please enter a User ID")
                else:
                    with st.spinner("Enrolling user..."):
                        success = st.session_state.fingerprint_system.enroll(user_id, img_gray)
                        
                        if success:
                            st.markdown(f'<div class="success-box">✅ User {user_id} enrolled successfully!</div>', 
                                      unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown('<div class="error-box">❌ Enrollment failed. Please try again.</div>', 
                                      unsafe_allow_html=True)
    
    with tab2:
        st.info("📷 Live capture feature coming soon! Please use Upload Image for now.")

def enroll_iris(user_id):
    """Iris enrollment section"""
    st.subheader("👁️ Iris Capture")
    
    # Eye side selector
    eye_side = st.radio(
        "Select Eye to Enroll",
        options=["Left Eye", "Right Eye"],
        horizontal=True,
        help="Choose which eye you're enrolling. The system will validate that you uploaded the correct eye.",
        key="iris_eye_selector"
    )
    
    # Map display text to internal value
    eye_map = {"Left Eye": "left", "Right Eye": "right"}
    eye_value = eye_map[eye_side]
    
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Capture"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload iris image", 
                                        type=['png', 'jpg', 'jpeg', 'bmp'],
                                        help="Upload a clear iris image (close-up of the eye)",
                                        key="iris_enroll_upload")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                # Segment iris and show
                with st.spinner("Segmenting iris..."):
                    seg_result, params = st.session_state.iris_system.segment_iris(img_gray)
                    
                    if seg_result is not None and params is not None:
                        # Draw circles on original image for visualization
                        vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                        cv2.circle(vis_img, params['iris_center'], params['iris_radius'], (0, 255, 0), 2)
                        cv2.circle(vis_img, params['pupil_center'], params['pupil_radius'], (255, 0, 0), 2)
                        st.image(vis_img, caption="Segmented Iris", use_container_width=True)
                        
                        # Auto-detect eye side and show warning if mismatch
                        detected_eye = st.session_state.iris_system.detect_eye_side(img_gray, params)
                        
                        if detected_eye in ["left", "right"]:
                            if detected_eye != eye_value:
                                st.warning(f"⚠️ **Eye Side Warning**\n\nYou selected: **{eye_value.upper()}** eye\n\nAuto-detected: **{detected_eye.upper()}** eye\n\nNote: Auto-detection may be inaccurate. Please verify your selection.")
                            else:
                                st.success(f"✓ Confirmed: **{eye_value.upper()}** eye")
                        else:
                            st.info(f"ℹ️ Cannot auto-detect eye side. Using your selection: **{eye_value.upper()}** eye")
                        
                        # Show quality assessment
                        quality = st.session_state.iris_system.assess_iris_quality(img_gray, params)
                        st.metric("Quality Score", f"{quality['overall']:.0f}/100")
                        
                        if quality['overall'] < 50:
                            st.warning(f"⚠️ Quality: {quality['recommendation']}")
                    else:
                        st.error("❌ Iris segmentation failed. Please use a clearer image.")
            
            # Enrollment button
            if st.button("✅ Enroll Iris", type="primary", use_container_width=True, key="enroll_iris_btn"):
                if not user_id:
                    st.error("❌ Please enter a User ID")
                else:
                    with st.spinner(f"Enrolling {eye_side.lower()} template..."):
                        success = st.session_state.iris_system.enroll(user_id, img_gray, 
                                                                      eye_side=eye_value)
                        
                        if success:
                            st.markdown(f'<div class="success-box">✅ User {user_id} {eye_side.lower()} enrolled successfully!</div>', 
                                      unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown('<div class="error-box">❌ Enrollment failed. Iris segmentation or feature extraction failed.</div>', 
                                      unsafe_allow_html=True)
    
    with tab2:
        st.info("📷 Live capture feature coming soon! Please use Upload Image for now.")

def enroll_face(user_id):
    """Face enrollment section"""
    st.subheader("😊 Face Capture")
    
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Capture"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload face image", 
                                        type=['png', 'jpg', 'jpeg'],
                                        help="Upload a clear face image",
                                        key="face_enroll_upload")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert RGB if needed (DeepFace expects RGB)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                # Extract and show detected faces
                with st.spinner("Detecting faces..."):
                    try:
                        faces = st.session_state.face_system.extract_faces(img_array)
                        
                        if faces and len(faces) > 0:
                            # Show first detected face
                            face_img = faces[0]['face']
                            confidence = faces[0]['confidence']
                            
                            st.image(face_img, caption="Detected Face", use_container_width=True)
                            st.metric("Detection Confidence", f"{confidence:.2%}")
                            
                            if confidence < 0.9:
                                st.warning("⚠️ Low confidence. Consider using a clearer image.")
                            
                            if len(faces) > 1:
                                st.info(f"ℹ️ Detected {len(faces)} faces. Using the first one.")
                        else:
                            st.error("❌ No face detected. Please use a clearer image.")
                    except Exception as e:
                        st.error(f"❌ Face detection failed: {str(e)}")
            
            # Enrollment button
            if st.button("✅ Enroll Face", type="primary", use_container_width=True, key="enroll_face_btn"):
                if not user_id:
                    st.error("❌ Please enter a User ID")
                else:
                    with st.spinner("Enrolling face..."):
                        try:
                            success = st.session_state.face_system.enroll(user_id, img_array)
                            
                            if success:
                                st.markdown(f'<div class="success-box">✅ User {user_id} face enrolled successfully!</div>', 
                                          unsafe_allow_html=True)
                                st.balloons()
                                st.success("🎉 Enrollment complete! You can now verify or identify this user.")
                            else:
                                st.markdown('<div class="error-box">❌ Enrollment failed. Face detection failed or user already exists.</div>', 
                                          unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', 
                                      unsafe_allow_html=True)
    
    with tab2:
        st.info("📷 Live webcam capture")
        
        # Warning and troubleshooting
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning("⚠️ **Tip:** Close webcam window with ESC, then wait 2-3 seconds before reopening.")
        with col2:
            if st.button("🔧 Fix Camera", help="Force release camera if stuck", key="fix_camera_btn"):
                with st.spinner("Releasing camera..."):
                    from modules.face_recognition import FaceRecognition
                    FaceRecognition.release_all_cameras()
                    st.success("✅ Camera released! Try again.")
                    st.session_state.camera_locked = False
        
        # Initialize camera lock state
        if 'camera_locked' not in st.session_state:
            st.session_state.camera_locked = False
        
        # Create button to start webcam
        button_disabled = st.session_state.camera_locked
        if st.button("📸 Start Webcam", key="face_webcam_btn", disabled=button_disabled):
            st.session_state.camera_locked = True  # Lock camera
            
            with st.spinner("Opening webcam... (Press ESC in webcam window to cancel)"):
                try:
                    captured_image = st.session_state.face_system.capture_from_webcam()
                    
                    if captured_image is not None:
                        # Store captured image in session state
                        st.session_state.face_captured_image = captured_image
                        st.session_state.camera_locked = False  # Unlock camera
                        st.success("✅ Image captured! Scroll down to enroll.")
                        # Don't rerun immediately - let camera cleanup finish
                    else:
                        st.warning("⚠️ No image captured. Webcam was cancelled.")
                        st.session_state.camera_locked = False  # Unlock camera
                        
                except Exception as e:
                    st.error(f"❌ Webcam error: {str(e)}")
                    st.session_state.camera_locked = False  # Unlock camera
        
        # Show captured image and enrollment button
        if 'face_captured_image' in st.session_state and st.session_state.face_captured_image is not None:
            st.success("✅ Image ready for enrollment!")
            st.image(st.session_state.face_captured_image, caption="Captured Face", use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            # Check if already enrolled
            is_enrolled = st.session_state.get('face_enrolled', False)
            
            with col1:
                if st.button("✅ Enroll This Face", type="primary", key="enroll_captured_face", 
                           use_container_width=True, disabled=is_enrolled):
                    if not user_id:
                        st.error("❌ Please enter a User ID")
                    else:
                        with st.spinner("Enrolling captured face..."):
                            try:
                                success = st.session_state.face_system.enroll(user_id, st.session_state.face_captured_image)
                                
                                if success:
                                    st.markdown(f'<div class="success-box">✅ User {user_id} face enrolled successfully!</div>', 
                                              unsafe_allow_html=True)
                                    st.balloons()
                                    st.success("🎉 Enrollment complete! Click 'Capture New Image' to enroll another user.")
                                    # DON'T clear image or rerun - let user see the message
                                    # Mark as enrolled so we don't enroll twice
                                    st.session_state.face_enrolled = True
                                else:
                                    st.markdown('<div class="error-box">❌ Enrollment failed.</div>', 
                                              unsafe_allow_html=True)
                            except Exception as e:
                                st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', 
                                          unsafe_allow_html=True)
            
            with col2:
                if st.button("🔄 Capture New Image", key="recapture_face", use_container_width=True):
                    # Clear captured image and enrolled flag
                    st.session_state.face_captured_image = None
                    st.session_state.camera_locked = False
                    if 'face_enrolled' in st.session_state:
                        del st.session_state.face_enrolled
                    st.rerun()

def enroll_voice(user_id):
    """Voice enrollment section"""
    st.subheader("🎤 Voice Capture")
    
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Live Recording"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload voice audio", 
                                        type=['wav', 'mp3', 'flac', 'ogg'],
                                        help="Upload a clear voice recording (at least 2-3 seconds)",
                                        key="voice_enroll_upload")
        
        if uploaded_file is not None:
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Save uploaded file temporarily
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_audio_path = tmp_file.name
            
            try:
                # Extract embedding preview
                with st.spinner("Analyzing voice..."):
                    embedding = st.session_state.voice_system.extract_embedding(temp_audio_path)
                    
                    if embedding is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Embedding Dimensions", f"{len(embedding)}D")
                            st.metric("Embedding Norm", f"{np.linalg.norm(embedding):.2f}")
                        
                        with col2:
                            st.metric("Mean Value", f"{np.mean(embedding):.4f}")
                            st.metric("Std Deviation", f"{np.std(embedding):.4f}")
                        
                        st.success("✅ Voice analysis successful!")
                    else:
                        st.error("❌ Voice analysis failed. Please check audio quality.")
                
                # Enrollment button
                if st.button("✅ Enroll Voice", type="primary", use_container_width=True, key="enroll_voice_upload_btn"):
                    if not user_id:
                        st.error("❌ Please enter a User ID")
                    else:
                        with st.spinner("Enrolling voice..."):
                            try:
                                success = st.session_state.voice_system.enroll(user_id, temp_audio_path)
                                
                                if success:
                                    st.markdown(f'<div class="success-box">✅ User {user_id} voice enrolled successfully!</div>', 
                                              unsafe_allow_html=True)
                                    st.balloons()
                                    st.success("🎉 Enrollment complete! You can now verify or identify this user.")
                                else:
                                    st.markdown('<div class="error-box">❌ Enrollment failed. User may already exist or audio quality is poor.</div>', 
                                              unsafe_allow_html=True)
                            except Exception as e:
                                st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', 
                                          unsafe_allow_html=True)
            finally:
                # Clean up temp file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
    
    with tab2:
        st.info("🎙️ Record your voice (speak for 3-5 seconds)")
        
        # Use Streamlit's audio_input (requires Streamlit >= 1.28.0)
        audio_value = st.audio_input("Record your voice", key="voice_live_recording")
        
        if audio_value is not None:
            st.audio(audio_value)
            
            # Save recorded audio temporarily
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_value.getvalue())
                temp_audio_path = tmp_file.name
            
            try:
                # Extract embedding preview
                with st.spinner("Analyzing recorded voice..."):
                    embedding = st.session_state.voice_system.extract_embedding(temp_audio_path)
                    
                    if embedding is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Embedding Dimensions", f"{len(embedding)}D")
                            st.metric("Quality Score", f"{np.linalg.norm(embedding):.2f}")
                        
                        with col2:
                            # Simple quality check
                            if np.linalg.norm(embedding) > 100:
                                st.success("✅ Good audio quality")
                            else:
                                st.warning("⚠️ Low quality - speak louder/clearer")
                        
                        # Enrollment button
                        if st.button("✅ Enroll Recorded Voice", type="primary", 
                                   use_container_width=True, key="enroll_voice_live_btn"):
                            if not user_id:
                                st.error("❌ Please enter a User ID")
                            else:
                                with st.spinner("Enrolling voice..."):
                                    try:
                                        success = st.session_state.voice_system.enroll(user_id, temp_audio_path)
                                        
                                        if success:
                                            st.markdown(f'<div class="success-box">✅ User {user_id} voice enrolled successfully!</div>', 
                                                      unsafe_allow_html=True)
                                            st.balloons()
                                            st.success("🎉 Enrollment complete! Record another sample to enroll a new user.")
                                        else:
                                            st.markdown('<div class="error-box">❌ Enrollment failed. User may already exist.</div>', 
                                                      unsafe_allow_html=True)
                                    except Exception as e:
                                        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', 
                                                  unsafe_allow_html=True)
                    else:
                        st.error("❌ Voice analysis failed. Please record again with better quality.")
            finally:
                # Clean up temp file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
        else:
            st.info("👆 Click the microphone button above to start recording")

def show_verification():
    st.header("🔍 User Verification (1:1 Matching)")
    
    st.info("📌 Verify if the biometric matches the claimed user identity")
    
    # User ID input
    user_id = st.text_input("👤 Claimed User ID", 
                            placeholder="Enter the user ID to verify")
    
    # Select modality
    st.subheader("🔐 Select Biometric Modality")
    modality = st.radio(
        "Choose modality to verify:",
        ["👆 Fingerprint", "👁️ Iris", "😊 Face", "🎤 Voice"],
        horizontal=True,
        key="verify_modality"
    )
    
    if modality == "👆 Fingerprint":
        verify_fingerprint(user_id)
    elif modality == "👁️ Iris":
        verify_iris(user_id)
    elif modality == "😊 Face":
        verify_face(user_id)
    elif modality == "🎤 Voice":
        verify_voice(user_id)

def verify_fingerprint(user_id):
    """Fingerprint verification section"""
    st.subheader("👆 Fingerprint to Verify")
    
    uploaded_file = st.file_uploader("Upload fingerprint image", 
                                    type=['png', 'jpg', 'jpeg', 'bmp'],
                                    key="verify_upload")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        
        with col2:
            preprocessed = st.session_state.fingerprint_system.preprocess(img_gray)
            st.image(preprocessed, caption="Preprocessed", 
                    use_container_width=True, clamp=True)
        
        # Verification settings
        with st.expander("⚙️ Advanced Settings"):
            threshold = st.slider("Verification Threshold", 
                                min_value=0.0, max_value=1.0, 
                                value=0.3, step=0.05,
                                help="Lower = stricter verification")
        
        # Verify button
        if st.button("🔍 Verify Identity", type="primary", use_container_width=True):
            if not user_id:
                st.error("❌ Please enter a User ID")
            else:
                with st.spinner("Verifying..."):
                    time.sleep(0.5)  # Simulate processing
                    is_verified, confidence = st.session_state.fingerprint_system.verify(
                        user_id, img_gray, threshold
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Verification Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("User ID", user_id)
                    
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2%}")
                    
                    with col3:
                        status = "✅ VERIFIED" if is_verified else "❌ REJECTED"
                        st.metric("Status", status)
                    
                    # Result message
                    if is_verified:
                        st.markdown(f'<div class="success-box">✅ Identity Verified! Confidence: {confidence:.2%}</div>', 
                                  unsafe_allow_html=True)
                        st.success("Access Granted ✓")
                    else:
                        st.markdown(f'<div class="error-box">❌ Verification Failed! Confidence: {confidence:.2%}</div>', 
                                  unsafe_allow_html=True)
                        st.error("Access Denied ✗")

def verify_iris(user_id):
    """Iris verification section"""
    st.subheader("👁️ Iris to Verify")
    
    # Eye side selector for verification
    eye_side = st.radio(
        "Eye to Verify",
        options=["Auto (Check All)", "Left Eye", "Right Eye"],
        horizontal=True,
        help="Select which eye to verify, or Auto to check all enrolled eyes",
        key="iris_verify_eye_selector"
    )
    
    # Map display text to internal value
    eye_map = {"Auto (Check All)": "auto", "Left Eye": "left", "Right Eye": "right"}
    eye_value = eye_map[eye_side]
    
    uploaded_file = st.file_uploader("Upload iris image", 
                                    type=['png', 'jpg', 'jpeg', 'bmp'],
                                    key="verify_iris_upload")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        
        with col2:
            # Segment and show
            with st.spinner("Segmenting iris..."):
                seg_result, params = st.session_state.iris_system.segment_iris(img_gray)
                
                if seg_result is not None and params is not None:
                    vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                    cv2.circle(vis_img, params['iris_center'], params['iris_radius'], (0, 255, 0), 2)
                    cv2.circle(vis_img, params['pupil_center'], params['pupil_radius'], (255, 0, 0), 2)
                    st.image(vis_img, caption="Segmented", use_container_width=True)
                else:
                    st.error("❌ Iris segmentation failed")
        
        # Verification settings
        with st.expander("⚙️ Advanced Settings"):
            # Slider for similarity threshold (NOT Hamming distance)
            similarity_threshold = st.slider("Similarity Threshold", 
                                min_value=0.0, max_value=1.0, 
                                value=0.65, step=0.01,
                                help="Higher = stricter verification (recommended: 0.65)")
        
        # Verify button
        if st.button("🔍 Verify Iris", type="primary", use_container_width=True, key="verify_iris_btn"):
            if not user_id:
                st.error("❌ Please enter a User ID")
            else:
                with st.spinner(f"Verifying iris pattern ({eye_side.lower()})..."):
                    is_verified, confidence = st.session_state.iris_system.verify(
                        user_id, img_gray, similarity_threshold, eye_side=eye_value
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Verification Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("User ID", user_id)
                    
                    with col2:
                        # For iris, confidence is 1 - hamming_distance
                        st.metric("Similarity", f"{confidence:.2%}")
                    
                    with col3:
                        status = "✅ VERIFIED" if is_verified else "❌ REJECTED"
                        st.metric("Status", status)
                    
                    # Result message
                    if is_verified:
                        st.markdown(f'<div class="success-box">✅ Iris Verified! Similarity: {confidence:.2%}</div>', 
                                  unsafe_allow_html=True)
                        st.success("Access Granted ✓")
                    else:
                        st.markdown(f'<div class="error-box">❌ Verification Failed! Similarity: {confidence:.2%}</div>', 
                                  unsafe_allow_html=True)
                        st.error("Access Denied ✗")

def verify_face(user_id):
    """Face verification section"""
    st.subheader("😊 Face to Verify")
    
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Capture"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload face image", 
                                        type=['png', 'jpg', 'jpeg'],
                                        key="verify_face_upload")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert RGB if needed
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
            
            with col2:
                # Extract and show face
                with st.spinner("Detecting face..."):
                    try:
                        faces = st.session_state.face_system.extract_faces(img_array)
                        
                        if faces and len(faces) > 0:
                            face_img = faces[0]['face']
                            confidence = faces[0]['confidence']
                            
                            st.image(face_img, caption="Detected Face", use_container_width=True)
                            st.metric("Detection Confidence", f"{confidence:.2%}")
                        else:
                            st.error("❌ No face detected")
                    except Exception as e:
                        st.error(f"❌ Face detection failed: {str(e)}")
            
            # Verify button
            if st.button("🔍 Verify Face", type="primary", use_container_width=True, key="verify_face_upload_btn"):
                if not user_id:
                    st.error("❌ Please enter a User ID")
                else:
                    with st.spinner("Verifying face..."):
                        try:
                            is_verified, distance, details = st.session_state.face_system.verify(
                                user_id, img_array
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📊 Verification Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("User ID", user_id)
                            
                            with col2:
                                st.metric("Distance", f"{distance:.4f}")
                            
                            with col3:
                                status = "✅ VERIFIED" if is_verified else "❌ REJECTED"
                                st.metric("Status", status)
                            
                            # Result message
                            if is_verified:
                                st.markdown(f'<div class="success-box">✅ Face Verified! Distance: {distance:.4f}</div>', 
                                          unsafe_allow_html=True)
                                st.success("Access Granted ✓")
                            else:
                                st.markdown(f'<div class="error-box">❌ Verification Failed! Distance: {distance:.4f}</div>', 
                                          unsafe_allow_html=True)
                                st.error("Access Denied ✗")
                        except Exception as e:
                            st.error(f"❌ Verification error: {str(e)}")
    
    with tab2:
        st.info("📷 Click 'Capture from Webcam' to take a photo")
        
        if st.button("� Capture from Webcam", type="primary", use_container_width=True, key="verify_face_webcam_btn"):
            with st.spinner("Opening webcam..."):
                captured_image = st.session_state.face_system.capture_from_webcam(
                    window_name="Face Verification - Press SPACE to capture, ESC to cancel"
                )
                
                if captured_image is not None:
                    st.session_state.verify_face_webcam_image = captured_image
                    st.success("✅ Image captured!")
                    st.rerun()
                else:
                    st.error("❌ Webcam capture cancelled or failed")
        
        # Show captured image if exists
        if 'verify_face_webcam_image' in st.session_state and st.session_state.verify_face_webcam_image is not None:
            img_array = st.session_state.verify_face_webcam_image
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_array, caption="Captured Image", use_container_width=True)
            
            with col2:
                # Extract and show face
                with st.spinner("Detecting face..."):
                    try:
                        faces = st.session_state.face_system.extract_faces(img_array)
                        
                        if faces and len(faces) > 0:
                            face_img = faces[0]['face']
                            confidence = faces[0]['confidence']
                            
                            st.image(face_img, caption="Detected Face", use_container_width=True)
                            st.metric("Detection Confidence", f"{confidence:.2%}")
                        else:
                            st.error("❌ No face detected")
                    except Exception as e:
                        st.error(f"❌ Face detection failed: {str(e)}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Verify Captured Face", type="primary", use_container_width=True, key="verify_face_webcam_verify_btn"):
                    if not user_id:
                        st.error("❌ Please enter a User ID")
                    else:
                        with st.spinner("Verifying face..."):
                            try:
                                is_verified, distance, details = st.session_state.face_system.verify(
                                    user_id, img_array
                                )
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("📊 Verification Results")
                                
                                result_col1, result_col2, result_col3 = st.columns(3)
                                
                                with result_col1:
                                    st.metric("User ID", user_id)
                                
                                with result_col2:
                                    st.metric("Distance", f"{distance:.4f}")
                                
                                with result_col3:
                                    status = "✅ VERIFIED" if is_verified else "❌ REJECTED"
                                    st.metric("Status", status)
                                
                                # Result message
                                if is_verified:
                                    st.markdown(f'<div class="success-box">✅ Face Verified! Distance: {distance:.4f}</div>', 
                                              unsafe_allow_html=True)
                                    st.success("Access Granted ✓")
                                else:
                                    st.markdown(f'<div class="error-box">❌ Verification Failed! Distance: {distance:.4f}</div>', 
                                              unsafe_allow_html=True)
                                    st.error("Access Denied ✗")
                            except Exception as e:
                                st.error(f"❌ Verification error: {str(e)}")
            
            with col2:
                if st.button("🔄 Retake Photo", use_container_width=True, key="verify_face_retake_btn"):
                    del st.session_state.verify_face_webcam_image
                    st.rerun()


def verify_voice(user_id):
    """Voice verification section"""
    st.subheader("🎤 Voice to Verify")
    
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Live Recording"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload voice audio", 
                                        type=['wav', 'mp3', 'flac', 'ogg'],
                                        key="verify_voice_upload")
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            # Save temporarily
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_audio_path = tmp_file.name
            
            try:
                # Verify button
                if st.button("🔍 Verify Voice", type="primary", use_container_width=True, key="verify_voice_upload_btn"):
                    if not user_id:
                        st.error("❌ Please enter a User ID")
                    else:
                        with st.spinner("Verifying voice..."):
                            try:
                                is_verified, confidence = st.session_state.voice_system.verify(
                                    user_id, temp_audio_path
                                )
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("📊 Verification Results")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("User ID", user_id)
                                
                                with col2:
                                    st.metric("Similarity", f"{confidence:.4f}")
                                
                                with col3:
                                    status = "✅ VERIFIED" if is_verified else "❌ REJECTED"
                                    st.metric("Status", status)
                                
                                # Result message
                                if is_verified:
                                    st.markdown(f'<div class="success-box">✅ Voice Verified! Similarity: {confidence:.4f}</div>', 
                                              unsafe_allow_html=True)
                                    st.success("Access Granted ✓")
                                else:
                                    st.markdown(f'<div class="error-box">❌ Verification Failed! Similarity: {confidence:.4f}</div>', 
                                              unsafe_allow_html=True)
                                    st.error("Access Denied ✗")
                            except Exception as e:
                                st.error(f"❌ Verification error: {str(e)}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
    
    with tab2:
        st.info("🎙️ Record your voice to verify")
        
        audio_value = st.audio_input("Record your voice", key="voice_verify_recording")
        
        if audio_value is not None:
            st.audio(audio_value)
            
            # Save temporarily
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_value.getvalue())
                temp_audio_path = tmp_file.name
            
            try:
                # Verify button
                if st.button("🔍 Verify Recorded Voice", type="primary", use_container_width=True, key="verify_voice_live_btn"):
                    if not user_id:
                        st.error("❌ Please enter a User ID")
                    else:
                        with st.spinner("Verifying voice..."):
                            try:
                                is_verified, confidence = st.session_state.voice_system.verify(
                                    user_id, temp_audio_path
                                )
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("📊 Verification Results")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("User ID", user_id)
                                
                                with col2:
                                    st.metric("Similarity", f"{confidence:.4f}")
                                
                                with col3:
                                    status = "✅ VERIFIED" if is_verified else "❌ REJECTED"
                                    st.metric("Status", status)
                                
                                # Result message
                                if is_verified:
                                    st.markdown(f'<div class="success-box">✅ Voice Verified! Similarity: {confidence:.4f}</div>', 
                                              unsafe_allow_html=True)
                                    st.success("Access Granted ✓")
                                else:
                                    st.markdown(f'<div class="error-box">❌ Verification Failed! Similarity: {confidence:.4f}</div>', 
                                              unsafe_allow_html=True)
                                    st.error("Access Denied ✗")
                            except Exception as e:
                                st.error(f"❌ Verification error: {str(e)}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

def show_identification():
    st.header("🔎 User Identification (1:N Matching)")
    
    st.info("📌 Identify the user from biometric without knowing their ID")
    
    # Select modality
    st.subheader("🔐 Select Biometric Modality")
    modality = st.radio(
        "Choose modality to identify:",
        ["👆 Fingerprint", "👁️ Iris", "😊 Face", "🎤 Voice"],
        horizontal=True,
        key="identify_modality"
    )
    
    if modality == "👆 Fingerprint":
        identify_fingerprint()
    elif modality == "👁️ Iris":
        identify_iris()
    elif modality == "😊 Face":
        identify_face()
    elif modality == "🎤 Voice":
        identify_voice()

def identify_fingerprint():
    """Fingerprint identification section"""
    st.subheader("👆 Fingerprint to Identify")
    
    uploaded_file = st.file_uploader("Upload fingerprint image", 
                                    type=['png', 'jpg', 'jpeg', 'bmp'],
                                    key="identify_upload")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        
        with col2:
            preprocessed = st.session_state.fingerprint_system.preprocess(img_gray)
            st.image(preprocessed, caption="Preprocessed", 
                    use_container_width=True, clamp=True)
        
        # Settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Matching Threshold", 
                                    min_value=0.0, max_value=1.0, 
                                    value=0.3, step=0.05)
            with col2:
                top_n = st.number_input("Top N Results", 
                                       min_value=1, max_value=10, value=5)
        
        # Identify button
        if st.button("🔎 Identify User", type="primary", use_container_width=True):
            with st.spinner("Searching database..."):
                time.sleep(0.5)
                results = st.session_state.fingerprint_system.identify(
                    img_gray, threshold, top_n
                )
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Identification Results")
                
                if results:
                    # Create results table
                    import pandas as pd
                    
                    df = pd.DataFrame(results, columns=["User ID", "Match Score"])
                    df["Rank"] = range(1, len(df) + 1)
                    df["Match Score"] = df["Match Score"].apply(lambda x: f"{x:.2%}")
                    df = df[["Rank", "User ID", "Match Score"]]
                    
                    # Highlight top match
                    st.success(f"🎯 Top Match: **{results[0][0]}** (Confidence: {results[0][1]:.2%})")
                    
                    # Show table
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[r[0] for r in results],
                            y=[r[1] for r in results],
                            text=[f"{r[1]:.2%}" for r in results],
                            textposition='auto',
                            marker_color=['#4CAF50' if i == 0 else '#2196F3' 
                                        for i in range(len(results))]
                        )
                    ])
                    
                    fig.update_layout(
                        title="Match Scores Comparison",
                        xaxis_title="User ID",
                        yaxis_title="Match Score",
                        yaxis_tickformat='.0%',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No matches found in database")

def identify_iris():
    """Iris identification section"""
    st.subheader("👁️ Iris to Identify")
    
    uploaded_file = st.file_uploader("Upload iris image", 
                                    type=['png', 'jpg', 'jpeg', 'bmp'],
                                    key="identify_iris_upload")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        
        with col2:
            with st.spinner("Segmenting iris..."):
                seg_result, params = st.session_state.iris_system.segment_iris(img_gray)
                
                if seg_result is not None and params is not None:
                    vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                    cv2.circle(vis_img, params['iris_center'], params['iris_radius'], (0, 255, 0), 2)
                    cv2.circle(vis_img, params['pupil_center'], params['pupil_radius'], (255, 0, 0), 2)
                    st.image(vis_img, caption="Segmented", use_container_width=True)
                else:
                    st.error("❌ Iris segmentation failed")
        
        # Settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold = st.slider("Similarity Threshold", 
                                    min_value=0.0, max_value=1.0, 
                                    value=0.65, step=0.01,
                                    help="Higher = stricter matching (recommended: 0.65)",
                                    key="identify_iris_threshold")
            with col2:
                top_n = st.number_input("Top N Results", 
                                       min_value=1, max_value=10, value=5,
                                       key="identify_iris_topn")
        
        # Identify button
        if st.button("🔎 Identify Iris", type="primary", use_container_width=True, key="identify_iris_btn"):
            with st.spinner("Searching iris database..."):
                results = st.session_state.iris_system.identify(
                    img_gray, similarity_threshold, top_n
                )
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Identification Results")
                
                if results:
                    # Create results table
                    import pandas as pd
                    
                    df = pd.DataFrame(results, columns=["User ID", "Similarity"])
                    df["Rank"] = range(1, len(df) + 1)
                    df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.2%}")
                    df = df[["Rank", "User ID", "Similarity"]]
                    
                    # Highlight top match
                    st.success(f"🎯 Top Match: **{results[0][0]}** (Similarity: {results[0][1]:.2%})")
                    
                    # Show table
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[r[0] for r in results],
                            y=[r[1] for r in results],
                            text=[f"{r[1]:.2%}" for r in results],
                            textposition='auto',
                            marker_color=['#4CAF50' if i == 0 else '#2196F3' 
                                        for i in range(len(results))]
                        )
                    ])
                    
                    fig.update_layout(
                        title="Iris Match Similarity Comparison",
                        xaxis_title="User ID",
                        yaxis_title="Similarity Score",
                        yaxis_tickformat='.0%',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No matches found in database")

def identify_face():
    """Face identification section"""
    st.subheader("😊 Face to Identify")
    
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Capture"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload face image", 
                                        type=['png', 'jpg', 'jpeg'],
                                        key="identify_face_upload")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Convert RGB if needed
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)
            
            with col2:
                with st.spinner("Detecting face..."):
                    try:
                        faces = st.session_state.face_system.extract_faces(img_array)
                        
                        if faces and len(faces) > 0:
                            face_img = faces[0]['face']
                            confidence = faces[0]['confidence']
                            
                            st.image(face_img, caption="Detected Face", use_container_width=True)
                            st.metric("Detection Confidence", f"{confidence:.2%}")
                        else:
                            st.error("❌ No face detected")
                    except Exception as e:
                        st.error(f"❌ Face detection failed: {str(e)}")
            
            # Settings (simplified - only Top N)
            top_n = st.number_input("Top N Results", 
                                   min_value=1, max_value=10, value=5,
                                   key="identify_face_upload_topn",
                                   help="Number of top matches to return")
            
            # Identify button
            if st.button("🔎 Identify Face", type="primary", use_container_width=True, key="identify_face_upload_btn"):
                with st.spinner("Searching face database..."):
                    try:
                        # identify() returns list of (user_id, distance, details) tuples
                        results = st.session_state.face_system.identify(
                            img_array, top_n=top_n
                        )
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Identification Results")
                        
                        if results:
                            # Create results table
                            import pandas as pd
                            
                            # Extract user_id and distance from tuples (ignore details dict)
                            df = pd.DataFrame([(r[0], r[1]) for r in results], columns=["User ID", "Distance"])
                            df["Rank"] = range(1, len(df) + 1)
                            df["Distance"] = df["Distance"].apply(lambda x: f"{x:.4f}")
                            df = df[["Rank", "User ID", "Distance"]]
                            
                            # Highlight top match
                            st.success(f"🎯 Top Match: **{results[0][0]}** (Distance: {results[0][1]:.4f})")
                            
                            # Show table
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            # Visualization
                            import plotly.graph_objects as go
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=[r[0] for r in results],
                                    y=[r[1] for r in results],
                                    text=[f"{r[1]:.4f}" for r in results],
                                    textposition='auto',
                                    marker_color=['#4CAF50' if i == 0 else '#2196F3' 
                                                for i in range(len(results))]
                                )
                            ])
                            
                            fig.update_layout(
                                title="Face Match Distance Comparison",
                                xaxis_title="User ID",
                                yaxis_title="Distance Score",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ No matches found in database")
                    except Exception as e:
                        st.error(f"❌ Identification error: {str(e)}")
    
    with tab2:
        st.info("📷 Click 'Capture from Webcam' to take a photo")
        
        if st.button("📷 Capture from Webcam", type="primary", use_container_width=True, key="identify_face_webcam_btn"):
            with st.spinner("Opening webcam..."):
                captured_image = st.session_state.face_system.capture_from_webcam(
                    window_name="Face Identification - Press SPACE to capture, ESC to cancel"
                )
                
                if captured_image is not None:
                    st.session_state.identify_face_webcam_image = captured_image
                    st.success("✅ Image captured!")
                    st.rerun()
                else:
                    st.error("❌ Webcam capture cancelled or failed")
        
        # Show captured image if exists
        if 'identify_face_webcam_image' in st.session_state and st.session_state.identify_face_webcam_image is not None:
            img_array = st.session_state.identify_face_webcam_image
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_array, caption="Captured Image", use_container_width=True)
            
            with col2:
                with st.spinner("Detecting face..."):
                    try:
                        faces = st.session_state.face_system.extract_faces(img_array)
                        
                        if faces and len(faces) > 0:
                            face_img = faces[0]['face']
                            confidence = faces[0]['confidence']
                            
                            st.image(face_img, caption="Detected Face", use_container_width=True)
                            st.metric("Detection Confidence", f"{confidence:.2%}")
                        else:
                            st.error("❌ No face detected")
                    except Exception as e:
                        st.error(f"❌ Face detection failed: {str(e)}")
            
            # Settings
            top_n = st.number_input("Top N Results", 
                                   min_value=1, max_value=10, value=5,
                                   key="identify_face_webcam_topn",
                                   help="Number of top matches to return")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔎 Identify Captured Face", type="primary", use_container_width=True, key="identify_face_webcam_identify_btn"):
                    with st.spinner("Searching face database..."):
                        try:
                            results = st.session_state.face_system.identify(
                                img_array, top_n=top_n
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📊 Identification Results")
                            
                            if results:
                                # Create results table
                                import pandas as pd
                                
                                df = pd.DataFrame([(r[0], r[1]) for r in results], columns=["User ID", "Distance"])
                                df["Rank"] = range(1, len(df) + 1)
                                df["Distance"] = df["Distance"].apply(lambda x: f"{x:.4f}")
                                df = df[["Rank", "User ID", "Distance"]]
                                
                                st.success(f"🎯 Top Match: **{results[0][0]}** (Distance: {results[0][1]:.4f})")
                                st.dataframe(df, use_container_width=True, hide_index=True)
                                
                                # Visualization
                                import plotly.graph_objects as go
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=[r[0] for r in results],
                                        y=[r[1] for r in results],
                                        text=[f"{r[1]:.4f}" for r in results],
                                        textposition='auto',
                                        marker_color=['#4CAF50' if i == 0 else '#2196F3' 
                                                    for i in range(len(results))]
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Face Match Distance Comparison",
                                    xaxis_title="User ID",
                                    yaxis_title="Distance Score",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ No matches found in database")
                        except Exception as e:
                            st.error(f"❌ Identification error: {str(e)}")
            
            with col2:
                if st.button("🔄 Retake Photo", use_container_width=True, key="identify_face_retake_btn"):
                    del st.session_state.identify_face_webcam_image
                    st.rerun()

def identify_voice():
    """Voice identification section"""
    st.subheader("🎤 Voice to Identify")
    
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Live Recording"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload voice audio", 
                                        type=['wav', 'mp3', 'flac', 'ogg'],
                                        key="identify_voice_upload")
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            # Save temporarily
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_audio_path = tmp_file.name
            
            try:
                # Settings - Top N
                top_n = st.number_input("Top N Results", 
                                       min_value=1, max_value=10, value=5,
                                       key="identify_voice_upload_topn",
                                       help="Number of top matches to return")
                
                # Identify button
                if st.button("🔎 Identify Voice", type="primary", use_container_width=True, key="identify_voice_upload_btn"):
                    with st.spinner("Searching voice database..."):
                        try:
                            # identify() returns list of (user_id, similarity) tuples
                            results = st.session_state.voice_system.identify(
                                temp_audio_path, top_n=top_n
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📊 Identification Results")
                            
                            if results:
                                # Create results table
                                import pandas as pd
                                
                                df = pd.DataFrame(results, columns=["User ID", "Similarity"])
                                df["Rank"] = range(1, len(df) + 1)
                                df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")
                                df = df[["Rank", "User ID", "Similarity"]]
                                
                                # Highlight top match
                                st.success(f"🎯 Top Match: **{results[0][0]}** (Similarity: {results[0][1]:.4f})")
                                
                                # Show table
                                st.dataframe(df, use_container_width=True, hide_index=True)
                                
                                # Visualization
                                import plotly.graph_objects as go
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=[r[0] for r in results],
                                        y=[r[1] for r in results],
                                        text=[f"{r[1]:.4f}" for r in results],
                                        textposition='auto',
                                        marker_color=['#4CAF50' if i == 0 else '#2196F3' 
                                                    for i in range(len(results))]
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Voice Match Similarity Comparison",
                                    xaxis_title="User ID",
                                    yaxis_title="Similarity Score",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ No matches found in database")
                        except Exception as e:
                            st.error(f"❌ Identification error: {str(e)}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
    
    with tab2:
        st.info("🎙️ Record your voice to identify")
        
        audio_value = st.audio_input("Record your voice", key="voice_identify_recording")
        
        if audio_value is not None:
            st.audio(audio_value)
            
            # Save temporarily
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_value.getvalue())
                temp_audio_path = tmp_file.name
            
            try:
                # Settings - Top N
                top_n = st.number_input("Top N Results", 
                                       min_value=1, max_value=10, value=5,
                                       key="identify_voice_live_topn",
                                       help="Number of top matches to return")
                
                # Identify button
                if st.button("🔎 Identify Recorded Voice", type="primary", use_container_width=True, key="identify_voice_live_btn"):
                    with st.spinner("Searching voice database..."):
                        try:
                            # identify() returns list of (user_id, similarity) tuples
                            results = st.session_state.voice_system.identify(
                                temp_audio_path, top_n=top_n
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📊 Identification Results")
                            
                            if results:
                                # Create results table
                                import pandas as pd
                                
                                df = pd.DataFrame(results, columns=["User ID", "Similarity"])
                                df["Rank"] = range(1, len(df) + 1)
                                df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.4f}")
                                df = df[["Rank", "User ID", "Similarity"]]
                                
                                # Highlight top match
                                st.success(f"🎯 Top Match: **{results[0][0]}** (Similarity: {results[0][1]:.4f})")
                                
                                # Show table
                                st.dataframe(df, use_container_width=True, hide_index=True)
                                
                                # Visualization
                                import plotly.graph_objects as go
                                
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=[r[0] for r in results],
                                        y=[r[1] for r in results],
                                        text=[f"{r[1]:.4f}" for r in results],
                                        textposition='auto',
                                        marker_color=['#4CAF50' if i == 0 else '#2196F3' 
                                                    for i in range(len(results))]
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Voice Match Similarity Comparison",
                                    xaxis_title="User ID",
                                    yaxis_title="Similarity Score",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("⚠️ No matches found in database")
                        except Exception as e:
                            st.error(f"❌ Identification error: {str(e)}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

def show_settings():
    st.header("⚙️ System Settings")
    
    # Fingerprint settings
    st.subheader("👆 Fingerprint Module Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("SIFT Features", value=500, step=50, 
                       help="Number of SIFT keypoints to extract")
        st.number_input("CLAHE Clip Limit", value=2.0, step=0.1,
                       help="Contrast limiting for CLAHE preprocessing")
    
    with col2:
        st.number_input("CLAHE Grid Size", value=8, step=1,
                       help="Tile grid size for CLAHE")
        st.number_input("Gaussian Blur Size", value=5, step=2,
                       help="Kernel size for denoising")
    
    st.markdown("---")
    
    # Database management
    st.subheader("💾 Database Management")
    
    # Select modality
    db_modality = st.radio(
        "Select Database:",
        ["👆 Fingerprint", "👁️ Iris", "😊 Face", "🎤 Voice"],
        horizontal=True,
        key="db_modality"
    )
    
    if db_modality == "👆 Fingerprint":
        system = st.session_state.fingerprint_system
        db_name = "Fingerprint"
    elif db_modality == "👁️ Iris":
        system = st.session_state.iris_system
        db_name = "Iris"
    elif db_modality == "😊 Face":
        system = st.session_state.face_system
        db_name = "Face"
    else:  # Voice
        system = st.session_state.voice_system
        db_name = "Voice"
    
    # Show enrolled users
    users = system.list_enrolled_users()
    st.metric(f"📊 Total {db_name} Users", len(users))
    
    if users:
        # Show enrolled users with eye details for Iris
        if db_modality == "👁️ Iris":
            st.write("**Enrolled Users:**")
            for user_id in users:
                enrolled_eyes = list(system.templates[user_id]['eyes'].keys())
                eye_labels = [e.capitalize() for e in enrolled_eyes]
                st.write(f"- **{user_id}**: {', '.join(eye_labels)}")
        else:
            st.write(f"**Enrolled Users:** {', '.join(users)}")
        
        # Delete individual user
        st.write("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_to_delete = st.selectbox(
                f"Select user to delete from {db_name} database:",
                options=users,
                key=f"delete_{db_modality}"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🗑️ Delete User", type="secondary", use_container_width=True, key=f"btn_delete_{db_modality}"):
                if system.delete_user(user_to_delete):
                    st.success(f"✅ User '{user_to_delete}' deleted from {db_name} database!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to delete user '{user_to_delete}'")
        
        # Clear entire database
        st.write("---")
        st.warning("⚠️ **Danger Zone**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Delete ALL users from {db_name} database (cannot be undone)")
        with col2:
            if st.button("�️ Clear Database", type="primary", use_container_width=True, key=f"btn_clear_{db_modality}"):
                if st.session_state.get(f'confirm_clear_{db_modality}', False):
                    system.clear_database()
                    st.success(f"✅ {db_name} database cleared!")
                    st.session_state[f'confirm_clear_{db_modality}'] = False
                    st.rerun()
                else:
                    st.session_state[f'confirm_clear_{db_modality}'] = True
                    st.warning("⚠️ Click again to confirm deletion of ALL users!")
    else:
        st.info(f"No users enrolled in {db_name} database yet.")
    
    st.markdown("---")
    
    # System info
    st.subheader("ℹ️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **👆 Fingerprint Module**
        
        - Database: `{st.session_state.fingerprint_system.database_path}`
        - Preprocessing: CLAHE + Gaussian Blur
        - Feature Extraction: SIFT (Scale-Invariant Feature Transform)
        - Matching: FLANN-based KNN with Lowe's ratio test
        """)
    
    with col2:
        st.info(f"""
        **👁️ Iris Module**
        
        - Database: `{st.session_state.iris_system.database_path}`
        - Segmentation: Hough Circle Transform
        - Normalization: Daugman's Rubber Sheet Model
        - Feature Extraction: Gabor Wavelets (4 orientations)
        - Matching: Hamming Distance with rotation handling
        """)
    
    with col3:
        st.info(f"""
        **😊 Face Module**
        
        - Database: `{st.session_state.face_system.database_path}`
        - Model: VGG-Face (default)
        - Detector: RetinaFace
        - Distance Metric: Cosine Similarity
        - Framework: DeepFace
        """)

if __name__ == "__main__":
    main()
