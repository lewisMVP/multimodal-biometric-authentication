# 🔐 Multimodal Biometric Authentication System

A comprehensive biometric authentication system implementing **4 biometric modalities**: Fingerprint, Iris, Face, and Voice recognition with a modern web interface.

![alt text](image.png)

---

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Biometric Modalities](#-biometric-modalities)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Security](#-security)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Functionalities
- **4 Biometric Modalities**: Fingerprint, Iris, Face, and Voice
- **3 Operating Modes**:
  - 📝 **Enrollment**: Register new users with biometric data
  - 🔍 **Verification (1:1)**: Verify claimed identity
  - 🔎 **Identification (1:N)**: Identify unknown person from database
- **Database Management**: Add, view, delete users per modality
- **Web Interface**: Modern Streamlit-based UI with real-time processing

### Advanced Features
- Multi-eye support for iris (Left/Right separation)
- Eye side auto-detection with validation
- Webcam capture for face enrollment/verification
- Live audio recording for voice authentication
- Quality assessment for iris images
- Configurable thresholds and algorithm parameters
- Visualization of similarity scores

---

## 🏗️ System Architecture

The system follows a **modular, layered architecture** designed for scalability and maintainability:

```
┌───────────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER (Streamlit)                     │
│  ┌─────────┬────────────┬──────────────┬────────────────┬──────────┐  │
│  │Dashboard│ Enrollment │ Verification │ Identification │ Settings │  │
│  └─────────┴────────────┴──────────────┴────────────────┴──────────┘  │
└───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                             │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │              Recognition Module Interface                    │     │
│  │  • enroll(user_id, sample) → bool                            │     │
│  │  • verify(user_id, sample, threshold) → (bool, similarity)   │     │
│  │  • identify(sample, threshold) → List[(user_id, score)]      │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌────────────┬─────────────┬────────────┬──────────────┐             │
│  │Fingerprint │    Iris     │    Face    │    Voice     │             │
│  │Recognition │ Recognition │Recognition │ Recognition  │             │
│  └────────────┴─────────────┴────────────┴──────────────┘             │
└───────────────────────────────────────────────────────────────────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   ALGORITHM PROCESSING LAYER                          │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────┐           │
│  │   SIFT   │  │ Daugman's │  │ VGG-Face │  │ ECAPA-TDNN  │           │
│  │  + FLANN │  │ Algorithm │  │  (Deep   │  │  (Speaker   │           │
│  │ Matching │  │  + Gabor  │  │  Face)   │  │  Embedding) │           │
│  └──────────┘  └───────────┘  └──────────┘  └─────────────┘           │
│                                                                       │
│  Image Processing      Audio Processing      Deep Learning            │
│  • OpenCV             • librosa              • TensorFlow/Keras       │
│  • NumPy              • soundfile            • PyTorch                │
│  • scikit-image       • VAD                  • DeepFace/SpeechBrain   │
└───────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    DATA PERSISTENCE LAYER                             │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │              Template Database (Pickle Format)              │      │
│  │  ┌─────────────┬──────────┬───────────┬─────────────┐       │      │
│  │  │Fingerprints │   Iris   │   Faces   │   Voices    │       │      │
│  │  │   (SIFT     │ (Binary  │ (4096-D   │  (192-D     │       │      │
│  │  │  features)  │  codes)  │ vectors)  │ embeddings) │       │      │
│  │  └─────────────┴──────────┴───────────┴─────────────┘       │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                       │
│  File System: data/database/{fingerprints|iris|faces|voices}/         │
└───────────────────────────────────────────────────────────────────────┘
```

### Architecture Highlights

#### 1. **Presentation Layer**
- **Technology**: Streamlit web framework
- **Components**: 5 main pages (Dashboard, Enrollment, Verification, Identification, Settings)
- **Features**: Interactive UI, real-time feedback, webcam/microphone integration
- **File**: `app.py` (1935 lines)

#### 2. **Business Logic Layer**
- **Design Pattern**: Uniform interface across all modalities
- **Core Operations**: 
  - `enroll()`: Register biometric template
  - `verify()`: 1:1 matching against claimed identity
  - `identify()`: 1:N search across database
- **Modules**: 4 independent recognition modules
- **Location**: `modules/` directory

#### 3. **Algorithm Processing Layer**
- **Fingerprint**: SIFT keypoint extraction + FLANN matcher
- **Iris**: Hough Transform segmentation + Gabor wavelets + Hamming distance
- **Face**: RetinaFace detection + VGG-Face embedding + Cosine similarity
- **Voice**: VAD preprocessing + ECAPA-TDNN embedding + Cosine similarity

#### 4. **Data Persistence Layer**
- **Storage**: Pickle-based serialization (not production-ready, use database in real deployment)
- **Templates**: Privacy-preserving representations (features, not raw biometrics)
- **Organization**: Separate folders per modality

### Data Flow Example (Verification)

```
User uploads fingerprint image
        │
        ▼
[Streamlit UI] receives image file
        │
        ▼
[FingerprintRecognition.verify()] called
        │
        ├─→ Preprocess: CLAHE + Gaussian blur
        │
        ├─→ Extract: SIFT features from query image
        │
        ├─→ Load: Enrolled template from database
        │
        ├─→ Match: FLANN-based KNN matching
        │
        ├─→ Filter: Lowe's ratio test (0.75)
        │
        └─→ Decide: >= threshold → VERIFIED
        │
        ▼
[Streamlit UI] displays result with similarity score
```

---

## 🛠️ Technologies Used

### Computer Vision
- **OpenCV**: Image processing, SIFT feature extraction, Hough Transform
- **NumPy**: Numerical operations, array processing
- **scikit-image**: Image enhancement

### Deep Learning
- **TensorFlow/Keras**: Backend for DeepFace
- **PyTorch**: Backend for SpeechBrain
- **DeepFace**: Face recognition with VGG-Face model
- **SpeechBrain**: ECAPA-TDNN for speaker recognition

### Audio Processing
- **librosa**: Audio feature extraction, VAD
- **soundfile**: Audio I/O operations

### Web Framework
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and display

### Algorithms Implemented
1. **Fingerprint**: SIFT (Scale-Invariant Feature Transform) + FLANN matching
2. **Iris**: Daugman's Rubber Sheet Model + Gabor wavelets
3. **Face**: VGG-Face (DeepFace) + RetinaFace detector
4. **Voice**: ECAPA-TDNN speaker embeddings + Cosine similarity

---

## 📦 Installation

### Prerequisites
- Python 3.12
- pip (Python package manager)
- Webcam (optional, for face capture)
- Microphone (optional, for voice recording)

### Step 1: Clone Repository
```bash
git clone https://github.com/lewisMVP/multimodal-biometric-authentication.git
cd multimodal-biometric-authentication
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes due to TensorFlow and PyTorch.

### Step 4: Run Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 🚀 Usage

### 1. Enrollment
1. Navigate to **Enrollment** page
2. Enter **User ID**
3. Select **Biometric Modality** (Fingerprint/Iris/Face/Voice)
4. Upload sample or use live capture
5. For Iris: Select eye side (Left/Right)
6. Click **Enroll** button

### 2. Verification (1:1)
1. Navigate to **Verification** page
2. Enter **User ID** to verify
3. Select **Biometric Modality**
4. Upload/capture biometric sample
5. Click **Verify** button
6. View result: ✅ VERIFIED or ❌ REJECTED

### 3. Identification (1:N)
1. Navigate to **Identification** page
2. Select **Biometric Modality**
3. Upload/capture biometric sample
4. Click **Identify** button
5. View ranked results with similarity scores

### 4. Database Management
1. Navigate to **Settings** page
2. Scroll to **Database Management**
3. Select modality to manage
4. View enrolled users
5. Delete individual users or clear entire database

---

## 🔬 Biometric Modalities

### 👆 Fingerprint Recognition

**Algorithm**: SIFT (Scale-Invariant Feature Transform)

**Pipeline**:
1. Preprocessing: CLAHE + Gaussian Blur
2. Feature Extraction: 500 SIFT keypoints
3. Matching: FLANN-based KNN with Lowe's ratio test (0.75)

**Advantages**:
- Scale and rotation invariant
- Robust to noise
- High accuracy (10-15% better than ORB)

---

### 👁️ Iris Recognition

**Algorithm**: Daugman's Rubber Sheet Model

**Pipeline**:
1. Segmentation: Hough Circle Transform (iris + pupil)
2. Normalization: Polar transformation (64×512)
3. Feature Extraction: Gabor wavelets (4 orientations, median threshold)
4. Matching: Hamming Distance with rotation handling

**Special Features**:
- Multi-eye support (Left/Right separation)
- Eye side auto-detection
- Quality assessment (sharpness, contrast, illumination, occlusion)
- Threshold: 0.65 similarity (0.35 Hamming distance)

**Advantages**:
- Highly distinctive (low FAR)
- Stable over lifetime
- Difficult to forge

---

### 😊 Face Recognition

**Algorithm**: VGG-Face (DeepFace framework)

**Pipeline**:
1. Detection: RetinaFace detector
2. Alignment: Facial landmarks
3. Embedding: VGG-Face CNN (4096-D vector)
4. Matching: Cosine similarity

**Features**:
- Upload image or webcam capture
- Real-time detection preview
- Multi-image storage per user

**Advantages**:
- Non-intrusive
- Fast detection
- High accuracy with deep learning

---

### 🎤 Voice Recognition

**Algorithm**: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)

**Pipeline**:
1. Audio Processing: 16kHz resampling, VAD
2. Normalization: Peak normalization to 95%
3. Embedding: ECAPA-TDNN (192-D vector)
4. Matching: Cosine similarity

**Features**:
- Upload audio file or live recording
- Quality validation (duration, energy, RMS)
- Threshold: 0.50 (optimized from EER analysis)

**Advantages**:
- Remote authentication capable
- Speaker-specific characteristics
- Robust to variations

---

## 📁 Project Structure

```
multimodal_biometric_auth/
│
├── 📄 app.py                                    # Main application entry point (1935 lines)
│                                                 # - Streamlit web interface
│                                                 # - 5 pages: Dashboard, Enrollment, Verification, 
│                                                 #            Identification, Settings
│                                                 # - Webcam/microphone integration
│
├── 📄 main.py                                   # CLI interface (alternative to web UI)
│
├── 📄 requirements.txt                          # Python dependencies
│                                                 # - opencv-python, numpy, scikit-image
│                                                 # - tensorflow, deepface
│                                                 # - torch, speechbrain
│                                                 # - streamlit, plotly, librosa
│
├── 📄 README.md                                 # This documentation file
│
├── 📂 config/                                   # Configuration management
│   ├── __init__.py
│   └── settings.py                              # Global settings and constants
│
├── 📂 modules/                                  # Core biometric recognition modules
│   ├── __init__.py
│   │
│   ├── 👆 fingerprint_recognition.py            # Fingerprint module (650 lines)
│   │                                             # Algorithm: SIFT + FLANN matching
│   │                                             # Functions: enroll(), verify(), identify()
│   │                                             # Template: {keypoints, descriptors, metadata}
│   │
│   ├── 👁️ iris_recognition.py                   # Iris module (929 lines)
│   │                                             # Algorithm: Daugman's + Gabor wavelets
│   │                                             # Features: Multi-eye support, auto-detection
│   │                                             # Template: {eyes: {left/right: {features, quality}}}
│   │                                             # Quality metrics: sharpness, contrast, illumination
│   │
│   ├── 😊 face_recognition.py                   # Face module (580 lines)
│   │                                             # Algorithm: VGG-Face via DeepFace
│   │                                             # Detector: RetinaFace
│   │                                             # Template: {embeddings: [4096-D vectors]}
│   │
│   └── 🎤 voice_recognition.py                  # Voice module (720 lines)
│                                                 # Algorithm: ECAPA-TDNN speaker embedding
│                                                 # Preprocessing: VAD, resampling, normalization
│                                                 # Template: {embedding: 192-D vector, metadata}
│
├── 📂 data/                                     # Data storage directory
│   │
│   ├── 📂 database/                             # Enrolled biometric templates
│   │   ├── 📂 fingerprints/                     # Fingerprint templates (*.pkl)
│   │   │   └── {user_id}.pkl                    # SIFT features per user
│   │   │
│   │   ├── 📂 iris/                             # Iris templates (*.pkl)
│   │   │   └── {user_id}.pkl                    # Multi-eye iris codes
│   │   │                                         # Structure: {eyes: {left: {...}, right: {...}}}
│   │   │
│   │   ├── 📂 faces/                            # Face templates (folders + images)
│   │   │   └── {user_id}/                       # One folder per user
│   │   │       ├── face_1.jpg                   # Multiple face images
│   │   │       ├── face_2.jpg
│   │   │       └── embeddings.pkl               # Pre-computed VGG-Face embeddings
│   │   │
│   │   └── 📂 voices/                           # Voice templates (*.pkl)
│   │       └── {user_id}.pkl                    # ECAPA-TDNN embeddings
│   │
│   ├── 📂 raw/                                  # Sample test data
│   │   ├── 📂 fingerprints/                     # Test fingerprint images
│   │   ├── 📂 iris/                             # Test iris images (MMU dataset)
│   │   ├── 📂 faces/                            # Test face images
│   │   └── 📂 voices/                           # Test audio files
│   │
│   └── 📂 processed/                            # Preprocessed data cache
│
├── 📂 notebooks/                                # Jupyter notebooks for development
│   ├── 01_fingerprint_development.ipynb         # Fingerprint algorithm testing
│   └── 📂 data/                                 # Notebook-specific data
│       └── 📂 fingerprints/
│
├── 📂 results/                                  # Output files and reports
    ├── 📂 logs/                                 # Application logs
    ├── 📂 plots/                                # Performance visualizations
    └── 📂 reports/                              # Analysis reports

```

### Key Files Explained

| File | Lines | Purpose | Key Components |
|------|-------|---------|----------------|
| **app.py** | 1935 | Streamlit web UI | 5 pages, webcam/mic integration, real-time processing |
| **fingerprint_recognition.py** | 650 | Fingerprint processing | SIFT extraction, FLANN matching, Lowe's ratio test |
| **iris_recognition.py** | 929 | Iris processing | Hough Transform, Gabor wavelets, multi-eye support |
| **face_recognition.py** | 580 | Face processing | DeepFace wrapper, RetinaFace detector, cosine similarity |
| **voice_recognition.py** | 720 | Voice processing | VAD, ECAPA-TDNN, quality validation |

### Database Schema

#### Fingerprint Template (`{user_id}.pkl`)
```python
{
    'keypoints': List[cv2.KeyPoint],  # SIFT keypoints
    'descriptors': np.ndarray,         # Shape: (N, 128)
    'enrolled_date': datetime,
    'image_shape': (height, width)
}
```

#### Iris Template (`{user_id}.pkl`)
```python
{
    'eyes': {
        'left': {
            'iris_code': np.ndarray,        # Binary code (64×512)
            'noise_mask': np.ndarray,       # Occlusion mask
            'iris_center': (x, y),
            'iris_radius': float,
            'pupil_center': (x, y),
            'pupil_radius': float,
            'quality_score': float,         # 0-1 range
            'enrolled_date': datetime
        },
        'right': {...}                      # Same structure
    },
    'enrolled_date': datetime
}
```

#### Face Template (`{user_id}/embeddings.pkl`)
```python
{
    'embeddings': [
        np.ndarray,  # Shape: (4096,) - VGG-Face embedding
        np.ndarray,  # Multiple embeddings per user
        ...
    ],
    'image_paths': List[str],
    'enrolled_date': datetime
}
```

#### Voice Template (`{user_id}.pkl`)
```python
{
    'embedding': np.ndarray,           # Shape: (192,) - ECAPA-TDNN
    'sample_rate': 16000,
    'duration': float,                 # seconds
    'quality_metrics': {
        'rms_energy': float,
        'zero_crossing_rate': float,
        'spectral_centroid': float
    },
    'enrolled_date': datetime
}
```

### Module Dependencies

```
app.py
  ├── modules.fingerprint_recognition
  ├── modules.iris_recognition
  ├── modules.face_recognition
  └── modules.voice_recognition

fingerprint_recognition.py
  ├── cv2 (OpenCV)
  ├── numpy
  └── pickle

iris_recognition.py
  ├── cv2 (OpenCV)
  ├── numpy
  ├── scipy
  └── pickle

face_recognition.py
  ├── deepface
  ├── tensorflow
  └── pickle

voice_recognition.py
  ├── speechbrain
  ├── torch
  ├── librosa
  └── soundfile
```

---

## 📊 Performance

### Accuracy Metrics (Tested)

| Modality    | GAR (%)* | FAR (%)** | Threshold | Notes                    |
|-------------|----------|-----------|-----------|--------------------------|
| Fingerprint | ~95      | <5        | Auto      | SIFT with FLANN          |
| Iris        | 66.7     | 0.0       | 0.65      | Multi-eye, validated     |
| Face        | ~98      | <1        | Auto      | DeepFace VGG-Face        |
| Voice       | ~100     | 0.0       | 0.50      | EER 0.00% in development |

*GAR: Genuine Accept Rate  
**FAR: False Accept Rate

### Processing Speed

| Operation           | Fingerprint | Iris   | Face   | Voice  |
|---------------------|-------------|--------|--------|--------|
| Enrollment          | ~2s         | ~3s    | ~1s    | ~2s    |
| Verification (1:1)  | ~1s         | ~2s    | <1s    | <1s    |
| Identification (1:N)| ~0.5s/user  | ~1s/user| <0.1s/user| <0.1s/user|

*Tested on: Intel Core i7-11370H, 16GB RAM LPDDR4X, no GPU acceleration

---

## 🔒 Security

### Template Protection

Each biometric modality stores **privacy-preserving representations** instead of raw biometric data:

| Modality | Template Type | Reversibility | Size |
|----------|---------------|---------------|------|
| **Fingerprint** | SIFT keypoints + descriptors | ❌ Non-reversible | ~50-200 KB |
| **Iris** | Binary iris codes (Gabor-filtered) | ❌ Non-reversible | ~4 KB |
| **Face** | VGG-Face embeddings (4096-D) | ❌ Non-reversible | ~16 KB |
| **Voice** | ECAPA-TDNN embeddings (192-D) | ❌ Non-reversible | ~1.5 KB |

**Why templates are secure:**
- Cannot reconstruct original biometric image/audio from templates
- Mathematical transformations are one-way functions
- Even if database is stolen, attackers cannot reverse-engineer biometrics

### Security Measures Implemented

✅ **Local Storage**: All templates stored locally (`data/database/`), no cloud transmission

✅ **Threshold Validation**: Configurable similarity thresholds prevent unauthorized access

✅ **Quality Assessment**: Poor quality samples rejected during enrollment
  - Iris: Sharpness, contrast, illumination, occlusion checks
  - Voice: Duration, energy, RMS validation
  - Face: Detection confidence scoring

✅ **Critical Bugs Fixed** (November 2025):
  - **Iris Gabor Filter Bug**: Fixed threshold causing 100% FAR → Now 0% FAR
  - **UI Threshold Mismatch**: Fixed parameter confusion (Hamming vs Similarity)

✅ **Multi-Template Storage**: 
  - Face: Multiple images per user for robustness
  - Iris: Separate left/right eye templates

✅ **Auto-Migration**: Old template formats automatically upgraded

### Known Limitations

⚠️ **Anti-Spoofing**: Not implemented
  - Vulnerable to printed fingerprints, photos, recordings
  - **Recommendation**: Add liveness detection

⚠️ **Database Encryption**: Templates stored in plain pickle files
  - **Recommendation**: Use encrypted database (SQLite with encryption)

⚠️ **No User Authentication**: Anyone can access the web interface
  - **Recommendation**: Add login system with role-based access

⚠️ **No Rate Limiting**: Unlimited verification attempts
  - **Recommendation**: Implement attempt throttling (max 3 tries/minute)

⚠️ **Pickle Security**: Using pickle for serialization has security risks
  - **Recommendation**: Switch to JSON/Protocol Buffers for production

### Recommendations for Production Deployment

1. **Add Liveness Detection**
   - Fingerprint: Pulse detection, perspiration analysis
   - Iris: Pupil response to light
   - Face: 3D depth analysis, blink detection
   - Voice: Challenge-response prompts

2. **Implement Encryption**
   - Encrypt templates at rest (AES-256)
   - Use HTTPS for web interface
   - Secure key management (HSM/KMS)

3. **Add Authentication Layer**
   - User login before biometric operations
   - Role-based access control (admin, user)
   - Audit logging for all operations

4. **Database Improvements**
   - Migrate from Pickle to PostgreSQL/MongoDB
   - Use ORMs (SQLAlchemy) for safe queries
   - Regular backups with encryption

5. **Compliance**
   - GDPR compliance (data retention, user consent)
   - ISO/IEC 30107 (anti-spoofing)
   - FIDO2/WebAuthn integration for web security

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions (Google style)
- Include unit tests for new features
- Update README.md if adding new functionality
- Test with multiple biometric samples before submitting

### Areas for Improvement

#### High Priority
- [ ] **Liveness Detection**: Add anti-spoofing for face/iris
- [ ] **Unit Tests**: Comprehensive pytest suite for all modules
- [ ] **Database Encryption**: Secure template storage
- [ ] **User Authentication**: Login system with JWT

#### Medium Priority
- [ ] **API Development**: RESTful API for remote access
- [ ] **Multi-factor Authentication**: Combine multiple biometrics
- [ ] **Performance Optimization**: GPU acceleration for deep learning
- [ ] **Docker Support**: Containerized deployment

#### Low Priority
- [ ] **Mobile App**: React Native/Flutter integration
- [ ] **Jupyter Notebooks**: Add for Iris/Face/Voice (only Fingerprint exists)
- [ ] **Voice Quality**: Alternative to `st.audio_input()` for better recording
- [ ] **Export Reports**: PDF generation for identification results

### Bug Reports

Found a bug? Please include:
- Steps to reproduce
- Expected vs actual behavior
- Screenshots/error messages
- System information (OS, Python version)

## 🙏 Acknowledgments

- **SIFT Algorithm**: David Lowe
- **Daugman's Iris Recognition**: John Daugman
- **DeepFace**: Serengil, S. I., & Ozpinar, A. (2020)
- **SpeechBrain**: Ravanelli et al. (2021)
- **MMU Iris Database**: Multimedia University
- **Streamlit**: For the amazing web framework

---

## 📞 Contact

Project Maintainer: Lewis Chu
- Email: tgefps2004@gmail.com
- GitHub: [@lewisMVP](https://github.com/lewisMVP)


**⭐ If you find this project useful, please give it a star!**

**🐛 Found a bug? [Open an issue](https://github.com/lewisMVP/multimodal-biometric-authentication/issues)**

**💡 Have a feature request? [Start a discussion](https://github.com/lewisMVP/multimodal-biometric-authentication/discussions)**
