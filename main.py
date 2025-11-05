"""
Multimodal Biometric Authentication System
Main Entry Point
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Import modules
from modules import FingerprintRecognition, IrisRecognition
from config import settings

def initialize_system():
    """Initialize the multimodal biometric system"""
    print("=" * 60)
    print("Multimodal Biometric Authentication System")
    print("=" * 60)
    
    # Create necessary directories
    settings.create_directories()
    
    # Initialize modules
    print("\n📌 Initializing biometric modules...")
    fp_system = FingerprintRecognition(database_path=str(settings.FINGERPRINT_DB))
    iris_system = IrisRecognition(database_path=str(settings.IRIS_DB))
    
    return {
        'fingerprint': fp_system,
        'iris': iris_system
    }

def demo_fingerprint_enrollment(fp_system):
    """Demo: Enroll a fingerprint"""
    print("\n" + "="*60)
    print("DEMO: Fingerprint Enrollment")
    print("="*60)
    
    # Example: Create a synthetic fingerprint for demo
    # In practice, you would load real fingerprint images
    print("\n⚠️  Please add fingerprint images to data/raw/fingerprints/")
    print("Example usage:")
    print("  fingerprint = cv2.imread('data/raw/fingerprints/user001.png', cv2.IMREAD_GRAYSCALE)")
    print("  fp_system.enroll('user001', fingerprint)")

def demo_fingerprint_verification(fp_system):
    """Demo: Verify a fingerprint"""
    print("\n" + "="*60)
    print("DEMO: Fingerprint Verification")
    print("="*60)
    
    stats = fp_system.get_statistics()
    print(f"\n📊 Current database: {stats['total_users']} users enrolled")
    
    if stats['total_users'] == 0:
        print("\n⚠️  No users enrolled yet. Please enroll users first.")
        return
    
    print("\nExample usage:")
    print("  fingerprint = cv2.imread('data/raw/fingerprints/test.png', cv2.IMREAD_GRAYSCALE)")
    print("  is_verified, confidence = fp_system.verify('user001', fingerprint)")
    print("  print(f'Verified: {is_verified}, Confidence: {confidence:.2f}')")

def demo_fingerprint_identification(fp_system):
    """Demo: Identify from fingerprint"""
    print("\n" + "="*60)
    print("DEMO: Fingerprint Identification")
    print("="*60)
    
    stats = fp_system.get_statistics()
    print(f"\n📊 Current database: {stats['total_users']} users enrolled")
    
    if stats['total_users'] == 0:
        print("\n⚠️  No users enrolled yet. Please enroll users first.")
        return
    
    print("\nExample usage:")
    print("  fingerprint = cv2.imread('data/raw/fingerprints/unknown.png', cv2.IMREAD_GRAYSCALE)")
    print("  results = fp_system.identify(fingerprint)")
    print("  for user_id, score in results:")
    print("      print(f'User: {user_id}, Score: {score:.2f}')")

def demo_iris_enrollment(iris_system):
    """Demo: Enroll an iris"""
    print("\n" + "="*60)
    print("DEMO: Iris Enrollment")
    print("="*60)
    
    print("\n⚠️  Please add iris images to data/raw/iris/")
    print("Example usage:")
    print("  iris_img = cv2.imread('data/raw/iris/user001.bmp', cv2.IMREAD_GRAYSCALE)")
    print("  iris_system.enroll('user001', iris_img)")
    print("\nNote: Image should be a close-up of the eye (recommended: 240x320 or similar)")

def demo_iris_verification(iris_system):
    """Demo: Verify an iris"""
    print("\n" + "="*60)
    print("DEMO: Iris Verification")
    print("="*60)
    
    stats = iris_system.get_statistics()
    print(f"\n📊 Current database: {stats['total_users']} users enrolled")
    
    if stats['total_users'] == 0:
        print("\n⚠️  No users enrolled yet. Please enroll users first.")
        return
    
    print("\nExample usage:")
    print("  iris_img = cv2.imread('data/raw/iris/test.bmp', cv2.IMREAD_GRAYSCALE)")
    print("  is_verified, similarity = iris_system.verify('user001', iris_img)")
    print("  print(f'Verified: {is_verified}, Similarity: {similarity:.2%}')")

def demo_iris_identification(iris_system):
    """Demo: Identify from iris"""
    print("\n" + "="*60)
    print("DEMO: Iris Identification")
    print("="*60)
    
    stats = iris_system.get_statistics()
    print(f"\n📊 Current database: {stats['total_users']} users enrolled")
    
    if stats['total_users'] == 0:
        print("\n⚠️  No users enrolled yet. Please enroll users first.")
        return
    
    print("\nExample usage:")
    print("  iris_img = cv2.imread('data/raw/iris/unknown.bmp', cv2.IMREAD_GRAYSCALE)")
    print("  results = iris_system.identify(iris_img)")
    print("  for user_id, similarity in results:")
    print("      print(f'User: {user_id}, Similarity: {similarity:.2%}')")

def multimodal_verify(user_id, fingerprint, face, iris, voice):
    scores = []
    
    # Get scores from each modality
    if fingerprint:
        _, fp_score = fingerprint_system.verify(user_id, fingerprint)
        scores.append(fp_score)
    
    if face:
        _, face_score = face_system.verify(user_id, face)
        scores.append(face_score)
    
    # ... iris, voice
    
    # Decision rule: 3/4 modalities must pass
    passed = sum(1 for s in scores if s >= threshold)
    return passed >= 3, np.mean(scores)

def show_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print("MAIN MENU")
    print("="*60)
    print("\nFingerprint Module:")
    print("  1. Enroll User (Fingerprint)")
    print("  2. Verify User (Fingerprint)")
    print("  3. Identify User (Fingerprint)")
    print("\nIris Module:")
    print("  4. Enroll User (Iris)")
    print("  5. Verify User (Iris)")
    print("  6. Identify User (Iris)")
    print("\nSystem:")
    print("  7. Show Statistics")
    print("  8. Run Development Notebook")
    print("  9. Exit")
    print("\n" + "="*60)

def show_statistics(systems):
    """Show system statistics"""
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    
    # Fingerprint statistics
    if 'fingerprint' in systems:
        fp_stats = systems['fingerprint'].get_statistics()
        print(f"\n📌 Fingerprint Module:")
        print(f"   Total users: {fp_stats['total_users']}")
        print(f"   Database: {fp_stats['database_path']}")
    
    # Iris statistics
    if 'iris' in systems:
        iris_stats = systems['iris'].get_statistics()
        print(f"\n📌 Iris Module:")
        print(f"   Total users: {iris_stats['total_users']}")
        print(f"   Database: {iris_stats['database_path']}")
    
    # Future modules
    print(f"\n📌 Face Module: Coming soon...")
    print(f"📌 Voice Module: Coming soon...")

def run_notebook():
    """Open development notebook"""
    print("\n📓 Opening Jupyter Notebook...")
    print("Run this command:")
    print("  jupyter notebook notebooks/01_fingerprint_development.ipynb")

def main():
    """Main application"""
    # Initialize system
    systems = initialize_system()
    fp_system = systems['fingerprint']
    iris_system = systems['iris']
    
    # Main loop
    while True:
        show_menu()
        choice = input("\nSelect option (1-9): ").strip()
        
        if choice == '1':
            demo_fingerprint_enrollment(fp_system)
        elif choice == '2':
            demo_fingerprint_verification(fp_system)
        elif choice == '3':
            demo_fingerprint_identification(fp_system)
        elif choice == '4':
            demo_iris_enrollment(iris_system)
        elif choice == '5':
            demo_iris_verification(iris_system)
        elif choice == '6':
            demo_iris_identification(iris_system)
        elif choice == '7':
            show_statistics(systems)
        elif choice == '8':
            run_notebook()
        elif choice == '9':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

