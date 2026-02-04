import sys
import mediapipe as mp

try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    test = mp_face_mesh.FaceMesh()
    print("✅ SUCCESS: AI Face Mesh Model Loaded Successfully!")
except Exception as e:
    print("❌ ERROR:", e)
    print("\nSystem Path:", sys.path)