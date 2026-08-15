import cv2
import numpy as np
import os

def create_sample_face(name, emp_id):
    """Create a simple face image for testing"""
    img = np.ones((250, 250, 3), dtype=np.uint8) * 240

    # Draw face oval
    cv2.ellipse(img, (125, 140), (60, 80), 0, 0, 360, (200, 160, 120), -1)

    # Eyes
    cv2.circle(img, (100, 120), 10, (40, 40, 40), -1)
    cv2.circle(img, (150, 120), 10, (40, 40, 40), -1)
    cv2.circle(img, (100, 120), 3, (255, 255, 255), -1)
    cv2.circle(img, (150, 120), 3, (255, 255, 255), -1)

    # Nose
    cv2.line(img, (125, 130), (120, 155), (180, 140, 100), 3)
    cv2.line(img, (120, 155), (130, 155), (180, 140, 100), 3)

    # Mouth
    cv2.ellipse(img, (125, 175), (25, 12), 0, 0, 180, (150, 50, 50), 2)

    # Hair (random color based on name)
    np.random.seed(hash(name) % 100)
    hair = tuple(np.random.randint(20, 80, 3).tolist())

    # Top hair
    points = np.array([
        [60, 140], [70, 80], [125, 50], [180, 80], [190, 140]
    ], np.int32)
    cv2.fillPoly(img, [points], hair)

    # Save
    os.makedirs("dataset", exist_ok=True)
    filename = f"dataset/{emp_id}_{name.replace(' ', '_')}.jpg"
    cv2.imwrite(filename, img)
    return filename

if __name__ == "__main__":
    print("Creating sample face images...")

    employees = [
        ("EMP001", "Rahul Sharma"),
        ("EMP002", "Priya Patel"),
        ("EMP003", "Amit Kumar"),
    ]

    for emp_id, name in employees:
        path = create_sample_face(name, emp_id)
        print(f"Created: {path}")

    print("Done! Run 'python attendance_system.py' to test.")