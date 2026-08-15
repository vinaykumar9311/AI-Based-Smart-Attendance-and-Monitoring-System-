import cv2
import numpy as np
import os
import csv
from datetime import datetime
import pickle

# Student: [Your Name]
# Roll No: [Your Roll Number]
# Project: Smart Attendance System using Face Recognition
# Subject: Artificial Intelligence / Computer Vision

class SmartAttendance:
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.known_ids = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.load_faces()

    def load_faces(self):
        """Load all registered faces from dataset folder"""
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            print("Dataset folder created. Add face images to register employees.")
            return

        # Try loading from cache first
        if os.path.exists("face_data.pkl"):
            try:
                with open("face_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.known_faces = data["faces"]
                    self.known_names = data["names"]
                    self.known_ids = data["ids"]
                print(f"Loaded {len(self.known_names)} faces from cache")
                return
            except:
                pass

        # Load from images
        for filename in os.listdir(dataset_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                parts = filename.split("_", 1)
                if len(parts) == 2:
                    emp_id = parts[0]
                    name = parts[1].rsplit(".", 1)[0].replace("_", " ")

                    img_path = os.path.join(dataset_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (100, 100))

                        # Simple feature vector
                        features = face_roi.flatten().astype(np.float32)
                        features = features / np.linalg.norm(features)

                        self.known_faces.append(features)
                        self.known_names.append(name)
                        self.known_ids.append(emp_id)

        print(f"Loaded {len(self.known_names)} faces")

        # Save cache
        if self.known_faces:
            with open("face_data.pkl", "wb") as f:
                pickle.dump({
                    "faces": self.known_faces,
                    "names": self.known_names,
                    "ids": self.known_ids
                }, f)

    def get_face_features(self, gray_face):
        """Extract features from a face image"""
        resized = cv2.resize(gray_face, (100, 100))
        features = resized.flatten().astype(np.float32)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features

    def recognize_face(self, frame):
        """Detect and recognize faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            features = self.get_face_features(face_roi)

            name = "Unknown"
            emp_id = None
            confidence = 0

            if len(self.known_faces) > 0:
                distances = []
                for known in self.known_faces:
                    dist = np.linalg.norm(features - known)
                    distances.append(dist)

                min_dist = min(distances)
                idx = distances.index(min_dist)

                # Threshold for recognition
                if min_dist < 0.6:
                    name = self.known_names[idx]
                    emp_id = self.known_ids[idx]
                    confidence = 1 - min_dist

            results.append({
                "box": (x, y, w, h),
                "name": name,
                "id": emp_id,
                "confidence": confidence
            })

        return results

    def mark_attendance(self, emp_id, name):
        """Mark attendance in CSV file"""
        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"attendance_records/{today}.csv"

        # Check if already marked today
        already_marked = False
        if os.path.exists(filename):
            with open(filename, "r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row and row[0] == emp_id:
                        already_marked = True
                        break

        if already_marked:
            return False

        # Write to CSV
        file_exists = os.path.exists(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Emp_ID", "Name", "Date", "Time", "Status"])
            writer.writerow([
                emp_id,
                name,
                today,
                datetime.now().strftime("%H:%M:%S"),
                "Present"
            ])

        return True

    def run(self):
        """Main attendance loop"""
        print("\n" + "="*50)
        print("   SMART ATTENDANCE SYSTEM")
        print("   Using OpenCV Face Recognition")
        print("="*50)
        print("\nPress 'q' to quit")
        print("Press 'r' to register new employee")
        print("="*50 + "\n")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        marked_today = set()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.recognize_face(frame)

                for res in results:
                    x, y, w, h = res["box"]
                    name = res["name"]
                    emp_id = res["id"]
                    conf = res["confidence"]

                    # Draw box
                    if name != "Unknown":
                        color = (0, 255, 0)

                        # Mark attendance
                        if emp_id and emp_id not in marked_today:
                            if self.mark_attendance(emp_id, name):
                                marked_today.add(emp_id)
                                print(f"[PRESENT] {name} ({emp_id}) - {conf:.0%}")
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{name}"
                    if conf > 0:
                        label += f" {conf:.0%}"

                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show count
                cv2.putText(frame, f"Marked: {len(marked_today)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Smart Attendance", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.register_employee(cap)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nSystem stopped. Attendance saved.")

    def register_employee(self, cap):
        """Register a new employee"""
        print("\n--- Register New Employee ---")
        emp_id = input("Employee ID: ").strip()
        name = input("Full Name: ").strip()

        if not emp_id or not name:
            print("ID and Name required!")
            return

        print("Look at camera and press SPACE to capture...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            display = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (255, 255, 0), 2)

            cv2.putText(display, "Press SPACE to capture, ESC to cancel", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Smart Attendance", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = frame[y:y+h, x:x+w]

                    filename = f"dataset/{emp_id}_{name.replace(' ', '_')}.jpg"
                    cv2.imwrite(filename, face_img)
                    print(f"Saved: {filename}")

                    # Add to known faces
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    features = self.get_face_features(gray_face)
                    self.known_faces.append(features)
                    self.known_names.append(name)
                    self.known_ids.append(emp_id)

                    # Update cache
                    with open("face_data.pkl", "wb") as f:
                        pickle.dump({
                            "faces": self.known_faces,
                            "names": self.known_names,
                            "ids": self.known_ids
                        }, f)

                    print("Registration successful!")
                    break
                else:
                    print("No face detected. Try again.")
            elif key == 27:  # ESC
                print("Registration cancelled.")
                break


def show_report():
    """Display attendance report"""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_records/{today}.csv"

    print("\n" + "="*50)
    print(f"   ATTENDANCE REPORT - {today}")
    print("="*50)

    if not os.path.exists(filename):
        print("No attendance records found for today.")
        return

    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"\n{'ID':<10} {'Name':<20} {'Time':<10} {'Status':<10}")
        print("-" * 50)
        for row in reader:
            if row:
                print(f"{row[0]:<10} {row[1]:<20} {row[3]:<10} {row[4]:<10}")

    print("-" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        show_report()
    else:
        app = SmartAttendance()
        app.run()