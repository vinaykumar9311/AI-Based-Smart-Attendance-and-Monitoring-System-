# AI-Based-Smart-Attendance-and-Monitoring-System-

Smart Attendance System using Face Recognition
A Python project for automatic attendance marking using OpenCV and face recognition.
Features
Face detection using Haar Cascade
Face recognition with feature matching
Automatic attendance logging to CSV
Real-time camera feed processing
Employee registration system
Requirements
Python 3.7+
OpenCV
NumPy
Install dependencies:
plain
pip install -r requirements.txt
How to Run
Run the system:
plain
python attendance_system.py
Register a new employee:
Press 'r' during runtime
Enter Employee ID and Name
Press SPACE to capture face
View attendance report:
plain
python attendance_system.py --report
Project Structure
plain
.
├── attendance_system.py    # Main application
├── requirements.txt        # Dependencies
├── dataset/               # Employee face images
├── attendance_records/    # Daily CSV reports
└── face_data.pkl         # Cached face encodings
How It Works
Camera captures video feed
Haar Cascade detects faces in each frame
Face features are extracted and normalized
Features are compared with stored employee data
If match found, attendance is logged to CSV
Green box = recognized, Red box = unknown
Notes
Place employee photos in dataset/ folder with format: ID_Name.jpg
The system creates a cache file (face_data.pkl) for faster loading
Attendance records are saved as CSV files in attendance_records/
Each employee is marked only once per day
Future Improvements
Use deep learning models (FaceNet, etc.) for better accuracy
Add GUI interface
Database integration (SQLite/MySQL)
Email notifications for absentees
Mobile app support
Author
[Your Name]
[Your Roll Number]
[Department]
[College Name]
