import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

# Function to draw text boxes
def draw_text_box(image, text, position, box_color, text_color):
    x, y = position
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(image, (x, y - 30), (x + w + 20, y + 10), box_color, -1)
    cv2.putText(image, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    return image

# Counter and feedback variables
correct_reps = 0
incorrect_reps = 0
stage = None
user_in_frame = False

# Function to process frame
def process_frame(image):
    global correct_reps, incorrect_reps, stage, user_in_frame
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        user_in_frame = True
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        
        # Get key points
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
        # Calculate angles
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        
        # Feedback and rep counting
        feedback = ""
        if hip_angle > 160 and knee_angle > 170:
            feedback = "Fully extended. Lower down to start a rep."
            if stage == "up":
                stage = "down"
        elif hip_angle < 80 and knee_angle < 100:
            feedback = "Bottom position reached! Stand back up."
            if stage == "down":
                stage = "up"
                correct_reps += 1
        elif stage == "down" or stage == "up":  # Only count incorrect if a deadlift is attempted
            feedback = "Maintain a neutral spine and controlled movement."
            incorrect_reps += 1  # Count as incorrect rep if posture isn't ideal
        
        # Draw UI elements
        image = draw_text_box(image, f'CORRECT: {correct_reps}', (400, 50), (0, 255, 0), (255, 255, 255))
        image = draw_text_box(image, f'INCORRECT: {incorrect_reps}', (400, 100), (0, 0, 255), (255, 255, 255))
        image = draw_text_box(image, feedback, (50, 400), (0, 165, 255), (255, 255, 255))
    else:
        user_in_frame = False
    
    return image

# Streamlit App
st.title("AI Deadlift Pose Estimation")
run = st.checkbox("Start Camera")
frame_window = st.image([])

cap = cv2.VideoCapture(0)
while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Error accessing webcam!")
        break
    
    frame = process_frame(frame)
    frame_window.image(frame, channels="BGR")

cap.release()
st.write("Stop Camera to exit.")
