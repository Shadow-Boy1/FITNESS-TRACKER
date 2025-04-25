import streamlit as st
import os
import sys
import cv2
import tempfile
import av
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro
import pyttsx3
import queue
import threading
from collections import deque

st.title("🏋️‍♂️ Workout Tracker")

# Top-level menu
main_menu = st.selectbox("Choose a workout type", ["Select", "Squats", "Deadlift", "Biceps Curl"])

if main_menu == "Squats":
    squat_option = st.radio("Squats Menu", ["📡 Live Stream", "📤 Upload Video"])

    # 📡 Live Stream Feature
    if squat_option == "📡 Live Stream":
        st.subheader("Squats - Live Stream")

        class SquatsLiveStreamProcessor(VideoTransformerBase):
            def __init__(self):
                self.pose = get_mediapipe_pose()
                self.process_frame = ProcessFrame(thresholds=get_thresholds_beginner(), flip_frame=True)

            def recv(self, frame):
                # Convert frame to ndarray
                frame = frame.to_ndarray(format="bgr24")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing

                # Process the frame
                frame, _ = self.process_frame.process(frame, self.pose)

                # Convert back to BGR and return
                return av.VideoFrame.from_ndarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), format="bgr24")

        webrtc_streamer(
            key="squats-live-stream",
            video_processor_factory=SquatsLiveStreamProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )






    # 📤 Upload Video Feature
    elif squat_option == "📤 Upload Video":
        st.subheader("Squats - Upload Video")

        # Set up paths and thresholds
        BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
        sys.path.append(BASE_DIR)

        st.title('AI Fitness Trainer: Squats Analysis')

        mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)

        thresholds = None
        if mode == 'Beginner':
            thresholds = get_thresholds_beginner()
        elif mode == 'Pro':
            thresholds = get_thresholds_pro()

        upload_process_frame = ProcessFrame(thresholds=thresholds)

        # Initialize pose detection
        pose = get_mediapipe_pose()

        download = None

        if 'download' not in st.session_state:
            st.session_state['download'] = False

        output_video_file = f'output_recorded.mp4'

        if os.path.exists(output_video_file):
            os.remove(output_video_file)

        with st.form('Upload', clear_on_submit=True):
            up_file = st.file_uploader("Upload a Video", ['mp4', 'mov', 'avi'])
            uploaded = st.form_submit_button("Upload")

        stframe = st.empty()

        ip_vid_str = '<p style="font-family:Helvetica; font-weight: bold; font-size: 16px;">Input Video</p>'
        warning_str = '<p style="font-family:Helvetica; font-weight: bold; color: Red; font-size: 17px;">Please Upload a Video first!!!</p>'

        warn = st.empty()

        download_button = st.empty()

        if up_file and uploaded:
            download_button.empty()
            tfile = tempfile.NamedTemporaryFile(delete=False)

            try:
                warn.empty()
                tfile.write(up_file.read())

                vf = cv2.VideoCapture(tfile.name)

                # ---------------------  Write the processed video frame. --------------------
                fps = int(vf.get(cv2.CAP_PROP_FPS))
                width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)
                # ----------------------------------------------------------------------------- 

                txt = st.sidebar.markdown(ip_vid_str, unsafe_allow_html=True)
                ip_video = st.sidebar.video(tfile.name)

                while vf.isOpened():
                    ret, frame = vf.read()
                    if not ret:
                        break

                    # convert frame from BGR to RGB before processing it.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out_frame, _ = upload_process_frame.process(frame, pose)
                    stframe.image(out_frame)
                    video_output.write(out_frame[..., ::-1])

                vf.release()
                video_output.release()
                stframe.empty()
                ip_video.empty()
                txt.empty()
                tfile.close()

            except AttributeError:
                warn.markdown(warning_str, unsafe_allow_html=True)

        if os.path.exists(output_video_file):
            with open(output_video_file, 'rb') as op_vid:
                download = download_button.download_button('Download Video', data=op_vid, file_name='output_recorded.mp4')

            if download:
                st.session_state['download'] = True

        if os.path.exists(output_video_file) and st.session_state['download']:
            os.remove(output_video_file)
            st.session_state['download'] = False
            download_button.empty()







elif main_menu == "Deadlift":
    deadlift_option = st.radio("Deadlift Menu", ["📡 Live Stream", "📤 Upload Video"])

    if deadlift_option == "📡 Live Stream":
        st.subheader("Deadlift - Live Stream")

        class DeadliftLiveStreamProcessor(VideoTransformerBase):
            def __init__(self):
                self.pose = mp.solutions.pose.Pose()
                self.drawing = mp.solutions.drawing_utils

            def recv(self, frame):
                frame = frame.to_ndarray(format="bgr24")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

                if results.pose_landmarks:
                    self.drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                return av.VideoFrame.from_ndarray(frame, format="bgr24")

        webrtc_streamer(
            key="deadlift-live-stream",
            video_processor_factory=DeadliftLiveStreamProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    elif deadlift_option == "📤 Upload Video":
        st.subheader("Deadlift - Upload Video")

        # Define Pose Detection
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        # Function to calculate angle between three points
        def calculate_angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba, bc = a - b, c - b
            angle = np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0))
            return np.degrees(angle)

        # Threshold settings
        def get_thresholds_beginner():
            return {'hip_angle': 160, 'knee_angle': 170}

        def get_thresholds_pro():
            return {'hip_angle': 140, 'knee_angle': 160}

        # Frame Processor Class
        class ProcessFrame:
            def __init__(self, thresholds):
                self.thresholds = thresholds
                self.correct_reps = 0
                self.incorrect_reps = 0
                self.stage = "up"
                self.user_in_view = False
                self.reached_bottom = False

            def is_full_body_in_view(self, landmarks):
                required_parts = [
                    mp_pose.PoseLandmark.NOSE,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.RIGHT_KNEE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE
                ]
                return all(landmarks[part.value].visibility > 0.5 for part in required_parts)

            def process(self, frame, pose):
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                feedback = ""
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    if not self.user_in_view and self.is_full_body_in_view(landmarks):
                        self.user_in_view = True
                        feedback = "✅ Full body detected, ready to start!"

                    if self.user_in_view:
                        # Get keypoints
                        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        hip_angle = calculate_angle(shoulder, hip, knee)
                        knee_angle = calculate_angle(hip, knee, ankle)

                        # Down position
                        if hip_angle < 90 and knee_angle < 100:
                            self.stage = "down"
                            self.reached_bottom = True
                            feedback = "⬇️ Full depth reached!"

                        # Back to standing
                        elif hip_angle > self.thresholds['hip_angle'] and knee_angle > self.thresholds['knee_angle']:
                            if self.stage == "down":
                                if self.reached_bottom:
                                    self.correct_reps += 1
                                    feedback = "✅ Correct rep!"
                                else:
                                    self.incorrect_reps += 1
                                    feedback = "❌ Incomplete range!"
                                self.stage = "up"
                                self.reached_bottom = False
                        else:
                            feedback = "🟡 Mid-rep in progress..."

                        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    feedback = "🔍 No user detected."

                # Overlay
                cv2.putText(frame, f"Correct Reps: {self.correct_reps}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Incorrect Reps: {self.incorrect_reps}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, feedback, (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                return frame, feedback

        # Streamlit UI for Deadlift Upload Video
        mode = st.radio("Select difficulty level:", ["Beginner", "Pro"], horizontal=True)
        uploaded = False
        video_file = None
        stframe = st.empty()

        with st.form("Upload", clear_on_submit=True):
            video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
            uploaded = st.form_submit_button("Upload")

        if uploaded and video_file:
            thresholds = get_thresholds_beginner() if mode == "Beginner" else get_thresholds_pro()
            processor = ProcessFrame(thresholds=thresholds)

            output_path = "output_recorded.mp4"
            if os.path.exists(output_path):
                os.remove(output_path)

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            vf = cv2.VideoCapture(tfile.name)

            fps = int(vf.get(cv2.CAP_PROP_FPS))
            width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break

                processed_frame, _ = processor.process(frame, pose)
                stframe.image(processed_frame, channels="BGR")
                video_writer.write(processed_frame)

            vf.release()
            video_writer.release()

            st.success("✅ Processing complete!")

            with open(output_path, 'rb') as f:
                st.download_button("⬇️ Download Processed Video", f, file_name="processed_deadlift.mp4")

elif main_menu == "Biceps Curl":
    biceps_option = st.radio("Biceps Curl Menu", ["📡 Live Stream", "📤 Upload Video"])

    if biceps_option == "📡 Live Stream":
        st.subheader("Biceps Curl - Live Stream")

        import pyttsx3
        import threading
        import queue
        import cv2
        import numpy as np
        import mediapipe as mp
        import time

        # Voice Feedback System
        class VoiceCoach:
            def __init__(self):
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.queue = queue.Queue()
                self.running = True
                self.thread = threading.Thread(target=self._process_queue)
                self.thread.start()

            def _process_queue(self):
                while self.running:
                    try:
                        msg = self.queue.get(timeout=1)
                        self.engine.say(msg)
                        self.engine.runAndWait()
                    except queue.Empty:
                        continue

            def say(self, message):
                self.queue.put(message)

            def stop(self):
                self.running = False
                self.thread.join()

        # Session vars
        for var in ["cap", "voice_coach", "counter", "stage", "session_active"]:
            if var not in st.session_state:
                st.session_state[var] = None if var in ["cap", "voice_coach"] else False if var == "session_active" else 0

        # UI layout
        col1, col2 = st.columns(2)
        with col1:
            mode = st.radio("Select difficulty level:", ["Beginner", "Advanced"], index=0, horizontal=True)
            run = st.checkbox("🎥 Start Camera")
            frame_window = st.empty()
        with col2:
            st.image("https://i.imgur.com/JqYeWZn.png", caption="Proper Biceps Curl Form")
            st.markdown("""
            **Biceps Curl Cues**  
            • Elbows pinned to sides  
            • Full curl range  
            • Avoid swinging  
            • Controlled tempo  
            """)
            if st.button("🔄 Reset Counters"):
                st.session_state.counter = 0
                st.session_state.stage = None
                st.rerun()

        st.markdown("---")
        st.markdown("""
        ### 🚨 Common Mistakes
        - 🔴 Elbows flaring  
        - 🔴 Incomplete curl  
        - 🔴 Using momentum  
        - 🔴 Leaning backward  
        """)

        # Pose setup
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=2)
        drawing = mp.solutions.drawing_utils
        LANDMARK_STYLE = drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
        CONNECTION_STYLE = drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)

        def calculate_angle(a, b, c):
            a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180.0 else angle

        def validate_position(landmarks):
            try:
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                dx = r_shoulder.x - l_shoulder.x
                dy = r_shoulder.y - l_shoulder.y
                angle = np.degrees(np.arctan2(-dy, dx)) % 180
                left_dist = abs(nose.x - l_shoulder.x)
                right_dist = abs(nose.x - r_shoulder.x)
                ratio = min(left_dist, right_dist) / max(left_dist, right_dist) if max(left_dist, right_dist) > 0 else 0
                return (160 < angle < 200) and (0.65 < ratio < 0.95)
            except:
                return False

        # Handle Camera Toggle
        if run and not st.session_state.session_active:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.voice_coach = VoiceCoach()
            st.session_state.counter = 0
            st.session_state.stage = None
            st.session_state.session_active = True
            time.sleep(1)

        elif not run and st.session_state.session_active:
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            if st.session_state.voice_coach:
                st.session_state.voice_coach.stop()
                st.session_state.voice_coach = None
            st.session_state.session_active = False

        # Camera loop
        if st.session_state.session_active and st.session_state.cap and st.session_state.cap.isOpened():
            cap = st.session_state.cap
            voice_coach = st.session_state.voice_coach
            rep_display = st.empty()
            score_display = st.empty()

            while run and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠️ Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                form_score_value = 0
                valid_position = False

                if results.pose_landmarks:
                    valid_position = validate_position(results.pose_landmarks.landmark)

                    if valid_position:
                        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                        angle = calculate_angle(shoulder, elbow, wrist)

                        if angle > 160:
                            st.session_state.stage = "Down"
                        if angle < 50 and st.session_state.stage == "Down":
                            st.session_state.stage = "Up"
                            st.session_state.counter += 1
                            voice_coach.say(f"Rep {st.session_state.counter}")

                        form_score_value = 100
                        drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                               LANDMARK_STYLE, CONNECTION_STYLE)
                        cv2.putText(image, f"{int(angle)}°",
                                    (int(elbow.x * image.shape[1]), int(elbow.y * image.shape[0])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    else:
                        form_score_value = 60
                        cv2.putText(image, "GOOD ENOUGH POSITION", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                frame_window.image(image, channels="BGR", use_column_width=True)
                rep_display.metric("Reps", st.session_state.counter)
                score_display.metric("Form Score", f"{form_score_value}%")

                if not run:
                    break

            # Clean up after loop
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            if st.session_state.voice_coach:
                st.session_state.voice_coach.stop()
                st.session_state.voice_coach = None
            st.session_state.session_active = False











elif main_menu == "Pushups":
    pushups_option = st.radio("Pushups Menu", ["📡 Live Stream", "📤 Upload Video"])

    if pushups_option == "📡 Live Stream":
        st.subheader("Pushups - Live Stream")
        # Add live stream logic for Pushups here

    elif pushups_option == "📤 Upload Video":
        st.subheader("Pushups - Upload Video")
        # Add upload video logic for Pushups here

