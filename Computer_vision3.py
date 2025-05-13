import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import datetime
import math
import threading


class ShoulderMotionTrainer:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_mediapipe()
        self.setup_state()
        self.create_ui()
        self.setup_camera()
        self.start_session()

    def setup_window(self):
        self.root.title("Shoulder Abduction/Adduction Trainer")
        self.root.configure(bg="#f0f2f5")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

    def setup_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def setup_state(self):
        self.motion_tasks = [
            {"label": "Shoulder Abduction", "type": "Abduction", "icon": "🔼"},
            {"label": "Shoulder Adduction", "type": "Adduction", "icon": "🔽"}
        ]
        self.current_task_index = 0
        self.trail = deque(maxlen=60)
        self.shoulder_angles = deque(maxlen=30)
        self.phase = "prompt"
        self.score = 0
        self.total_attempts = 0
        self.session_start = datetime.datetime.now()
        self.debug_mode = True
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.last_display_time = time.time()
        self.hand_path_color = (255, 0, 255)
        self.min_movement_threshold = 30
        self.initial_angle = None

    def create_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(header_frame, text="Shoulder Abduction/Adduction Trainer",
                  font=("Helvetica", 24, "bold")).pack(side=tk.LEFT)

        self.score_label = ttk.Label(header_frame, text="Score: 0/0",
                                     font=("Helvetica", 16))
        self.score_label.pack(side=tk.RIGHT)

        self.timer_label = ttk.Label(header_frame, text="Session time: 00:00",
                                     font=("Helvetica", 16))
        self.timer_label.pack(side=tk.RIGHT, padx=20)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        video_frame = ttk.Frame(content_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        video_container = tk.Frame(video_frame, bg="#fff", bd=1, relief=tk.SOLID)
        video_container.pack(fill=tk.BOTH, expand=True)
        self.video_label = ttk.Label(video_container)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        control_frame = ttk.Frame(content_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        task_frame = ttk.Frame(control_frame)
        task_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(task_frame, text="Current Task", font=("Helvetica", 16, "bold")).pack(anchor=tk.W)
        self.task_icon = ttk.Label(task_frame, text="", font=("Helvetica", 40))
        self.task_icon.pack(pady=(10, 0), anchor=tk.W)
        self.message_label = ttk.Label(task_frame, text="", font=("Helvetica", 18), wraplength=280)
        self.message_label.pack(pady=(10, 0), anchor=tk.W)

        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=20)
        self.status_label = ttk.Label(status_frame, text="Get ready", font=("Helvetica", 16, "bold"))
        self.status_label.pack(anchor=tk.W)
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=280, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(10, 0))

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Button(button_frame, text="Skip Task", command=self.next_task).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Restart Session", command=self.restart_session).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Toggle Debug", command=self.toggle_debug).pack(side=tk.LEFT, padx=10)

        history_frame = ttk.Frame(control_frame)
        history_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Label(history_frame, text="Hand Path History", font=("Helvetica", 14, "bold")).pack(anchor=tk.W)
        self.history_canvas = tk.Canvas(history_frame, width=280, height=150, bg="white", bd=1, relief=tk.SOLID)
        self.history_canvas.pack(pady=(10, 0))

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.start()

    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def start_session(self):
        self.update_message(self.motion_tasks[self.current_task_index]["label"])
        self.update()

    def update(self):
        with self.lock:
            if self.frame is not None:
                frame = self.frame.copy()
            else:
                self.root.after(10, self.update)
                return

        frame = self.process_frame(frame)

        current_time = time.time()
        if current_time - self.last_display_time > 0.033:
            self.display_frame(frame)
            self.last_display_time = current_time

        self.root.after(10, self.update)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.update_timer()

        if self.phase == "prompt":
            self.start_countdown()
        elif self.phase == "countdown":
            frame = self.handle_countdown(frame, frame.shape[1])
        elif self.phase == "track":
            pose_result = self.pose.process(rgb)
            frame = self.track_shoulder_movement(frame, pose_result)
        elif self.phase == "feedback":
            self.handle_feedback()

        return frame

    def start_countdown(self):
        task = self.motion_tasks[self.current_task_index]
        self.update_message(task["label"], "Get ready", "#555")
        self.countdown_start = time.time()
        self.phase = "countdown"
        self.progress.config(value=0)
        self.initial_angle = None

    def handle_countdown(self, frame, width):
        elapsed = time.time() - self.countdown_start
        remaining = max(0, 3 - int(elapsed))
        self.progress.config(value=min(100, (elapsed / 3) * 100))

        if remaining > 0:
            self.update_message(
                self.motion_tasks[self.current_task_index]["label"],
                f"Starting in {remaining}...",
                "#f1bb4e"
            )
            overlay = frame.copy()
            cv2.circle(overlay, (width - 80, 80), 60, (0, 120, 255), -1)
            cv2.putText(overlay, str(remaining), (width - 93, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        else:
            self.track_start = time.time()
            self.trail.clear()
            self.shoulder_angles.clear()
            self.phase = "track"
            instruction = "Go! Perform the shoulder movement"
            self.update_message(
                self.motion_tasks[self.current_task_index]["label"],
                instruction,
                "#4CAF50"
            )
        return frame

    def track_shoulder_movement(self, frame, pose_result):
        elapsed = time.time() - self.track_start
        track_duration = 5.0
        self.progress.config(value=min(100, (elapsed / track_duration) * 100))

        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

            h, w = frame.shape[:2]
            rs = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            re = (int(right_elbow.x * w), int(right_elbow.y * h))
            rw = (int(right_wrist.x * w), int(right_wrist.y * h))
            rh = (int(right_hip.x * w), int(right_hip.y * h))

            # Draw only the right arm and shoulder-hip connection
            cv2.line(frame, rs, rh, (255, 0, 0), 2)  # Shoulder to hip (blue)
            cv2.line(frame, rs, re, (0, 0, 255), 2)  # Shoulder to elbow (red)
            cv2.line(frame, re, rw, (0, 255, 0), 2)  # Elbow to wrist (green)

            # Track wrist movement
            self.trail.append(rw)
            for i in range(1, len(self.trail)):
                cv2.line(frame, self.trail[i - 1], self.trail[i], self.hand_path_color, 3)

            # Calculate abduction/adduction angle
            shoulder_hip = np.array([rh[0] - rs[0], rh[1] - rs[1]])
            shoulder_elbow = np.array([re[0] - rs[0], re[1] - rs[1]])

            angle = self.calculate_angle(shoulder_hip, shoulder_elbow)
            self.shoulder_angles.append(angle)

            # Set initial angle if not set
            if self.initial_angle is None:
                self.initial_angle = angle

            # Display angle information
            cv2.putText(frame, f"Angle: {angle:.1f}°", (rs[0] + 20, rs[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display movement direction
            if len(self.shoulder_angles) > 1:
                angle_change = angle - self.shoulder_angles[-2]
                current_task = self.motion_tasks[self.current_task_index]["type"]

                # Corrected movement direction display
                if current_task == "Abduction":
                    direction = "Abduction" if angle_change > 0 else "Adduction"
                else:
                    direction = "Adduction" if angle_change < 0 else "Abduction"

                cv2.putText(frame, f"Movement: {direction}", (rs[0] + 20, rs[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if elapsed > track_duration:
            self.evaluate_motion()

        return frame

    def calculate_angle(self, a, b):
        """Calculate angle between two vectors in degrees (0-180)"""
        unit_a = a / np.linalg.norm(a)
        unit_b = b / np.linalg.norm(b)
        dot_product = np.dot(unit_a, unit_b)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        return angle

    def evaluate_motion(self):
        if len(self.shoulder_angles) < 10:
            self.feedback_result = "❌ Not enough movement"
            status_color = "#F44336"
        else:
            current_task = self.motion_tasks[self.current_task_index]["type"]
            angle_change = self.shoulder_angles[-1] - self.initial_angle

            # Corrected evaluation logic
            if current_task == "Abduction":
                success = angle_change > self.min_movement_threshold  # Angle increases for abduction
            else:
                success = angle_change < -self.min_movement_threshold  # Angle decreases for adduction

            if success:
                self.feedback_result = "✅ Great job!"
                status_color = "#4CAF50"
                self.score += 1
            else:
                self.feedback_result = f"❌ Not enough {current_task} movement"
                status_color = "#F44336"

            self.total_attempts += 1

        self.update_message(
            self.motion_tasks[self.current_task_index]["label"],
            self.feedback_result,
            status_color
        )
        self.update_score()
        self.feedback_start = time.time()
        self.phase = "feedback"
        self.draw_trail_history()

    def next_task(self):
        self.current_task_index = (self.current_task_index + 1) % len(self.motion_tasks)
        self.trail.clear()
        self.shoulder_angles.clear()
        self.initial_angle = None
        self.phase = "prompt"
        self.update_message(self.motion_tasks[self.current_task_index]["label"])
        self.progress.config(value=0)
        self.history_canvas.delete("all")

    def restart_session(self):
        self.score = 0
        self.total_attempts = 0
        self.session_start = datetime.datetime.now()
        self.update_score()
        self.next_task()

    def draw_trail_history(self):
        if len(self.trail) < 2:
            return

        self.history_canvas.delete("all")
        points = np.array(self.trail)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        width_range, height_range = max(1, max_x - min_x), max(1, max_y - min_y)
        canvas_w, canvas_h = 270, 140
        scale = min(canvas_w / width_range, canvas_h / height_range) * 0.8
        offset_x = (canvas_w - width_range * scale) / 2
        offset_y = (canvas_h - height_range * scale) / 2

        motion_type = self.motion_tasks[self.current_task_index]["type"]

        start = (self.trail[0][0] - min_x) * scale + offset_x, (self.trail[0][1] - min_y) * scale + offset_y
        end = (self.trail[-1][0] - min_x) * scale + offset_x, (self.trail[-1][1] - min_y) * scale + offset_y
        self.history_canvas.create_line(*start, *end, fill="#AAAAAA", width=1, dash=(4, 4))

        for i in range(1, len(self.trail)):
            x1 = (self.trail[i - 1][0] - min_x) * scale + offset_x
            y1 = (self.trail[i - 1][1] - min_y) * scale + offset_y
            x2 = (self.trail[i][0] - min_x) * scale + offset_x
            y2 = (self.trail[i][1] - min_y) * scale + offset_y

            progress = i / len(self.trail)
            color = f'#{0:02x}{int(100 + 155 * progress):02x}{int(200 * (1 - progress)):02x}'

            self.history_canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

    def update_message(self, msg, status="", status_color="#333"):
        task = self.motion_tasks[self.current_task_index]
        self.message_label.config(text=msg)
        self.task_icon.config(text=task["icon"])
        if status:
            self.status_label.config(text=status, foreground=status_color)

    def update_score(self):
        self.score_label.config(text=f"Score: {self.score}/{self.total_attempts}")

    def update_timer(self):
        elapsed = datetime.datetime.now() - self.session_start
        minutes, seconds = elapsed.seconds // 60, elapsed.seconds % 60
        self.timer_label.config(text=f"Session time: {minutes:02d}:{seconds:02d}")

    def display_frame(self, frame):
        img = Image.fromarray(frame)
        if img.size[0] > 800:
            img = img.resize((800, int(800 * img.size[1] / img.size[0])))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def toggle_debug(self):
        self.debug_mode = not self.debug_mode

    def handle_feedback(self):
        if time.time() - self.feedback_start > 2:
            self.next_task()

    def on_closing(self):
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        self.cap.release()
        self.root.destroy()

    def on_closing(self):
        """Clean up when window is closed"""
        self.cap.release()
if __name__ == "__main__":
    root = tk.Tk()
    app = ShoulderMotionTrainer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()