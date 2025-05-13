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


class HandMotionTrainer:
    def __init__(self, root):
        # === Window Setup ===
        self.root = root
        self.root.title("Hand Motion Trainer")
        self.root.configure(bg="#f0f2f5")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # === MediaPipe Setup ===
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # === App State ===
        self.motion_tasks = [
            {"label": "Make angular movements", "type": "Angular", "icon": "⤴", "color": "#4e9af1"},
            {"label": "Move in a straight line", "type": "Linear", "icon": "➡", "color": "#f14e4e"}
        ]

        self.current_question_index = 0
        self.trail = deque(maxlen=120)
        self.phase = "prompt"  # prompt → countdown → track → feedback
        self.countdown_start = None
        self.track_start = None
        self.feedback_start = None
        self.feedback_result = ""
        self.score = 0
        self.total_attempts = 0
        self.session_start = datetime.datetime.now()

        # Debug mode toggle
        self.debug_mode = True  # Set to True to show detailed metrics

        # === Create UI Elements ===
        self.create_ui()

        # === OpenCV Capture ===
        self.cap = cv2.VideoCapture(0)

        # Start app
        self.update_message(self.motion_tasks[self.current_question_index]["label"])
        self.update()

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Configure styles
        style = ttk.Style()
        style.configure("TFrame", background="#f0f2f5")
        style.configure("Header.TLabel", font=("Helvetica", 24, "bold"), background="#f0f2f5", foreground="#333")
        style.configure("Score.TLabel", font=("Helvetica", 16), background="#f0f2f5", foreground="#555")
        style.configure("TButton", font=("Helvetica", 12))

        # Header
        header_frame = ttk.Frame(main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(header_frame, text="Hand Motion Trainer", style="Header.TLabel").pack(side=tk.LEFT)

        self.score_label = ttk.Label(header_frame, text="Score: 0/0", style="Score.TLabel")
        self.score_label.pack(side=tk.RIGHT)

        self.timer_label = ttk.Label(header_frame, text="Session time: 00:00", style="Score.TLabel")
        self.timer_label.pack(side=tk.RIGHT, padx=20)

        # Content
        content_frame = ttk.Frame(main_frame, style="TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Video display
        video_frame = ttk.Frame(content_frame, style="TFrame")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video container with border and shadow effect
        video_container = tk.Frame(video_frame, bg="#fff", bd=1, relief=tk.SOLID)
        video_container.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_container)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Right panel - Instructions and controls
        control_frame = ttk.Frame(content_frame, width=300, style="TFrame")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)

        # Task info
        task_frame = ttk.Frame(control_frame, style="TFrame")
        task_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(task_frame, text="Current Task", font=("Helvetica", 16, "bold"), background="#f0f2f5").pack(
            anchor=tk.W)

        self.task_icon = ttk.Label(task_frame, text="🔄", font=("Helvetica", 40), background="#f0f2f5")
        self.task_icon.pack(pady=(10, 0), anchor=tk.W)

        self.message_label = ttk.Label(task_frame, text="", font=("Helvetica", 18), wraplength=280,
                                       background="#f0f2f5")
        self.message_label.pack(pady=(10, 0), anchor=tk.W)

        # Status frame for progress, countdown, etc.
        status_frame = ttk.Frame(control_frame, style="TFrame")
        status_frame.pack(fill=tk.X, pady=20)

        self.status_label = ttk.Label(status_frame, text="Get ready", font=("Helvetica", 16, "bold"),
                                      background="#f0f2f5")
        self.status_label.pack(anchor=tk.W)

        # Progress bar for timing
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=280, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(10, 0))

        # Button frame
        button_frame = ttk.Frame(control_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=(20, 0))

        self.skip_button = ttk.Button(button_frame, text="Skip Task", command=self.next_question)
        self.skip_button.pack(side=tk.LEFT)

        self.restart_button = ttk.Button(button_frame, text="Restart Session", command=self.restart_session)
        self.restart_button.pack(side=tk.RIGHT)

        # Debug toggle button
        self.debug_button = ttk.Button(button_frame, text="Toggle Debug",
                                       command=lambda: setattr(self, 'debug_mode', not self.debug_mode))
        self.debug_button.pack(side=tk.LEFT, padx=10)

        # Motion history frame
        history_frame = ttk.Frame(control_frame, style="TFrame")
        history_frame.pack(fill=tk.X, pady=(20, 0))

        ttk.Label(history_frame, text="Motion History", font=("Helvetica", 14, "bold"), background="#f0f2f5").pack(
            anchor=tk.W)

        self.history_canvas = tk.Canvas(history_frame, width=280, height=150, bg="white", bd=1, relief=tk.SOLID)
        self.history_canvas.pack(pady=(10, 0))

    def update(self):
        """Main update loop with motion visualization and debug information"""
        # Update session timer
        self.update_timer()

        # Get video frame
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        result = self.hands.process(rgb)

        # Draw background for status info
        overlay = frame.copy()

        # State machine for training flow
        if self.phase == "prompt":
            task = self.motion_tasks[self.current_question_index]
            self.update_message(task["label"], "Get ready", "#555")
            self.countdown_start = time.time()
            self.phase = "countdown"
            self.progress.config(value=0)

        elif self.phase == "countdown":
            elapsed = time.time() - self.countdown_start
            remaining = 3 - int(elapsed)
            progress = min(100, (elapsed / 3) * 100)
            self.progress.config(value=progress)

            if remaining > 0:
                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    f"Starting in {remaining}...",
                    "#f1bb4e"
                )

                # Add countdown circle
                cv2.circle(
                    overlay,
                    (w - 80, 80),
                    60,
                    (0, 120, 255),
                    -1
                )
                cv2.putText(
                    overlay,
                    str(remaining),
                    (w - 93, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    3
                )
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            else:
                self.track_start = time.time()
                self.trail.clear()
                self.phase = "track"
                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    "Go! Move your hand now",
                    "#4CAF50"
                )

        elif self.phase == "track":
            elapsed = time.time() - self.track_start
            track_duration = 5.0
            progress = min(100, (elapsed / track_duration) * 100)
            self.progress.config(value=progress)

            # Process hand landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    cx_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    cy_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

                    # Find palm center
                    cx = sum(cx_list) // len(cx_list)
                    cy = sum(cy_list) // len(cy_list)

                    # Add to trail
                    self.trail.append((cx, cy))

                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

                    # Draw palm center
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                    cv2.circle(frame, (cx, cy), 25, (0, 0, 255), 2)

                    # Draw path
                    for i in range(1, len(self.trail)):
                        # Color fades from green to blue
                        progress = i / len(self.trail)
                        g = int(255 * (1 - progress))
                        b = int(255 * progress)
                        cv2.line(frame, self.trail[i - 1], self.trail[i], (0, g, b), 3)

                    # Add motion debug info if enabled
                    if self.debug_mode and len(self.trail) > 10:
                        frame = self.draw_debug_info(frame, self.trail)

            # Time progress indicator
            remaining = track_duration - elapsed
            if remaining > 0:
                cv2.rectangle(overlay, (w - 150, 30), (w - 30, 60), (0, 0, 0), -1)
                cv2.rectangle(overlay, (w - 150, 30), (int(w - 150 + 120 * (1 - remaining / track_duration)), 60),
                              (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Check if tracking phase is complete
            if elapsed > track_duration:
                motion_type, confidence = self.detect_motion_type(self.trail)
                expected = self.motion_tasks[self.current_question_index]["type"]
                self.total_attempts += 1

                if motion_type == expected:
                    self.feedback_result = "✅ Great job!"
                    status_color = "#4CAF50"
                    self.score += 1
                else:
                    self.feedback_result = f"❌ Detected: {motion_type}"
                    status_color = "#F44336"

                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    self.feedback_result,
                    status_color
                )
                self.update_score()
                self.feedback_start = time.time()
                self.phase = "feedback"

                # Update history canvas
                self.draw_trail_on_history()

        elif self.phase == "feedback":
            elapsed = time.time() - self.feedback_start

            # Show detailed motion analysis in feedback phase
            if len(self.trail) > 10:
                if self.debug_mode:
                    frame = self.draw_debug_info(frame, self.trail)

            if elapsed > 3:  # Extended feedback time when in debug mode
                self.next_question()

        # Show video in GUI
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Continue the loop
        self.root.after(10, self.update)

    def detect_motion_type(self, points):
        """
        Improved motion detection with more sensitivity to angular motion.
        """
        if len(points) < 12:
            return "Waiting...", 0

        pts = np.array(points, dtype=np.float32)

        # Filter out tiny hand tremors
        filtered_points = [pts[0]]
        min_distance = 5
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - filtered_points[-1]) >= min_distance:
                filtered_points.append(pts[i])

        if len(filtered_points) < 8:
            return "Waiting for more movement...", 0

        pts = np.array(filtered_points)

        # Path properties
        segments = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        path_length = np.sum(segment_lengths)
        direct_distance = np.linalg.norm(pts[-1] - pts[0])
        linearity_ratio = direct_distance / max(path_length, 1e-3)

        # Angle detection
        angles = []
        significant_turns = 0
        if path_length > 50:
            for i in range(1, len(segments)):
                v1, v2 = segments[i - 1], segments[i]
                len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if len1 < 8 or len2 < 8:
                    continue
                dot = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot))
                angles.append(angle_deg)
                if angle_deg > 15:  # More sensitive to curves
                    significant_turns += 1

        # Deviation from best-fit line
        if len(pts) >= 3 and direct_distance > 20:
            if abs(pts[-1][0] - pts[0][0]) > abs(pts[-1][1] - pts[0][1]):
                x, y = pts[:, 0], pts[:, 1]
            else:
                y, x = pts[:, 0], pts[:, 1]
            n = len(x)
            if n > 2 and np.std(x) > 0:
                coeffs = np.polyfit(x, y, 1)
                line_y = np.polyval(coeffs, x)
                mse = np.sum((y - line_y) ** 2) / n
                norm_deviation = np.sqrt(mse) / (path_length + 1e-3)
            else:
                norm_deviation = 0.5
        else:
            norm_deviation = 0.5

        # Early exit: movement too small
        if path_length < 40:
            return "Movement too small", 0

        # Final classification
        if (linearity_ratio > 0.85 and norm_deviation < 0.08 and significant_turns <= 1 and len(angles) > 0):
            return "Linear", linearity_ratio
        elif (linearity_ratio > 0.75 and norm_deviation < 0.1 and significant_turns <= 1):
            return "Linear", linearity_ratio * 0.8
        elif significant_turns >= 2 or (significant_turns == 1 and linearity_ratio < 0.7):
            return "Angular", min(1.0, significant_turns / 4)
        else:
            if linearity_ratio > 0.7:
                return "Linear", linearity_ratio * 0.7
            else:
                return "Angular", (1 - linearity_ratio) * 0.7

    def update_message(self, msg, status="", status_color="#333"):
        task = self.motion_tasks[self.current_question_index]
        self.message_label.config(text=msg)
        self.task_icon.config(text=task["icon"])

        if status:
            self.status_label.config(text=status, foreground=status_color)

    def update_score(self):
        self.score_label.config(text=f"Score: {self.score}/{self.total_attempts}")

    def update_timer(self):
        elapsed = datetime.datetime.now() - self.session_start
        minutes = elapsed.seconds // 60
        seconds = elapsed.seconds % 60
        self.timer_label.config(text=f"Session time: {minutes:02d}:{seconds:02d}")

    def restart_session(self):
        self.score = 0
        self.total_attempts = 0
        self.session_start = datetime.datetime.now()
        self.update_score()
        self.next_question()

    def next_question(self):
        self.current_question_index = (self.current_question_index + 1) % len(self.motion_tasks)
        self.trail.clear()
        self.phase = "prompt"
        self.countdown_start = None
        self.track_start = None
        self.feedback_start = None

        task = self.motion_tasks[self.current_question_index]
        self.update_message(task["label"])
        self.progress.config(value=0)

        # Clear history canvas
        self.history_canvas.delete("all")

    def draw_trail_on_history(self):
        """Draw the movement trail with analysis on the history canvas"""
        if len(self.trail) < 2:
            return

        # Clear canvas
        self.history_canvas.delete("all")

        # Scale points to fit canvas
        points = np.array(self.trail)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        # Prevent division by zero
        width_range = max(1, max_x - min_x)
        height_range = max(1, max_y - min_y)

        # Scale to fit in canvas with padding
        canvas_width = 270
        canvas_height = 140
        scale_x = canvas_width / width_range
        scale_y = canvas_height / height_range
        scale = min(scale_x, scale_y) * 0.8

        # Center the drawing
        offset_x = (canvas_width - width_range * scale) / 2
        offset_y = (canvas_height - height_range * scale) / 2

        # Get motion type
        motion_type, confidence = self.detect_motion_type(self.trail)

        # Draw the direct line from start to end (for linearity reference)
        start_x = (self.trail[0][0] - min_x) * scale + offset_x
        start_y = (self.trail[0][1] - min_y) * scale + offset_y
        end_x = (self.trail[-1][0] - min_x) * scale + offset_x
        end_y = (self.trail[-1][1] - min_y) * scale + offset_y

        # Draw direct line in light gray
        self.history_canvas.create_line(
            start_x, start_y, end_x, end_y,
            fill="#AAAAAA", width=1, dash=(4, 4)
        )

        # Draw start and end points
        self.history_canvas.create_oval(
            start_x - 5, start_y - 5, start_x + 5, start_y + 5,
            fill="#00AA00", outline=""
        )
        self.history_canvas.create_oval(
            end_x - 5, end_y - 5, end_x + 5, end_y + 5,
            fill="#AA0000", outline=""
        )

        # Draw the path with changing colors
        for i in range(1, len(self.trail)):
            x1 = (self.trail[i - 1][0] - min_x) * scale + offset_x
            y1 = (self.trail[i - 1][1] - min_y) * scale + offset_y
            x2 = (self.trail[i][0] - min_x) * scale + offset_x
            y2 = (self.trail[i][1] - min_y) * scale + offset_y

            # Gradient color - varies based on motion type
            progress = i / len(self.trail)

            if motion_type == "Linear":
                # Blue-green gradient for linear
                r = int(0)
                g = int(100 + 155 * progress)
                b = int(200 * (1 - progress))
            else:
                # Red-orange gradient for angular
                r = int(200 + 55 * progress)
                g = int(100 * (1 - progress))
                b = int(50 * (1 - progress))

            color = f'#{r:02x}{g:02x}{b:02x}'

            # Thicker line for more confident sections
            width = 2

            self.history_canvas.create_line(x1, y1, x2, y2, fill=color, width=width, smooth=True)

            # Mark direction changes
            if i > 1:
                v1 = np.array([self.trail[i - 1][0] - self.trail[i - 2][0],
                               self.trail[i - 1][1] - self.trail[i - 2][1]])
                v2 = np.array([self.trail[i][0] - self.trail[i - 1][0],
                               self.trail[i][1] - self.trail[i - 1][1]])

                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                    angle_deg = np.degrees(angle)

                    if angle_deg > 45:  # Significant direction change
                        self.history_canvas.create_oval(
                            x2 - 4, y2 - 4, x2 + 4, y2 + 4,
                            fill="#FF6600", outline=""
                        )

        # Add text showing the detected motion type
        if motion_type in ["Linear", "Angular"]:
            color = "#006699" if motion_type == "Linear" else "#993300"
            self.history_canvas.create_text(
                canvas_width / 2, 15,
                text=f"Detected: {motion_type}",
                fill=color,
                font=("Helvetica", 12, "bold")
            )

            # Add an icon/symbol to represent the motion type
            if motion_type == "Linear":
                # Draw a straight arrow for linear
                self.history_canvas.create_line(
                    60, canvas_height - 15, 210, canvas_height - 15,
                    fill="#006699", width=3, arrow="last"
                )
            else:
                # Draw a curved arrow for angular
                self.history_canvas.create_arc(
                    70, canvas_height - 40, 190, canvas_height + 10,
                    style="arc", outline="#993300", width=3,
                    start=180, extent=180
                )
                # Add arrowhead
                self.history_canvas.create_polygon(
                    190, canvas_height - 15,
                    185, canvas_height - 20,
                    185, canvas_height - 10,
                    fill="#993300", outline=""
                )

    def update(self):
        # Update session timer
        self.update_timer()

        # Get video frame
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        result = self.hands.process(rgb)

        # Draw background for status info
        overlay = frame.copy()

        # State machine for training flow
        if self.phase == "prompt":
            task = self.motion_tasks[self.current_question_index]
            self.update_message(task["label"], "Get ready", "#555")
            self.countdown_start = time.time()
            self.phase = "countdown"
            self.progress.config(value=0)

        elif self.phase == "countdown":
            elapsed = time.time() - self.countdown_start
            remaining = 3 - int(elapsed)
            progress = min(100, (elapsed / 3) * 100)
            self.progress.config(value=progress)

            if remaining > 0:
                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    f"Starting in {remaining}...",
                    "#f1bb4e"
                )

                # Add countdown circle
                cv2.circle(
                    overlay,
                    (w - 80, 80),
                    60,
                    (0, 120, 255),
                    -1
                )
                cv2.putText(
                    overlay,
                    str(remaining),
                    (w - 93, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    3
                )
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            else:
                self.track_start = time.time()
                self.trail.clear()
                self.phase = "track"
                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    "Go! Move your hand now",
                    "#4CAF50"
                )

        elif self.phase == "track":
            elapsed = time.time() - self.track_start
            track_duration = 5.0  # Increased tracking duration for better detection
            progress = min(100, (elapsed / track_duration) * 100)
            self.progress.config(value=progress)

            # Process hand landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    cx_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    cy_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

                    # Find palm center
                    cx = sum(cx_list) // len(cx_list)
                    cy = sum(cy_list) // len(cy_list)

                    # Add to trail
                    self.trail.append((cx, cy))

                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

                    # Draw palm center
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                    cv2.circle(frame, (cx, cy), 25, (0, 0, 255), 2)

                    # Draw path
                    for i in range(1, len(self.trail)):
                        # Color fades from green to blue
                        progress = i / len(self.trail)
                        g = int(255 * (1 - progress))
                        b = int(255 * progress)
                        cv2.line(frame, self.trail[i - 1], self.trail[i], (0, g, b), 3)

            # Time progress indicator
            remaining = track_duration - elapsed
            if remaining > 0:
                cv2.rectangle(overlay, (w - 150, 30), (w - 30, 60), (0, 0, 0), -1)
                cv2.rectangle(overlay, (w - 150, 30), (int(w - 150 + 120 * (1 - remaining / track_duration)), 60),
                              (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Check if tracking phase is complete
            if elapsed > track_duration:
                motion_type, confidence = self.detect_motion_type(self.trail)
                expected = self.motion_tasks[self.current_question_index]["type"]
                self.total_attempts += 1

                if motion_type == expected:
                    self.feedback_result = "✅ Great job!"
                    status_color = "#4CAF50"
                    self.score += 1
                else:
                    self.feedback_result = f"❌ Detected: {motion_type}"
                    status_color = "#F44336"

                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    self.feedback_result,
                    status_color
                )
                self.update_score()
                self.feedback_start = time.time()
                self.phase = "feedback"

                # Update history canvas
                self.draw_trail_on_history()

        elif self.phase == "feedback":
            elapsed = time.time() - self.feedback_start
            if elapsed > 2:
                self.next_question()

        # Show video in GUI
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Continue the loop
        self.root.after(10, self.update)

    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def draw_motion_analysis(self, frame, points):
        """Draw visual indicators of motion type detection on frame"""
        if len(points) < 15:
            return frame

        h, w, _ = frame.shape
        pts = np.array(points, dtype=np.int32)

        # Create analysis overlay
        overlay = frame.copy()
        analysis_panel = np.ones((h, 200, 3), dtype=np.uint8) * 255

        # Draw the points trajectory
        if len(pts) >= 2:
            # Draw path with gradient color
            for i in range(1, len(pts)):
                progress = i / len(pts)
                b = int(255 * (1 - progress))
                g = int(200 * progress)
                r = int(100 + 155 * progress)
                cv2.line(overlay, tuple(pts[i - 1]), tuple(pts[i]), (b, g, r), 3)

        # Calculate the analysis metrics
        motion_type, confidence = self.detect_motion_type(pts)

        # Draw start and end points
        if len(pts) >= 2:
            cv2.circle(overlay, tuple(pts[0]), 8, (0, 255, 0), -1)  # Start point
            cv2.circle(overlay, tuple(pts[-1]), 8, (0, 0, 255), -1)  # End point

            # Draw the straight line between start and end
            cv2.line(overlay, tuple(pts[0]), tuple(pts[-1]), (255, 0, 0), 2)

        # Calculate direction changes
        if len(pts) >= 3:
            directions = np.diff(pts, axis=0)

            # Draw direction change indicators
            for i in range(1, len(directions)):
                v1 = directions[i - 1]
                v2 = directions[i]

                # Skip if vectors are too small
                if np.linalg.norm(v1) < 5 or np.linalg.norm(v2) < 5:
                    continue

                # Calculate angle
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))

                # Mark significant direction changes
                if angle_deg > 30:
                    point_idx = i + 1  # +1 because directions are calculated from differences
                    if 0 <= point_idx < len(pts):
                        # Draw angle marker
                        cv2.circle(overlay, tuple(pts[point_idx]), 10, (0, 165, 255), 2)

                        # For sharp turns, add more prominent marker
                        if angle_deg > 60:
                            cv2.circle(overlay, tuple(pts[point_idx]), 15, (0, 0, 255), 3)

        # Add the analysis panel
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text indicator for motion type
        y_pos = h - 70
        if motion_type == "Linear":
            cv2.putText(frame, "LINEAR MOTION", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)
        elif motion_type == "Angular":
            cv2.putText(frame, "ANGULAR MOTION", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        elif "Waiting" not in motion_type:
            cv2.putText(frame, f"MOTION: {motion_type}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        return frame

    def update(self):
        """Main update loop with motion visualization improvements"""
        # Update session timer
        self.update_timer()

        # Get video frame
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        result = self.hands.process(rgb)

        # Draw background for status info
        overlay = frame.copy()

        # State machine for training flow
        if self.phase == "prompt":
            task = self.motion_tasks[self.current_question_index]
            self.update_message(task["label"], "Get ready", "#555")
            self.countdown_start = time.time()
            self.phase = "countdown"
            self.progress.config(value=0)

        elif self.phase == "countdown":
            elapsed = time.time() - self.countdown_start
            remaining = 3 - int(elapsed)
            progress = min(100, (elapsed / 3) * 100)
            self.progress.config(value=progress)

            if remaining > 0:
                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    f"Starting in {remaining}...",
                    "#f1bb4e"
                )

                # Add countdown circle
                cv2.circle(
                    overlay,
                    (w - 80, 80),
                    60,
                    (0, 120, 255),
                    -1
                )
                cv2.putText(
                    overlay,
                    str(remaining),
                    (w - 93, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    3
                )
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            else:
                self.track_start = time.time()
                self.trail.clear()
                self.phase = "track"
                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    "Go! Move your hand now",
                    "#4CAF50"
                )

        elif self.phase == "track":
            elapsed = time.time() - self.track_start
            track_duration = 5.0  # Increased tracking duration for better detection
            progress = min(100, (elapsed / track_duration) * 100)
            self.progress.config(value=progress)

            # Process hand landmarks
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    cx_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    cy_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

                    # Find palm center
                    cx = sum(cx_list) // len(cx_list)
                    cy = sum(cy_list) // len(cy_list)

                    # Add to trail
                    self.trail.append((cx, cy))

                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

                    # Draw palm center
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                    cv2.circle(frame, (cx, cy), 25, (0, 0, 255), 2)

                    # Add real-time motion analysis visualization
                    frame = self.draw_motion_analysis(frame, self.trail)

            # Time progress indicator
            remaining = track_duration - elapsed
            if remaining > 0:
                cv2.rectangle(overlay, (w - 150, 30), (w - 30, 60), (0, 0, 0), -1)
                cv2.rectangle(overlay, (w - 150, 30), (int(w - 150 + 120 * (1 - remaining / track_duration)), 60),
                              (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Check if tracking phase is complete
            if elapsed > track_duration:
                motion_type, confidence = self.detect_motion_type(self.trail)
                expected = self.motion_tasks[self.current_question_index]["type"]
                self.total_attempts += 1

                if motion_type == expected:
                    self.feedback_result = "✅ Great job!"
                    status_color = "#4CAF50"
                    self.score += 1
                else:
                    self.feedback_result = f"❌ Detected: {motion_type}"
                    status_color = "#F44336"

                self.update_message(
                    self.motion_tasks[self.current_question_index]["label"],
                    self.feedback_result,
                    status_color
                )
                self.update_score()
                self.feedback_start = time.time()
                self.phase = "feedback"

                # Update history canvas
                self.draw_trail_on_history()

        elif self.phase == "feedback":
            elapsed = time.time() - self.feedback_start
            # Show detailed motion analysis in feedback phase
            if len(self.trail) > 10:
                frame = self.draw_motion_analysis(frame, self.trail)

            if elapsed > 2:
                self.next_question()

        # Show video in GUI
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Continue the loop
        self.root.after(10, self.update)

    def draw_debug_info(self, frame, points):
        """
        Draw detailed metrics for motion detection to help debug classification issues.
        """
        if len(points) < 12:
            return frame

        h, w, _ = frame.shape
        pts = np.array(points, dtype=np.float32)

        # Create semi-transparent overlay for debug panel
        debug_overlay = frame.copy()
        cv2.rectangle(debug_overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(debug_overlay, 0.7, frame, 0.3, 0, frame)

        # Calculate metrics similar to detect_motion_type
        # Filter points like in the detection function
        filtered_points = [pts[0]]
        min_distance = 5
        for i in range(1, len(pts)):
            dist = np.linalg.norm(pts[i] - filtered_points[-1])
            if dist >= min_distance:
                filtered_points.append(pts[i])

        if len(filtered_points) < 8:
            cv2.putText(frame, "Not enough movement data", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            return frame

        pts = np.array(filtered_points)

        # Calculate path metrics
        segments = np.diff(pts, axis=0)
        segment_lengths = np.sqrt(np.sum(segments ** 2, axis=1))
        path_length = np.sum(segment_lengths)
        direct_distance = np.linalg.norm(pts[-1] - pts[0])
        linearity_ratio = direct_distance / max(path_length, 0.001)

        # Count direction changes
        significant_turns = 0
        if path_length > 50:
            for i in range(1, len(segments)):
                v1 = segments[i - 1]
                v2 = segments[i]

                len1 = np.linalg.norm(v1)
                len2 = np.linalg.norm(v2)

                if len1 < 8 or len2 < 8:
                    continue

                dot = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot))

                if angle_deg > 40:
                    significant_turns += 1

        # Calculate deviation
        if len(pts) >= 3 and direct_distance > 20:
            if abs(pts[-1][0] - pts[0][0]) > abs(pts[-1][1] - pts[0][1]):
                x, y = pts[:, 0], pts[:, 1]
            else:
                y, x = pts[:, 0], pts[:, 1]

            if len(x) > 2 and np.std(x) > 0:
                coeffs = np.polyfit(x, y, 1)
                line_y = np.polyval(coeffs, x)
                mse = np.sum((y - line_y) ** 2) / len(x)
                norm_deviation = np.sqrt(mse) / (path_length + 0.001)
            else:
                norm_deviation = 0.5
        else:
            norm_deviation = 0.5

        # Get current classification
        motion_type, confidence = self.detect_motion_type(points)

        # Draw debug text
        y_offset = 40
        line_height = 25

        cv2.putText(frame, f"MOTION DEBUG: {motion_type} (conf: {confidence:.2f})",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height

        # Metrics with color-coded values (green = favors linear, red = favors angular)
        metrics = [
            ("Path length", f"{path_length:.1f}px", (200, 200, 200)),
            ("Direct distance", f"{direct_distance:.1f}px", (200, 200, 200)),
            ("Linearity ratio", f"{linearity_ratio:.2f}",
             (0, 255, 0) if linearity_ratio > 0.8 else
             (255, 255, 0) if linearity_ratio > 0.7 else (255, 150, 0)),
            ("Significant turns", f"{significant_turns}",
             (0, 255, 0) if significant_turns == 0 else
             (255, 255, 0) if significant_turns == 1 else (0, 100, 255)),
            ("Norm deviation", f"{norm_deviation:.3f}",
             (0, 255, 0) if norm_deviation < 0.1 else
             (255, 255, 0) if norm_deviation < 0.15 else (0, 100, 255))
        ]

        for label, value, color in metrics:
            cv2.putText(frame, f"{label}: {value}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += line_height

        # Classification explanation
        if motion_type == "Linear":
            rules = []
            if linearity_ratio > 0.85 and norm_deviation < 0.1 and significant_turns <= 1:
                rules.append("Strong linear motion pattern")
            elif linearity_ratio > 0.75 and norm_deviation < 0.15 and significant_turns <= 2:
                rules.append("Somewhat linear pattern")
            elif linearity_ratio > 0.7:
                rules.append("Linearity ratio > 0.7")

            cv2.putText(frame, f"Why Linear: {', '.join(rules)}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        elif motion_type == "Angular":
            rules = []
            if significant_turns >= 3:
                rules.append(f"{significant_turns} significant turns")
            elif significant_turns >= 2 and linearity_ratio < 0.65:
                rules.append(f"{significant_turns} turns + low linearity")
            else:
                rules.append(f"Linearity ratio {linearity_ratio:.2f} < 0.7")

            cv2.putText(frame, f"Why Angular: {', '.join(rules)}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

        return frame
if __name__ == "__main__":
    root = tk.Tk()
    app = HandMotionTrainer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()