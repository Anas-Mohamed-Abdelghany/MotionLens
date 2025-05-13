import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time


class CompactStyle:
    BG_COLOR = "#f0f2f5"
    PRIMARY = "#3b82f6"
    SUCCESS = "#10b981"
    ERROR = "#ef4444"
    TEXT_DARK = "#1f2937"
    TEXT_MUTED = "#6b7280"

    @staticmethod
    def configure_styles():
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=CompactStyle.BG_COLOR)
        style.configure('TButton', background=CompactStyle.PRIMARY, foreground="white", padding=(8, 5))
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'), background=CompactStyle.BG_COLOR,
                        foreground=CompactStyle.TEXT_DARK)
        style.configure('Task.TLabel', font=('Helvetica', 11, 'bold'), background=CompactStyle.BG_COLOR,
                        foreground=CompactStyle.PRIMARY)
        style.configure('Info.TLabel', font=('Helvetica', 10), background=CompactStyle.BG_COLOR,
                        foreground=CompactStyle.TEXT_MUTED)
        style.configure('Result.TLabel', font=('Helvetica', 10, 'bold'), background=CompactStyle.BG_COLOR)
        style.configure("Horizontal.TProgressbar", thickness=8, background=CompactStyle.PRIMARY)


class MomentArmTrainer:
    def __init__(self, root):
        self.root = root
        CompactStyle.configure_styles()
        self.setup_ui()
        self.setup_mediapipe()
        self.reset_state()
        self.setup_camera()
        self.update()

    def setup_ui(self):
        self.root.title("Moment Arm Trainer")
        self.root.geometry("800x600")  # Compact window size
        self.root.configure(bg=CompactStyle.BG_COLOR)

        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top header with title and controls
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 5))

        # Title
        ttk.Label(header_frame, text="Moment Arm Trainer",
                  style='Title.TLabel').pack(side=tk.LEFT)

        # Control buttons
        buttons_frame = ttk.Frame(header_frame)
        buttons_frame.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(buttons_frame, text="Start",
                                       command=self.start_exercise)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_button = ttk.Button(buttons_frame, text="Reset",
                                       command=self.reset_exercise)
        self.reset_button.pack(side=tk.LEFT)

        # Content area - video and controls
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left side - Video feed (larger portion)
        video_frame = ttk.Frame(content_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video container with border
        self.video_container = tk.Frame(video_frame, bd=1, relief=tk.GROOVE,
                                        background=CompactStyle.TEXT_DARK)
        self.video_container.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(self.video_container)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Right side - Compact controls
        controls_frame = ttk.Frame(content_frame, width=200)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        controls_frame.pack_propagate(False)  # Maintain fixed width

        # Current task
        task_frame = ttk.Frame(controls_frame)
        task_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(task_frame, text="Current Task:",
                  style='Info.TLabel').pack(anchor=tk.W)

        self.task_label = ttk.Label(task_frame, text="Get Ready",
                                    style='Task.TLabel')
        self.task_label.pack(pady=(2, 0), anchor=tk.W)

        # Progress bar
        progress_frame = ttk.Frame(controls_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 5))

        progress_label_frame = ttk.Frame(progress_frame)
        progress_label_frame.pack(fill=tk.X)

        ttk.Label(progress_label_frame, text="Progress:",
                  style='Info.TLabel').pack(side=tk.LEFT)

        self.timer_label = ttk.Label(progress_label_frame, text="0:00",
                                     style='Info.TLabel')
        self.timer_label.pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL,
                                            length=180, mode='determinate',
                                            style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # Results
        results_frame = ttk.Frame(controls_frame)
        results_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(results_frame, text="Measurements:",
                  style='Info.TLabel').pack(anchor=tk.W)

        self.small_arm_label = ttk.Label(results_frame, text="Small arm: --",
                                         style='Result.TLabel')
        self.small_arm_label.pack(anchor=tk.W, pady=(2, 0))

        self.large_arm_label = ttk.Label(results_frame, text="Large arm: --",
                                         style='Result.TLabel')
        self.large_arm_label.pack(anchor=tk.W, pady=(2, 0))

        self.delta_label = ttk.Label(results_frame, text="Difference: --",
                                     style='Result.TLabel',
                                     foreground=CompactStyle.TEXT_DARK)
        self.delta_label.pack(anchor=tk.W, pady=(2, 0))

        # Feedback
        self.feedback_frame = ttk.Frame(controls_frame)
        self.feedback_frame.pack(fill=tk.X, pady=(10, 0))

        self.feedback_label = ttk.Label(self.feedback_frame, text="",
                                        style='Result.TLabel',
                                        wraplength=180)
        self.feedback_label.pack(anchor=tk.W)

        # Status bar at bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_label = ttk.Label(status_frame, text="Ready",
                                      style='Info.TLabel')
        self.status_label.pack(side=tk.LEFT)

        pass  # Omitted here for brevity; it’s identical to your original code

    def setup_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    def reset_state(self):
        self.phase = "idle"
        self.countdown_start = None
        self.small_arm_distance = 0
        self.large_arm_distance = 0
        self.phase_start_time = None
        self.initial_wrist_pos = None
        self.motion_started = False
        self.motion_threshold = 30
        self.wait_timer_start = None
        self.task_label.config(text="Get Ready")
        self.feedback_label.config(text="")
        self.small_arm_label.config(text="Small arm: --")
        self.large_arm_label.config(text="Large arm: --")
        self.delta_label.config(text="Difference: --")
        self.progress_bar["value"] = 0
        self.timer_label.config(text="0:00")
        self.status_label.config(text="Ready")

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def start_exercise(self):
        if self.phase == "idle":
            self.phase = "countdown"
            self.countdown_start = time.time()
            self.start_button.config(text="Cancel")
            self.status_label.config(text="Exercise in progress...")

    def reset_exercise(self):
        self.reset_state()
        self.start_button.config(text="Start")

    def format_time(self, seconds):
        return time.strftime("%M:%S", time.gmtime(seconds))

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        now = time.time()

        if self.phase == "idle":
            pass

        elif self.phase == "countdown":
            elapsed = int(now - self.countdown_start)
            remaining = 3 - elapsed
            if remaining > 0:
                cv2.putText(frame, str(remaining), (w // 2 - 20, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 120, 255), 3)
                self.task_label.config(text=f"Starting in {remaining}...")
                self.progress_bar["value"] = (3 - remaining) / 3 * 100
            else:
                self.phase = "small_arm"
                self.phase_start_time = now
                self.motion_started = False
                self.initial_wrist_pos = None

        elif self.phase == "wait":
            elapsed = now - self.wait_timer_start
            remaining = 3 - elapsed
            if remaining > 0:
                self.task_label.config(text=f"Next in {int(remaining)}...")
                self.progress_bar["value"] = (3 - remaining) / 3 * 100
                cv2.putText(frame, f"Next in {int(remaining)}", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (255, 255, 0), 3)
            else:
                self.phase = "large_arm"
                self.phase_start_time = now
                self.motion_started = False
                self.initial_wrist_pos = None

        elif results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            wrist = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            x1, y1 = int(shoulder.x * w), int(shoulder.y * h)
            x2, y2 = int(wrist.x * w), int(wrist.y * h)
            current_wrist = np.array([x2, y2])
            shoulder_pos = np.array([x1, y1])
            distance = np.linalg.norm(current_wrist - shoulder_pos)
            cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if self.initial_wrist_pos is None:
                self.initial_wrist_pos = current_wrist
            else:
                if not self.motion_started:
                    if np.linalg.norm(current_wrist - self.initial_wrist_pos) > self.motion_threshold:
                        self.motion_started = True
                        self.phase_start_time = now

            if self.phase == "small_arm":
                self.task_label.config(text="Raise arm with SMALL moment")
                if self.motion_started:
                    elapsed = now - self.phase_start_time
                    duration = 3
                    progress = min(100, (elapsed / duration) * 100)
                    self.progress_bar["value"] = progress
                    self.timer_label.config(text=self.format_time(elapsed))
                    bar_width = int((w - 20) * (progress / 100))
                    cv2.rectangle(frame, (10, h - 20), (10 + bar_width, h - 10), (0, 255, 120), -1)
                    if elapsed >= duration:
                        self.small_arm_distance = distance
                        self.small_arm_label.config(text=f"Small arm: {int(distance)} px")
                        self.phase = "wait"
                        self.wait_timer_start = now
                        self.progress_bar["value"] = 0

            elif self.phase == "large_arm":
                self.task_label.config(text="Raise arm with LARGE moment")
                if self.motion_started:
                    elapsed = now - self.phase_start_time
                    duration = 3
                    progress = min(100, (elapsed / duration) * 100)
                    self.progress_bar["value"] = progress
                    self.timer_label.config(text=self.format_time(elapsed))
                    bar_width = int((w - 20) * (progress / 100))
                    cv2.rectangle(frame, (10, h - 20), (10 + bar_width, h - 10), (0, 255, 120), -1)
                    if elapsed >= duration:
                        self.large_arm_distance = distance
                        self.large_arm_label.config(text=f"Large arm: {int(distance)} px")
                        self.phase = "feedback"
                        delta = int(self.large_arm_distance - self.small_arm_distance)
                        self.delta_label.config(text=f"Difference: {delta} px")
                        if delta > 20:
                            self.feedback_label.config(text=f"\u2713 Success! The difference is {delta} pixels.",
                                                       foreground=CompactStyle.SUCCESS)
                        else:
                            self.feedback_label.config(
                                text=f"\u2717 Try again with a bigger difference ({delta} pixels).",
                                foreground=CompactStyle.ERROR)
                        self.start_button.config(text="Start")
                        self.status_label.config(text="Exercise complete")

            elif self.phase == "feedback":
                self.task_label.config(text="Exercise Complete")
                self.progress_bar["value"] = 100

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        self.root.after(30, self.update)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MomentArmTrainer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
