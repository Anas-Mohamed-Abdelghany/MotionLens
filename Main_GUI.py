import tkinter as tk
from tkinter import ttk, font, messagebox
from PIL import Image, ImageTk
import os
import sv_ttk # For modern theme
import webbrowser


class MotionAnalysisLauncher:
    def __init__(self, root):
        # Main window setup
        self.root = root
        self.root.title("MotionLens")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Configure styles before creating UI
        self.configure_styles()

        # Set dark/light theme
        sv_ttk.set_theme("light")

        # Load custom fonts
        self.load_fonts()

        # Load images and animations
        self.load_assets()

        # Create UI elements
        self.create_ui()

        # Start animations
        self.animate_hero()

    def configure_styles(self):
        """Configure all custom styles before UI creation"""
        style = ttk.Style()

        # Card style for containers
        style.configure("Card.TFrame",
                        background="white",
                        borderwidth=2,
                        relief="solid",
                        lightcolor="white",
                        darkcolor="white")

        style.configure("Card.Hover.TFrame",
                        background="#f8f9fc")

        # Label styles
        style.configure("Accent.TLabel",
                        foreground="#4e73df")

        style.configure("Hero.TLabel",
                        background="#4e73df",
                        foreground="white")

        # Button styles
        style.configure("Accent.TButton",
                        background="#4e73df",
                        foreground="white",
                        borderwidth=0)

        style.map("Accent.TButton",
                  background=[("active", "#3a56b4"), ("disabled", "#dddddd")],
                  foreground=[("disabled", "#aaaaaa")])

        style.configure("Accent.Outline.TButton",
                        background="white",
                        foreground="#4e73df",
                        borderwidth=1)

        style.map("Accent.Outline.TButton",
                  background=[("active", "#f8f9fc")],
                  foreground=[("active", "#3a56b4")])

        style.configure("Bold.TLabel",
                        font=("Helvetica", 10, "bold"))

    def load_fonts(self):
        # Create font objects (using system fonts)
        self.title_font = font.Font(family="Helvetica", size=28, weight="bold")
        self.header_font = font.Font(family="Helvetica", size=20, weight="bold")
        self.subheader_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.body_font = font.Font(family="Helvetica", size=12)
        self.button_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.small_font = font.Font(family="Helvetica", size=10)

    def load_assets(self):
        # Load placeholder images (in a real app, these would be actual biomechanics images)
        try:
            # Hero image (would be a biomechanics illustration)
            self.hero_img = Image.open("assets/hero_image.png").resize((300, 300)) if os.path.exists("assets/hero_image.png") else None

            # Module icons
            self.hand_icon = Image.open("assets/hand_icon.png").resize((80, 80)) if os.path.exists(
                "assets/hand_icon.png") else None
            self.posture_icon = Image.open("assets/posture_icon.png").resize((80, 80)) if os.path.exists(
                "assets/posture_icon.png") else None
            self.gait_icon = Image.open("assets/gait_icon.png").resize((80, 80)) if os.path.exists(
                "assets/gait_icon.png") else None
            self.quiz_icon = Image.open("assets/quiz_icon.png").resize((80, 80)) if os.path.exists(
                "assets/gait_icon.png") else None


            # Biomechanics figures
            self.joint_angles_img = Image.open("assets/joint_angles.png").resize((400, 200)) if os.path.exists(
                "assets/joint_angles.png") else None
            self.kinematics_img = Image.open("assets/kinematics.png").resize((300, 200)) if os.path.exists(
                "assets/kinematics.png") else None

            # Animation frames for the hero section
            self.animation_frames = []
            for i in range(1, 6):
                try:
                    frame = Image.open(f"assets/animation_frame_{i}.png")
                    self.animation_frames.append(frame)
                except:
                    continue

        except Exception as e:
            print(f"Error loading assets: {e}")
            # Create blank placeholders if images don't exist
            self.hero_img = None
            self.hand_icon = None
            self.posture_icon = None
            self.gait_icon = None
            self.joint_angles_img = None
            self.kinematics_img = None
            self.animation_frames = []

    def create_ui(self):
        # Main container with modern styling
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas for smooth scrolling
        self.canvas = tk.Canvas(main_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel for scrolling
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Hero section with animation
        self.hero_frame = ttk.Frame(self.scrollable_frame, style="Card.TFrame")
        self.hero_frame.pack(fill=tk.X, padx=20, pady=20)

        # Hero content will be populated by animate_hero()

        # Content section with two columns
        content_frame = ttk.Frame(self.scrollable_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left column (60% width)
        left_column = ttk.Frame(content_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Right column (40% width)
        right_column = ttk.Frame(content_frame, width=400)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)

        # Learning Modules section
        modules_header = ttk.Frame(left_column)
        modules_header.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            modules_header,
            text="Learning Modules",
            font=self.header_font,
            style="Accent.TLabel"
        ).pack(side=tk.LEFT, anchor=tk.W)

        # Modules grid (3 cards)
        modules_grid = ttk.Frame(left_column)
        modules_grid.pack(fill=tk.BOTH, expand=True)

        # Module 1: Hand Motion Analysis
        self.create_module_card(
            modules_grid,
            "Hand Motion Analysis",
            "Train to recognize and perform linear and angular hand movements using computer vision tracking.",
            self.hand_icon,
            "#4e73df",
            lambda: self.launch_program("hand_motion"),
            row=0, col=0
        )

        # Module 2: Joint Movement Analysis
        self.create_module_card(
            modules_grid,
            "Joint Movement Analysis",
            "Train to recognize and perform abduction and adduction joint movements using computer vision tracking.",
            self.posture_icon,
            "#1cc88a",
            lambda: self.launch_program("joint_movement"),
            row=0, col=1
        )

        # Module 3: Moment Arm Analysis
        # In create_ui(), change the Moment Arm Analysis module card creation to:
        self.create_module_card(
            modules_grid,
            "Moment Arm Analysis",
            "Interactive tool to analyze and calculate moment Arm in limb movements.",
            self.gait_icon,
            "#f6c23e",
            lambda: self.launch_program("moment_arm"),  # Changed from show_coming_soon
            row=1, col=0
        )

        # Change from "Coming Soon" to "Biomechanics Quiz"
        self.create_module_card(
            modules_grid,
            "Quiz platform",
            "Test your knowledge of motion analysis with this interactive quizzes.",
            self.quiz_icon,  # You can add a quiz icon later
            "#e74a3b",
            lambda: self.launch_quiz(),  # Changed to launch quiz
            row=1, col=1
        )

        # Biomechanics Figures section
        ttk.Label(
            left_column,
            text="Biomechanics Concepts",
            font=self.header_font,
            style="Accent.TLabel"
        ).pack(anchor=tk.W, pady=(20, 10))

        figures_frame = ttk.Frame(left_column)
        figures_frame.pack(fill=tk.X, pady=(0, 20))

        # Figure 1: Impulse Force
        self.create_figure_card(
            figures_frame,
            "Impulse-Force Concept",
            "Understanding Impulse force concept and it's applications.",
            self.joint_angles_img,
            "#36b9cc",
            0
        )

        # Figure 2: Kinematics
        self.create_figure_card(
            figures_frame,
            "Movement Kinematics",
            "Analysis of position, velocity, and acceleration of body segments.",
            self.kinematics_img,
            "#5a5c69",
            1
        )

        # Right column content
        # About section
        about_card = ttk.Frame(right_column, style="Card.TFrame")
        about_card.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(
            about_card,
            text="About the Platform",
            font=self.subheader_font,
            style="Accent.TLabel"
        ).pack(anchor=tk.W, padx=15, pady=(15, 5))

        about_text = (
            "This interactive learning platform uses computer vision and deep learning "
            "to analyze human movement patterns and provide real-time biomechanical feedback. "
            "The system helps users understand complex biomechanics principles through "
            "practical exercises and visual learning."
        )

        ttk.Label(
            about_card,
            text=about_text,
            font=self.body_font,
            wraplength=350,
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=15, pady=(0, 15))

        # How it works section
        how_it_works_card = ttk.Frame(right_column, style="Card.TFrame")
        how_it_works_card.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(
            how_it_works_card,
            text="How It Works",
            font=self.subheader_font,
            style="Accent.TLabel"
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        steps = [
            ("1. Select a Module", "Choose from our biomechanics learning modules"),
            ("2. Camera Setup", "Position yourself in the camera view as instructed"),
            ("3. Interactive Training", "Follow the guided exercises and demonstrations"),
            ("4. Get Feedback", "Receive real-time biomechanical analysis")
        ]

        for i, (title, desc) in enumerate(steps):
            step_frame = ttk.Frame(how_it_works_card)
            step_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

            # Step number
            ttk.Label(
                step_frame,
                text=str(i + 1),
                font=self.subheader_font,
                foreground="#4e73df",
                width=3
            ).pack(side=tk.LEFT, anchor=tk.N)

            # Step content
            step_content = ttk.Frame(step_frame)
            step_content.pack(side=tk.LEFT, fill=tk.X, expand=True)

            ttk.Label(
                step_content,
                text=title,
                font=self.body_font,
                style="Bold.TLabel"
            ).pack(anchor=tk.W)

            ttk.Label(
                step_content,
                text=desc,
                font=self.small_font,
                foreground="#5a5c69"
            ).pack(anchor=tk.W, pady=(2, 0))

        # Features highlights
        features_card = ttk.Frame(right_column, style="Card.TFrame")
        features_card.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(
            features_card,
            text="Key Features",
            font=self.subheader_font,
            style="Accent.TLabel"
        ).pack(anchor=tk.W, padx=15, pady=(15, 10))

        features = [
            ("Real-time Analysis", "Computer vision tracks your movements instantly"),
            ("Progress Tracking", "Save your progress with point system"),
            ("Interactive Quizzes", "Test your understanding with real motion videos"),
            ("Cartoon-based Learning", "Learn biomechanics concepts through animated educational films")
        ]

        for feat, desc in features:
            feat_frame = ttk.Frame(features_card)
            feat_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

            ttk.Label(
                feat_frame,
                text="•",
                font=self.body_font,
                foreground="#4e73df"
            ).pack(side=tk.LEFT)

            ttk.Label(
                feat_frame,
                text=f" {feat}: {desc}",
                font=self.small_font,
                wraplength=320,
                justify=tk.LEFT
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Footer
        footer = ttk.Frame(self.scrollable_frame)
        footer.pack(fill=tk.X, pady=(20, 0))

        ttk.Label(
            footer,
            text="© 2025 Biomechanics Interactive Learning Tool | Team 6 ",
            font=self.small_font,
            foreground="#5a5c69"
        ).pack(side=tk.LEFT, padx=20, pady=10)

        ttk.Label(
            footer,
            text="v1.0.0",
            font=self.small_font,
            foreground="#5a5c69"
        ).pack(side=tk.RIGHT, padx=20, pady=10)

    def animate_hero(self):
        """Animate the hero section with changing images/text"""
        hero_content = ttk.Frame(self.hero_frame)
        hero_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side - text
        text_frame = ttk.Frame(hero_content)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.hero_title = ttk.Label(
            text_frame,
            text="MotionLens",
            font=("Helvetica", 28, "bold"),
            style="Hero.TLabel"
        )
        self.hero_title.pack(anchor=tk.W, pady=(0, 15))

        self.hero_subtitle = ttk.Label(
            text_frame,
            text="See Motion...Master Biomechanics",
            font=("Helvetica", 16),
            style="Hero.TLabel"
        )
        self.hero_subtitle.pack(anchor=tk.W, pady=(0, 20))

        hero_desc = (
            "Our platform combines advanced computer vision with biomechanics principles "
            "to provide real-time feedback on your movements. Perfect for students, "
            "therapists, and athletes looking to improve their motion patterns."
        )

        ttk.Label(
            text_frame,
            text=hero_desc,
            font=self.body_font,
            wraplength=500,
            justify=tk.LEFT,
            style="Hero.TLabel"
        ).pack(anchor=tk.W, pady=(0, 30))

        # CTA button
        cta_button = ttk.Button(
            text_frame,
            text="Who Us ? ",
            style="Accent.TButton",
            command=lambda: self.show_team_info()
        )
        cta_button.pack(anchor=tk.W, ipadx=20, ipady=10)

        # Right side - image/animation
        img_frame = ttk.Frame(hero_content, width=400)
        img_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        if self.animation_frames:
            self.animation_label = ttk.Label(img_frame)
            self.animation_label.pack(fill=tk.BOTH, expand=True)

            # Start animation
            self.animate(0)
        elif self.hero_img:
            hero_img = ImageTk.PhotoImage(self.hero_img)
            img_label = ttk.Label(img_frame, image=hero_img)
            img_label.image = hero_img
            img_label.pack(fill=tk.BOTH, expand=True)
        else:
            # Placeholder if no images available
            ttk.Label(
                img_frame,
                text="Biomechanics Illustration",
                font=self.small_font,
                foreground="#5a5c69",
                background="#f8f9fc"
            ).pack(fill=tk.BOTH, expand=True)

    def animate(self, idx):
        """Cycle through animation frames"""
        if not self.animation_frames:
            return

        frame = self.animation_frames[idx]
        frame = frame.resize((400, 300))
        photo = ImageTk.PhotoImage(frame)

        self.animation_label.configure(image=photo)
        self.animation_label.image = photo

        # Schedule next frame
        self.root.after(300, lambda: self.animate((idx + 1) % len(self.animation_frames)))

    def create_module_card(self, parent, title, description, icon, color, command, row, col):
        """Create a modern card for each learning module"""
        card = ttk.Frame(parent, style="Card.TFrame")
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        parent.grid_rowconfigure(row, weight=1)

        # Card header
        header = ttk.Frame(card)
        header.pack(fill=tk.X, padx=15, pady=(15, 10))

        # Icon
        if icon:
            icon_img = ImageTk.PhotoImage(icon)
            icon_label = ttk.Label(header, image=icon_img)
            icon_label.image = icon_img
            icon_label.pack(side=tk.LEFT, padx=(0, 15))

        # Title
        title_label = ttk.Label(
            header,
            text=title,
            font=self.subheader_font,
            style="Accent.TLabel"
        )
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Description
        ttk.Label(
            card,
            text=description,
            font=self.body_font,
            wraplength=300,
            justify=tk.LEFT
        ).pack(fill=tk.X, padx=15, pady=(0, 20))

        # Button
        button = ttk.Button(
            card,
            text="Launch Module",
            style="Accent.TButton",
            command=command
        )
        button.pack(side=tk.RIGHT, padx=15, pady=(0, 15))

        # Hover effect
        def on_enter(e):
            card.configure(style="Card.Hover.TFrame")

        def on_leave(e):
            card.configure(style="Card.TFrame")

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

    def create_figure_card(self, parent, title, description, image, color, col):
        """Create a card for biomechanics figures"""
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10) if col == 0 else (10, 0))

        # Title
        ttk.Label(
            card,
            text=title,
            font=self.subheader_font,
            style="Accent.TLabel"
        ).pack(anchor=tk.W, padx=15, pady=15)

        # Image
        img_frame = ttk.Frame(card, height=150)
        img_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        if image:
            img = ImageTk.PhotoImage(image)
            img_label = ttk.Label(img_frame, image=img)
            img_label.image = img
            img_label.pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(
                img_frame,
                text="Figure Placeholder",
                background="#f8f9fc",
                foreground="#5a5c69"
            ).pack(fill=tk.BOTH, expand=True)

        # Description
        ttk.Label(
            card,
            text=description,
            font=self.small_font,
            wraplength=250,
            justify=tk.LEFT
        ).pack(fill=tk.X, padx=15, pady=(0, 15))

        # Learn more button
        ttk.Button(
            card,
            text="Learn More",
            style="Accent.Outline.TButton",
            command=lambda: self.show_figure_details(title)
        ).pack(side=tk.RIGHT, padx=15, pady=(0, 15))

    def launch_program(self, program_type):
        """Launch the selected motion analysis program"""
        if program_type == "hand_motion":
            try:
                from Computer_vision import HandMotionTrainer
                trainer_root = tk.Toplevel(self.root)
                app = HandMotionTrainer(trainer_root)
                self.root.withdraw()

                def on_trainer_close():
                    if hasattr(app, 'on_closing'):
                        app.on_closing()
                    trainer_root.destroy()
                    self.root.deiconify()

                trainer_root.protocol("WM_DELETE_WINDOW", on_trainer_close)

            except ImportError:
                messagebox.showerror(
                    "Module Error",
                    "Could not load the Hand Motion Trainer module. Please check your installation."
                )

        elif program_type == "moment_arm":
            try:
                from Computer_vision2 import MomentArmTrainer
                trainer_root = tk.Toplevel(self.root)
                app = MomentArmTrainer(trainer_root)
                self.root.withdraw()

                def on_trainer_close():
                    if hasattr(app, 'on_closing'):
                        app.on_closing()
                    trainer_root.destroy()
                    self.root.deiconify()

                trainer_root.protocol("WM_DELETE_WINDOW", on_trainer_close)

            except ImportError as e:
                messagebox.showerror(
                    "Module Error",
                    f"Could not load the Moment Arm Trainer module. Error: {str(e)}"
                )

        elif program_type == "joint_movement":
            try:
                from Computer_vision3 import ShoulderMotionTrainer
                trainer_root = tk.Toplevel(self.root)
                app = ShoulderMotionTrainer(trainer_root)
                self.root.withdraw()

                def on_trainer_close():
                    if hasattr(app, 'on_closing'):
                        app.on_closing()
                    trainer_root.destroy()
                    self.root.deiconify()

                trainer_root.protocol("WM_DELETE_WINDOW", on_trainer_close)

            except ImportError as e:
                messagebox.showerror(
                    "Module Error",
                    f"Could not load the Shoulder Motion Trainer module. Error: {str(e)}"
                )
        else:
            self.show_coming_soon(program_type)

    def launch_quiz(self):
        """Open the biomechanics quiz HTML file in the default web browser"""
        quiz_file = r"web\index.html"  # Note the 'r' prefix for raw string # Your HTML quiz file
        quiz_path = os.path.abspath(quiz_file)

        if os.path.exists(quiz_path):
            try:
                import webbrowser
                webbrowser.open(f"file://{quiz_path}")
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Could not open quiz: {str(e)}"
                )
        else:
            messagebox.showerror(
                "File Not Found",
                f"Quiz file not found at: {quiz_path}\n"
                "Please ensure biomechanics_quiz.html is in the same directory."
            )


    def show_coming_soon(self, module_name):
        """Show a message for modules that are not yet implemented"""
        messagebox.showinfo(
            "Coming Soon",
            f"The {module_name} module is coming soon!\n\nPlease try the Hand Motion Analysis module."
        )

    def show_team_info(self):
        """Open the MotionLens team HTML file in the default web browser"""
        team_file = r"team_info.html"  # Raw string for Windows path
        team_path = os.path.abspath(team_file)

        if os.path.exists(team_path):
            try:
                webbrowser.open(f"file://{team_path}")
            except Exception as e:
                tk.messagebox.showerror(
                    "Error",
                    f"Could not open team info: {str(e)}"
                )
        else:
            tk.messagebox.showerror(
                "File Not Found",
                f"Team file not found at: {team_path}\n"
                "Please ensure motionlens_team.html exists at this location."
            )


    def show_figure_details(self, figure_name):
        """Show detailed information about a biomechanics figure"""
        details = {
            "Joint Angle Analysis": (
                "Joint angles are fundamental to understanding human movement. "
                "They determine the range of motion, affect muscle activation patterns, "
                "and influence movement efficiency. Proper joint angles can prevent injuries "
                "and improve performance in sports and daily activities."
            ),
            "Movement Kinematics": (
                "Kinematics describes motion without considering its causes. "
                "It includes the study of position, velocity, and acceleration "
                "of body segments. Kinematic analysis helps identify abnormal "
                "movement patterns that may lead to injury or reduced performance."
            )
        }

        message = details.get(figure_name,
                              "Detailed information about this biomechanics concept will be available soon.")

        detail_window = tk.Toplevel(self.root)
        detail_window.title(figure_name)
        detail_window.geometry("500x400")

        ttk.Label(
            detail_window,
            text=figure_name,
            font=self.header_font,
            style="Accent.TLabel"
        ).pack(pady=(20, 10))

        ttk.Label(
            detail_window,
            text=message,
            font=self.body_font,
            wraplength=450,
            justify=tk.LEFT
        ).pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Add a close button
        ttk.Button(
            detail_window,
            text="Close",
            command=detail_window.destroy
        ).pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = MotionAnalysisLauncher(root)
    root.mainloop()
