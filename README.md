![hero_image](https://github.com/user-attachments/assets/34e02cb2-1563-4339-b91e-bd5635880007)
# 🧠 MotionLens – Interactive Biomechanics Learning Platform


An interactive desktop application that uses **computer vision and deep learning** to analyze human motion and teach biomechanics concepts through real-time feedback and guided exercises. Built with **Tkinter**, this tool is ideal for students, physical therapists, and athletes seeking to improve their movement patterns.

---

## 📷 Project Overview

This platform provides an educational environment where users can learn about biomechanics principles using real-time motion tracking. With a clean GUI, users can:

🔹 Launch modules for hand motion, joint movement, and moment arm analysis  
🔹 Access interactive quizzes on biomechanics topics  
🔹 View biomechanical figures and explanations  
🔹 Navigate between modules seamlessly  

The system integrates computer vision models (imported from external files) to track user movements and provide real-time feedback.

---
### 🖥️ Application Screenshot

<img width="960" alt="Screenshot 2025-05-14 000730" src="https://github.com/user-attachments/assets/0b600f5a-de22-436e-b85e-b89e478f0762" />


> *Main dashboard showing biomechanics modules and quiz platform.*

## 💻 Features

✅ **Interactive Learning Modules** – Three core modules: Hand Motion, Joint Movement, Moment Arm  
✅ **Computer Vision Integration** – Connects to camera-based motion tracking systems  
✅ **Educational Quizzes** – Launch HTML-based quizzes in the default browser  
✅ **Animated Hero Section** – Engaging visual introduction with image animation  
✅ **Modern UI/UX Design** – Light/dark theme support with smooth scrolling and hover effects  

---

## 🧠 Core Concepts Covered

- **Hand Motion Analysis**: Linear and angular hand movements
- **Joint Movement Analysis**: Abduction/adduction motions
- **Moment Arm Analysis**: Torque and lever mechanics in limb movements
- **Impulse-Force Concept**: Force-time relationship in movement
- **Kinematics**: Position, velocity, and acceleration of body segments

---

## 🛠️ Tech Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Tkinter      | GUI development                |
| PIL / Pillow | Image handling and processing  |
| sv_ttk       | Modern ttk themes              |
| webbrowser   | Open quiz and team info pages  |
| OpenCV (external) | Camera input and tracking    |

---

## 📁 Project Structure

```
├── main.py                    # Main launcher app
├── Computer_vision.py         # Hand motion trainer logic
├── Computer_vision2.py        # Moment arm trainer logic
├── Computer_vision3.py        # Shoulder/joint motion trainer logic
├── assets/                    # Icons, images, animations
├── web/                       # Biomechanics quiz HTML
└── team_info.html             # Team information page
```

---

## 🔍 Application Flow

### 🔄 How It Works

1. **User Interface Initialization**
   - The `MotionAnalysisLauncher` class initializes the main window using Tkinter.
   - A responsive scrollable interface is created using Canvas and Frames.
   - Modern styles are applied using `sv_ttk` and custom styling.
   
### 🖥️ Application Screenshot

<img width="960" alt="Screenshot 2025-05-14 000730" src="https://github.com/user-attachments/assets/adb88455-5957-44ea-8aed-78ee9ca71129" />

> *Main dashboard showing biomechanics modules and quiz platform.*
      

2. **Module Selection**
   - Users click on any module card (e.g., Hand Motion, Moment Arm).
   - Based on selection, the corresponding Python file (`Computer_vision.py`, etc.) is imported dynamically.
   - A new Toplevel window opens with the selected module's GUI and functionality.

3. **Computer Vision Integration**
   - Each module contains a separate trainer class (`HandMotionTrainer`, `MomentArmTrainer`, etc.).
   - These classes connect to your camera and use OpenCV + Deep Learning models (not shown in this code but expected to be in those files).
   - Real-time tracking and feedback are displayed in these windows.
  
    ### 🤖 Computer Vision Module

<img width="803" alt="Screenshot 2025-05-14 002433" src="https://github.com/user-attachments/assets/e2e7d208-7594-4a23-b6a1-4280c8f40fcc" />
<img width="803" alt="Screenshot 2025-05-14 002406" src="https://github.com/user-attachments/assets/d4ab0886-3042-4061-ab4d-22942e79bf06" />
<img width="803" alt="Screenshot 2025-05-14 002514" src="https://github.com/user-attachments/assets/a5aa0b17-a99e-4ffa-9121-71f792eeb755" />


> *Example of how hand motion tracking could be visualized during training.*

4. **Quiz Platform**
   - On clicking "Quiz Platform", the program opens an HTML quiz file (`web/index.html`) in the default browser.
   - This allows for interactive, web-based assessment without needing additional frameworks like Flask.
  
     ### 🎮 Quiz Section Preview

<img width="960" alt="Screenshot 2025-05-14 000859" src="https://github.com/user-attachments/assets/86bf4986-9bf9-4014-be19-982454f35a26" />

> *HTML-based quiz opened in browser for interactive learning and testing.*

5. **Team & Figure Information**
   - Clicking "Who Us?" opens `team_info.html` in the browser.
   - Clicking "Learn More" on any biomechanics figure opens a detailed pop-up explanation.
  
### 🎮 Team Section Preview

<img width="960" alt="Screenshot 2025-05-14 000811" src="https://github.com/user-attachments/assets/dcdb4004-dec9-4704-9be0-ea3419c9dc70" />

> *HTML-based team-info opened in browser .*

6. **Animation & Visuals**
   - The hero section includes animated frames stored as PNGs in the `assets/` folder.
   - Hover effects and modern card designs enhance interactivity.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install pillow tk sv-ttk
```

> Note: `Computer_vision*.py` modules must exist and contain your actual motion detection logic.

### 2️⃣ Launch the App

```bash
python main.py
```

### 3️⃣ Use the Platform

- Click on any **Launch Module** button to start a specific motion trainer
- Use the **Quiz Platform** to test your biomechanics knowledge
- Click **Who Us?** to see team info

---

## 🔬 Use Cases

This tool is ideal for:

🎓 **Biomechanics Education**  
🧑‍⚕️ **Physical Therapy Training**  
🏃‍♂️ **Athletic Performance Enhancement**  
🖥️ **Interactive eLearning Modules**

---

## 🙌 Contributors

Special thanks to our highly talented team:  
**Team 6** – For creating this interactive motion learning platform  
- **Abdullah Gamil**
- **Ibrahim Abdelqader**
- **Abdulrahman Hassan**
- **Mohmad Ehab**
- **Ahmed Mahmoud**

---

## 🔗 Contact

📧 **Email**: [anas.bayoumi05@eng-st.cu.edu.eg](mailto:anas.bayoumi05@eng-st.cu.edu.eg)  
🔗 **LinkedIn**: [Anas Mohamed](https://www.linkedin.com/in/anas-mohamed-716959313/)
