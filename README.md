<img width="950" height="605" alt="Screenshot 2026-03-12 225507" src="https://github.com/user-attachments/assets/51eac502-2a58-49f6-8926-c99678183329" />



Real-Time Style Change
An application that applies artistic style transformations to a live webcam feed using Computer Vision.

Overview
This project implements Neural Style Transfer (NST) in a real-time environment. By leveraging OpenCV and deep learning models (VGG-19), the application captures video from a webcam and applies artistic styles to the frames. It includes features for person segmentation and background styling to create a dynamic visual experience.

Key Features
Live Transformation: Processes webcam input frame-by-frame for immediate stylistic feedback.

Person Segmentation: Specifically detects and styles the person in the frame.

Background Effects: Ability to change or style the background independently of the subject.

Neural Style Transfer: Utilizes VGG-19 architecture for high-quality artistic rendering.

Tech Stack
Language: Python

Libraries: OpenCV, NumPy

Model: VGG-19 (Neural Style Transfer)

Getting Started
Prerequisites
Make sure you have Python installed, then install the necessary dependencies:

Bash
pip install opencv-python numpy
Installation
Clone the repository:

Bash
git clone https://github.com/shry28/Real-time-style-change.git
Navigate to the project directory:

Bash
cd Real-time-style-change
Run the application:

Bash
python main.py

Usage
Once the application starts, your webcam will activate.

The styled output will appear in a new window.

To Exit: Press q on your keyboard.

Project Structure
main.py: The core script handling the video stream and style application.

models/: Directory containing the pre-trained weights for style transfer.

requirements.txt: List of Python packages required.
