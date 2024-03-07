# Sign Language Recognition with Streamlit

This project is a real-time sign language recognition application built using Streamlit, MediaPipe, and TensorFlow Lite. It allows users to use their webcam or upload a video file to recognize sign language gestures and actions.

## Features

- Real-time sign language recognition using webcam.
- Sign language recognition from uploaded video files.
- Displays recognized sign language actions along with their probabilities.
- Supports multiple sign language gestures.

## Requirements

- Python 3.x
- OpenCV (cv2)
- MediaPipe
- NumPy
- Streamlit
- TensorFlow Lite

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/sign-language-recognition.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Choose an option:
    - **Use Webcam**: Allows real-time sign language recognition using the webcam.
    - **Upload Video File**: Allows uploading a video file for sign language recognition.

3. Follow the instructions on the Streamlit web interface to interact with the application.

## File Structure

- **app.py**: Main Python script containing the Streamlit application code.
- **mediapipe5.tflite**: TensorFlow Lite model for sign language recognition.
- **requirements.txt**: File containing the required Python dependencies.
