import cv2
import mediapipe as mp
import numpy as np
import tempfile
import streamlit as st
from tensorflow import lite

gloss_list = ['doctor', 'emergency', 'fire', 'firefighter', 'help', 'hurt', 'medicine', 'police']

frame_count = 0
frame_step = 4
MAX_SEQ_LENGTH = 15
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
model_path = 'mediapipe5.tflite'
interpreter = lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

frame_sequence = []


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    return results


def getKeypoint(frame):
    results = mediapipe_detection(frame, holistic)
    return extract_keypoints(results).astype(np.float32)


def predict_action():
    input_data = np.expand_dims(frame_sequence, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]
    max_prob_index = np.argmax(res)
    prob = res[max_prob_index]
    return gloss_list[max_prob_index], prob


def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Gesture Recognition with FastAPI and Streamlit")
    st.caption("Powered by OpenCV, Streamlit")

    option = st.radio("Choose an option:", ("Use Webcam", "Upload Video File"))
    cap = None

    if option == "Use Webcam":
        cap = cv2.VideoCapture(0)
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())

            video_path = temp_file.name
            cap = cv2.VideoCapture(video_path)
        else:
            st.warning("Please upload a video file.")

    if 'cap' in locals():
        frame_placeholder = st.empty()
        output = st.empty()
        stop_button_pressed = st.button("Stop")
        global frame_count, frame_sequence

        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("Video Capture Ended")
                break
            frame_count += 1
            keypoint = getKeypoint(frame)

            if frame_count % frame_step == 0:
                frame_sequence.append(keypoint.astype(np.float32))
                frame_sequence = frame_sequence[-MAX_SEQ_LENGTH:]

            if len(frame_sequence) == MAX_SEQ_LENGTH:
                action, prob = predict_action()
                if prob > 0.85:
                    output.text(action + str(prob))
                else:
                    output.text('')
            else:
                output.text('')
            frame_placeholder.image(frame, channels="BGR")
            if stop_button_pressed:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
