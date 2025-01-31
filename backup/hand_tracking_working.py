import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe components
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Load hand detection model
model_path = 'custom_models/hand_landmarker.task'

# Open webcam
webcam = cv2.VideoCapture(0)

# Global variable to store results
latest_result = None


def draw_landmarks(frame, result):
    """Draw hand landmarks on the frame and overlay the Z-coordinates of the index finger."""
    if not result.hand_landmarks:
        return frame

    height, width, _ = frame.shape
    for hand in result.hand_landmarks:
        for i, landmark in enumerate(hand):
            x, y = int(landmark.x * width), int(landmark.y * height)

            # Draw landmark points
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dot

            # Display Z-coordinates only for index finger joints
            if i in [5, 6, 7, 8]:  # Index finger landmarks
                z_text = f"Z: {landmark.z:.4f}"
                cv2.putText(frame, z_text, (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    return frame


def print_result(result: HandLandmarkerResult, output_image, timestamp_ms: int):
    """Callback function to process results."""
    global latest_result
    latest_result = result  # Store latest result for visualization


# Set up the hand landmark detector
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

with HandLandmarker.create_from_options(options) as landmarker:
    while webcam.isOpened():
        frame_timestamp_ms = int(time.time() * 1000)
        ret, frame = webcam.read()
        if not ret:
            continue  # Skip frame if webcam read fails

        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Draw landmarks if detection has results
        if latest_result is not None:
            frame = draw_landmarks(frame, latest_result)

        # Display the frame
        cv2.imshow("Hand Tracking", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
webcam.release()
cv2.destroyAllWindows()
