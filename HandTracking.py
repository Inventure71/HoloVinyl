import math
import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class HandTrackingManager:
    def __init__(self):
        """VARIABLES"""
        self.is_pinching = False
        self.latest_result = None
        self.frame_distances = []
        self.N = 8
        self.PINCH_THRESHOLD = 50
        self.SEPARATION_THRESHOLD = 100
        self.model_path = 'custom_models/hand_landmarker.task'
        self.touch_margin = 10 # radius of the circle to click, with single point would be ass


        self.mouse_position = (0,0)

        # Set up the hand landmark detector
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self.print_result
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def analyze_frame(self, frame, frame_timestamp_ms):
        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks asynchronously
        self.landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Draw landmarks if detection has results
        if self.latest_result is not None:

            if self.identify_pinch_gesture(self.latest_result, frame):
                print("Pinch Detected!")
                cv2.putText(frame, "Pinch Detected!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            frame = self.draw_landmarks(frame, self.latest_result)

        # Display the frame
        cv2.imshow("Hand Tracking", frame)

    def identify_pinch_gesture(self, latest_result, frame): #
        """Identifies a pinch gesture when the distance shrinks from >100 to <30 within N frames."""
        if not latest_result.hand_landmarks:
            return False

        height, width, _ = frame.shape
        hand = latest_result.hand_landmarks[0]

        start_point = (int(hand[4].x * width), int(hand[4].y * height))
        end_point = (int(hand[8].x * width), int(hand[8].y * height))
        mid_point = (int((hand[4].x + hand[8].x) * width / 2), int((hand[4].y + hand[8].y) * height / 2))
        last_distance = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)

        self.mouse_position = mid_point

        self.frame_distances.append(last_distance)
        if len(self.frame_distances) > self.N:
            self.frame_distances.pop(0)  # Keep only last N frames

        if last_distance <=  self.PINCH_THRESHOLD:
            if self.is_pinching:
                print("Hand still pinching", self.mouse_position)
            else:
                print("Hand closed")
            for old_distance in self.frame_distances:
                if old_distance >=  self.SEPARATION_THRESHOLD:
                    self.is_pinching = True
                    self.frame_distances = []
                    return True

            self.is_pinching = True
            return False

        self.is_pinching = False
        return False

    def draw_landmarks(self, frame, result):
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

            # half_point = end_point[0] - int((end_point[0] - start_point[0]) / 2), end_point[1] - int((end_point[1] - start_point[1]) / 2)

            start_point = (int(hand[4].x * width), int(hand[4].y * height))
            end_point = (int(hand[8].x * width), int(hand[8].y * height))
            distance = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
            length = "Distance between index and thumb " + str(distance)

            self.frame_distances.append(distance)
            if len(self.frame_distances) > self.N:
                self.frame_distances.pop(0)  # Keep only last N frames

            cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
            cv2.putText(frame, length, (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        return frame

    def print_result(self, result: HandLandmarkerResult, output_image, timestamp_ms: int):
        """Callback function to process results."""
        self.latest_result = result  # Store latest result for visualization


if __name__ == "__main__":
    hand_tracking_manager = HandTrackingManager()
    webcam = cv2.VideoCapture(0)
    while webcam.isOpened():
        frame_timestamp_ms = int(time.time() * 1000)
        ret, frame = webcam.read()
        hand_tracking_manager.analyze_frame(frame, frame_timestamp_ms)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()