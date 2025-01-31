from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracking:
  def __init__(self, number_of_hands=1, stream_mode=False):
    self.MARGIN = 10  # pixels
    self.FONT_SIZE = 1
    self.FONT_THICKNESS = 1
    self.HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    #Image optimized
    base_options = python.BaseOptions(model_asset_path='custom_models/hand_landmarker.task')
    #Live optimized

    self.stream_mode = False

    if stream_mode:
      options = vision.HandLandmarkerOptions(base_options=base_options,
                                             running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                                             result_callback=self.print_result_stream,
                                             num_hands=number_of_hands)
    else:
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                                 num_hands=number_of_hands)

    self.detector = vision.HandLandmarker.create_from_options(options)

  def print_result_stream(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    self.display_overlay(output_image, result)

  def draw_landmarks_on_image(self, rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]

      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

      # Get the top left corner of the detected hand's bounding box.
      height, width, _ = annotated_image.shape
      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - self.MARGIN

      # Draw handedness (left or right hand) on the image.
      cv2.putText(annotated_image, f"{handedness[0].category_name}",
                  (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                  self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

  def identify_source(self, source):
    if self.stream_mode:
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=source)
      self.detector.detect_async(mp_image, 0)

    else:
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=source)
      detection_result = self.detector.detect(mp_image)
      self.display_overlay(mp_image, detection_result)

  def display_overlay(self, mp_image, detection_result):
    annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
  hand_tracking = HandTracking(number_of_hands=1, stream_mode=False)

  image = cv2.imread("custom_models/image.jpg")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  hand_tracking.identify_source(image)



