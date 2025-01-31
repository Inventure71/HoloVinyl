import cv2
import numpy as np
import os

class ManualBoardCalibration:
    def __init__(self, camera, load_last_calibration=True):
        self.calibration_file = "variables/calibration.npy"
        if load_last_calibration:
            self.points = self.load_calibration()
            if self.points is not None:
                print("Loaded previous calibration points:")
                print(self.points)
                self.calibrated = True
            else:
                self.capture = cv2.VideoCapture(camera)
                self.points = []
                self.calibrated = False
        else:
            self.capture = cv2.VideoCapture(camera)
            self.points = []  # Stores manually selected points
            self.calibrated = False

    def load_calibration(self):
        """Loads the calibration points from a file if available."""
        if os.path.exists(self.calibration_file):
            try:
                points = np.load(self.calibration_file)
                return points.astype(np.float32)
            except Exception as e:
                print(f"Error loading calibration file: {e}")
        return None

    def save_calibration(self, points):
        """Saves the calibration points to a file."""
        try:
            np.save(self.calibration_file, points)
            print("Calibration points saved successfully.")
        except Exception as e:
            print(f"Error saving calibration file: {e}")

    def select_points(self, event, x, y, flags, param):
        """Captures mouse clicks to select points."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: {x}, {y}")

    def calibrate_board(self):
        """Allows the user to manually select four points on the board."""
        print("Click on the four corners of the board in the following order:")
        print("1️⃣ Top-Left → 2️⃣ Top-Right → 3️⃣ Bottom-Right → 4️⃣ Bottom-Left")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.select_points)

        while not self.calibrated:
            ret, frame = self.capture.read()
            if not ret:
                break

            for i, point in enumerate(self.points):
                cv2.circle(frame, point, 10, (0, 255, 0), -1)
                cv2.putText(frame, f"{i + 1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Calibration", frame)

            if len(self.points) == 4:
                self.calibrated = True
                print("Calibration complete! Points selected:", self.points)
                break

            if cv2.waitKey(1) == 27:
                print("Calibration cancelled.")
                self.points = []
                break

        cv2.destroyWindow("Calibration")

    def order_points(self, pts):
        """Ensures the four points are ordered correctly for perspective transformation."""
        pts = np.array(pts, dtype=np.float32)
        sorted_pts = sorted(pts, key=lambda p: (p[1], p[0]))
        top_pts = sorted(sorted_pts[:2], key=lambda p: p[0])
        bottom_pts = sorted(sorted_pts[2:], key=lambda p: p[0])
        ordered_pts = np.array([top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]], dtype=np.float32)
        return ordered_pts

    def run(self):
        """Runs the manual calibration and returns ordered points."""
        if self.calibrated:
            print("Calibration already complete. Exiting.")
            return self.points
        else:
            self.calibrate_board()
            if not self.calibrated or len(self.points) != 4:
                print("Calibration failed. Exiting.")
                self.capture.release()
                cv2.destroyAllWindows()
                return None

            self.capture.release()
            cv2.destroyAllWindows()

            ordered_points = self.order_points(self.points)
            self.save_calibration(ordered_points)

            print("\nFinal Detected Corners (Ordered for Perspective Warp):")
            print(ordered_points)
            return ordered_points

