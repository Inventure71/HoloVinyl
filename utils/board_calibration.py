import cv2
import numpy as np


class ManualBoardCalibration:
    def __init__(self):
        self.capture = cv2.VideoCapture(4)
        self.points = []  # Stores manually selected points
        self.calibrated = False

    def select_points(self, event, x, y, flags, param):
        """
        Callback function to capture mouse clicks.
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: {x}, {y}")

    def calibrate_board(self):
        """
        Allows the user to manually select four points on the board.
        """
        print("Click on the four corners of the board in the following order:")
        print("Top-left, Top-right, Bottom-right, Bottom-left")

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.select_points)

        while not self.calibrated:
            ret, frame = self.capture.read()
            if not ret:
                break

            # Show selected points on the live feed
            for i, point in enumerate(self.points):
                cv2.circle(frame, point, 10, (0, 255, 0), -1)
                cv2.putText(frame, f"{i + 1}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Calibration", frame)

            # Check if 4 points have been selected
            if len(self.points) == 4:
                self.calibrated = True
                print("Calibration complete! Points selected:", self.points)
                break

            # Exit with 'Esc'
            if cv2.waitKey(1) == 27:
                print("Calibration cancelled.")
                break

        cv2.destroyWindow("Calibration")

    def run(self):
        """
        Main loop to display the transformed board after calibration.
        """
        # Calibration phase
        self.calibrate_board()
        if not self.calibrated or len(self.points) != 4:
            print("Calibration failed. Exiting.")
            self.capture.release()
            cv2.destroyAllWindows()
            return

        cv2.destroyAllWindows()


        return self.points

        print("Using calibrated view.")
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            # Transform to the calibrated square view
            transformed_view = transform_to_square(frame)

            # Display the transformed board
            cv2.imshow("Transformed Board", transformed_view)

            # Exit with 'Esc'
            if cv2.waitKey(1) == 27:
                break

        self.capture.release()


if __name__ == "__main__":
    calibration = ManualBoardCalibration()
    calibration.run()
