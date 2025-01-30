from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time


class RobotHandling:
    def __init__(self):
        self.bot = InterbotixManipulatorXS(
            robot_model='wx250s',
            group_name='arm',
            gripper_name='gripper',
        )

        self.safe_location = (0, 0, 0.3)

        self.vertical_pitch = 1.43

        # coordinates of the tower
        self.tower_coords = (0, 0, 0.1)

        # coordinates of the closest cell (0)
        self.cell_0_cord = (0, 0, 0.23)

        self.distance_between_cells = 0.031

        self.adjustment_multiplier = 0.0025

        robot_startup()

    def move_to_safe_location(self):
        self.bot.arm.set_ee_pose_components(self.safe_location[0], self.safe_location[1], self.safe_location[2],
                                            pitch=self.vertical_pitch)

    def pickup_cell(self):
        self.move_to_safe_location()

        self.bot.gripper.release(2.0)

        self.bot.arm.set_ee_pose_components(self.tower_coords[0], self.tower_coords[1], self.tower_coords[2] + 0.1,
                                            pitch=self.vertical_pitch)
        time.sleep(0.1)
        self.bot.arm.set_ee_pose_components(self.tower_coords[0], self.tower_coords[1], self.tower_coords[2],
                                            pitch=self.vertical_pitch)

        self.bot.gripper.grasp(2.0)

        self.move_to_safe_location()

    def move_to_cell(self, column: int):
        """
        Moves the arm to drop a piece at the specified column.
        column should be an int from 0 to 6 (for 7 columns).
        """
        print(f"[Robot] Received command to move to column {column}.")

        self.move_to_safe_location()

        # Compute the target X offset based on the column
        # target_x = 0.18 + int(column) * 0.031

        target_x = self.cell_0_cord[0] + int(column) * self.distance_between_cells
        target_y = self.cell_0_cord[1]
        target_z = self.cell_0_cord[2]

        print(f"[Robot] Moving end effector to X={target_x:.3f}, Z={self.z:.3f}.")
        self.bot.arm.set_ee_pose_components(x=target_x, y=target_y, z=target_z, pitch=1.43)

        # perform a slight Z adjustment if column < 4
        if int(column) < 4:
            adjustment_z = -(8 - int(column)) * self.adjustment_multiplier
            print(f"[Robot] Applying Z offset: {adjustment_z:.4f}")
            self.bot.arm.set_ee_cartesian_trajectory(z=adjustment_z)

        # Drop the piece
        self.bot.gripper.release(2.0)
        time.sleep(0.1)

        # Move arm back up
        self.move_to_safe_location()

        print("[Robot] Move completed.\n")

    def stop(self):
        """Move the robot to a 'sleep' or safe pose."""
        print("[Robot] Stopping and going to sleep pose.")
        self.move_to_safe_location()
        self.bot.arm.go_to_sleep_pose()

if __name__ == '__main__':
    robot = RobotHandling()
    while True:
        cell = input("Cell: ")
        if cell == "q":
            break
        else:
            try:
                cell = int(cell)
                robot.move_to_cell(1)
            except:
                print("input was not an int")

    robot.stop()
