import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nuturtlebot_msgs.msg import WheelCommands
import math
import time

class LocalWaypointFollower(Node):
    def __init__(self):
        super().__init__('local_waypoint_follower')

        # Publisher for /wheel_cmd on TurtleBot
        self.wheel_cmd_publisher = self.create_publisher(WheelCommands, '/wheel_cmd', 10)

        # Example waypoints (replace with YOLO-generated waypoints)
        self.waypoints = [
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0)
        ]

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0  # Orientation in radians
        self.wheel_radius = 0.033  # Wheel radius in meters
        self.wheel_base = 0.16    # Distance between wheels in meters
        self.linear_speed = 0.2   # Linear speed in m/s
        self.angular_speed = 1.0  # Angular speed in rad/s

        self.follow_waypoints()

    def follow_waypoints(self):
        """Send wheel commands to follow waypoints."""
        for goal_x, goal_y in self.waypoints:
            self.move_to_waypoint(goal_x, goal_y)

    def move_to_waypoint(self, goal_x, goal_y):
        """Calculate and send wheel commands to reach the waypoint."""
        while True:
            dx = goal_x - self.current_x
            dy = goal_y - self.current_y
            distance = math.sqrt(dx**2 + dy**2)
            target_theta = math.atan2(dy, dx)
            angle_diff = target_theta - self.current_theta

            # Normalize angle_diff to [-pi, pi]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

            # Rotate if needed
            if abs(angle_diff) > 0.1:  
                self.publish_wheel_commands(0.0, angle_diff * self.angular_speed)
                time.sleep(0.5)
                continue

            # Move forward
            if distance > 0.1:  
                self.publish_wheel_commands(self.linear_speed, 0.0)
                time.sleep(0.5)
            else:
                self.publish_wheel_commands(0.0, 0.0)  # Stop
                self.current_x, self.current_y, self.current_theta = goal_x, goal_y, target_theta
                break

    def publish_wheel_commands(self, linear_vel, angular_vel):
        """Convert velocity to wheel commands and send to TurtleBot."""
        left_wheel_velocity = (linear_vel - angular_vel * self.wheel_base / 2) / self.wheel_radius
        right_wheel_velocity = (linear_vel + angular_vel * self.wheel_base / 2) / self.wheel_radius

        wheel_cmd = WheelCommands()
        wheel_cmd.left_velocity = int(left_wheel_velocity)
        wheel_cmd.right_velocity = int(right_wheel_velocity)

        self.wheel_cmd_publisher.publish(wheel_cmd)
        self.get_logger().info(f"Sent wheel commands: L={wheel_cmd.left_velocity}, R={wheel_cmd.right_velocity}")

def main(args=None):
    rclpy.init(args=args)
    node = LocalWaypointFollower()
    rclpy.spin(node)
    rclpy.shutdown()
