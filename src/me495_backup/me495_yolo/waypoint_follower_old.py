# modifyin plan: i want to waypoint_follower to opublish directly to the turtlebots /wheel_cmd topic 
# need to calcualte the appropriate left_velocity and right_velocity values based on the way points 
# msr@shredder:~$ ros2 topic pub /wheel_cmd nuturtlebot_msgs/msg/WheelCmands "{left_velocity: 100, right_velocity: 100}"
# building off of ths ^ 

# # calculate wheel veleoctied based on distance to waypoint 
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Point
# from nuturtlebot_msgs.msg import WheelCommands
# import math

# class WaypointFollower(Node):
#     def __init__(self):
#         super().__init__('waypoint_follower')

#         # Publisher for wheel commands
#         self.wheel_cmd_publisher = self.create_publisher(WheelCommands, '/wheel_cmd', 10)

#         # Subscribe to the waypoints topic
#         self.create_subscription(Point, '/waypoints', self.waypoint_callback, 10)

#         # Waypoints list and robot state
#         self.waypoints = []
#         self.current_x = 0.0
#         self.current_y = 0.0
#         self.current_theta = 0.0  # Orientation in radians
#         self.wheel_radius = 0.033  # Wheel radius in meters (example value)
#         self.wheel_base = 0.16    # Distance between wheels in meters (example value)
#         self.linear_speed = 0.2   # Linear speed in m/s (example value)
#         self.angular_speed = 1.0  # Angular speed in rad/s (example value)

#     def waypoint_callback(self, msg):
#         """Callback to receive waypoints."""
#         self.waypoints.append((msg.x, msg.y))
#         self.get_logger().info(f"Received waypoint: ({msg.x}, {msg.y})")

#     def follow_waypoints(self):
#         """Follow waypoints in sequence."""
#         for waypoint in self.waypoints:
#             self.move_to_waypoint(waypoint[0], waypoint[1])

#     def move_to_waypoint(self, goal_x, goal_y):
#         """Move the robot to a specific waypoint."""
#         while True:
#             # Calculate distance and angle to the waypoint
#             dx = goal_x - self.current_x
#             dy = goal_y - self.current_y
#             distance = math.sqrt(dx**2 + dy**2)
#             target_theta = math.atan2(dy, dx)
#             angle_diff = target_theta - self.current_theta

#             # Normalize angle_diff to [-pi, pi]
#             angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

#             # Rotate the robot to face the waypoint
#             if abs(angle_diff) > 0.1:  # Threshold for angular alignment
#                 self.publish_wheel_commands(0.0, angle_diff * self.angular_speed)
#                 continue

#             # Move the robot forward
#             if distance > 0.1:  # Threshold for reaching the waypoint
#                 self.publish_wheel_commands(self.linear_speed, 0.0)
#             else:
#                 # Stop the robot and update its position
#                 self.publish_wheel_commands(0.0, 0.0)
#                 self.current_x = goal_x
#                 self.current_y = goal_y
#                 self.current_theta = target_theta
#                 break

#     def publish_wheel_commands(self, linear_vel, angular_vel):
#         """Calculate wheel velocities and publish WheelCommands."""
#         left_wheel_velocity = (linear_vel - angular_vel * self.wheel_base / 2) / self.wheel_radius
#         right_wheel_velocity = (linear_vel + angular_vel * self.wheel_base / 2) / self.wheel_radius

#         # Create and publish wheel commands
#         wheel_cmd = WheelCommands()
#         wheel_cmd.left_velocity = int(left_wheel_velocity)
#         wheel_cmd.right_velocity = int(right_wheel_velocity)
#         self.wheel_cmd_publisher.publish(wheel_cmd)

# def main(args=None):
#     rclpy.init(args=args)
#     node = WaypointFollower()
#     rclpy.spin(node)
#     rclpy.shutdown()


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nuturtlebot_msgs.msg import WheelCommands
import math
import pyrealsense2 as rs
import cv2

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        # Publisher for wheel commands
        self.wheel_cmd_publisher = self.create_publisher(WheelCommands, '/wheel_cmd', 10)

        # Subscribe to the waypoints topic
        self.create_subscription(Path, '/waypoints', self.path_callback, 10)
        

        # Waypoints and robot state
        self.waypoints = []
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0  # Orientation in radians
        self.wheel_radius = 0.033  # Example value
        self.wheel_base = 0.16    # Example value
        self.linear_speed = 6.0   # m/s
        self.angular_speed = 1.0  # rad/s

        # Timer to process waypoints
        self.timer = self.create_timer(0.1, self.follow_waypoints)

    def path_callback(self, msg):
        """Callback to receive waypoints as a Path message."""
        self.waypoints = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.get_logger().info(f"Received {len(self.waypoints)} waypoints.")

    def follow_waypoints(self):
        """Process waypoints one by one."""
        if not self.waypoints:
            return

        goal_x, goal_y = self.waypoints[0]  # Get next waypoint

        dx = goal_x - self.current_x
        dy = goal_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)
        target_theta = math.atan2(dy, dx)
        angle_diff = target_theta - self.current_theta
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize to [-pi, pi]

        # Rotate the robot to face the waypoint
        if abs(angle_diff) > 0.1:
            self.publish_wheel_commands(0.0, angle_diff * self.angular_speed)
            return

        # Move the robot forward
        if distance > 0.1:
            self.publish_wheel_commands(self.linear_speed, 0.0)
        else:
            self.get_logger().info(f"Reached waypoint ({goal_x}, {goal_y})")
            self.publish_wheel_commands(0.0, 0.0)
            self.current_x = goal_x
            self.current_y = goal_y
            self.current_theta = target_theta
            self.waypoints.pop(0)  # Remove reached waypoint

    def publish_wheel_commands(self, linear_vel, angular_vel):
        """Calculate wheel velocities and publish WheelCommands."""
        left_wheel_velocity = (linear_vel - angular_vel * self.wheel_base / 2) / self.wheel_radius
        right_wheel_velocity = (linear_vel + angular_vel * self.wheel_base / 2) / self.wheel_radius

        wheel_cmd = WheelCommands()
        wheel_cmd.left_velocity = int(left_wheel_velocity)
        wheel_cmd.right_velocity = int(right_wheel_velocity)
        self.wheel_cmd_publisher.publish(wheel_cmd)

def main():
    rclpy.init()
    node = WaypointFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
