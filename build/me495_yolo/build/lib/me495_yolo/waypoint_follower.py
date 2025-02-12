

# need to subscribe to /waypoints

# iterate through each waypoint (waypoints publishes a /nav_msgs/Path message)

# use /cmd_vel to convert waypoitns to velocoty commands 

# stop when close to the waypoinf 

# turn towards eachj waypoint beofre movign forwards

# stop at last waypoint 



import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import math
from std_msgs.msg import Float32MultiArray

import time

class TurtleBotWaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        # Subscribe to waypoints
        self.subscription = self.create_subscription(
            Path,
            '/waypoints',
            self.waypoints_callback,
            10
        )
        self.create_subscription(Float32MultiArray, '/turtlebot_position', self.position_callback, 10)


        # Publisher for movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_waypoints = []  # Store received waypoints
        self.current_index = 0       # Track which waypoint we are moving to
        self.robot_position = (0.0, 0.0)
    

        self.timer = self.create_timer(0.1, self.navigate_to_waypoint)
    
    def update_turtlebot_position(self, x, y):
        """Update the TurtleBot's position using the detected blue object."""
        self.robot_position = (x, y)
        self.get_logger().info(f"🔵 Updated TurtleBot Position: X={x:.2f}, Y={y:.2f}")


    def waypoints_callback(self, msg):
        """Receive waypoints and store them."""
        self.current_waypoints = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.current_index = 0  # Start from the first waypoint
        self.get_logger().info(f"✅ Received {len(self.current_waypoints)} waypoints.")

    def position_callback(self, msg):
        """Update the TurtleBot's position from the YOLO node."""
        if len(msg.data) >= 2:
            self.robot_position = (msg.data[0], msg.data[1])
            self.get_logger().info(f"🔵 Updated TurtleBot Position from YOLO: X={self.robot_position[0]:.2f}, Y={self.robot_position[1]:.2f}")
        else:
         self.get_logger().warn("⚠️ Received invalid TurtleBot position message!")


    # def navigate_to_waypoint(self):
    #     """Move the TurtleBot to each waypoint in sequence."""
    #     if not self.current_waypoints or self.current_index >= len(self.current_waypoints):
    #         self.stop_robot()
    #         return
        
    #     # Prevent error if robot_position is None --> if delay in detection 
    #     if self.robot_position is None:
    #         self.get_logger().warn("⚠️ Robot position not updated yet! Waiting for update...")
    #         return

    #     goal_x, goal_y = self.current_waypoints[self.current_index]
    #     robot_x, robot_y = self.robot_position  

    #     distance = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
    #     angle_to_target = math.atan2(goal_y - robot_y, goal_x - robot_x)

    #     if distance < 0.1:  # Close enough to waypoint
    #         self.get_logger().info(f"✅ Reached waypoint {self.current_index + 1}/{len(self.current_waypoints)}")
    #         self.current_index += 1
    #         return

    #     # Create movement command
    #     twist = Twist()
    #     twist.linear.x = min(0.2, distance)  # Move forward
    #     twist.angular.z = min(0.5, angle_to_target)  # Turn towards waypoint

    #     self.cmd_vel_pub.publish(twist)

    def navigate_to_waypoint(self):
        """Move the TurtleBot to each waypoint in sequence."""
        if not self.current_waypoints or self.current_index >= len(self.current_waypoints):
            self.stop_robot()
            return
        
        # Make sure we have a valid position update
        if self.robot_position is None:
            self.get_logger().warn("⚠️ Robot position not updated yet! Waiting for update...")
            return

        # Get current waypoint
        goal_x, goal_y = self.current_waypoints[self.current_index]
        robot_x, robot_y = self.robot_position  # Use real-time position from YOLO

        # Compute distance and angle to target
        distance = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        angle_to_target = math.atan2(goal_y - robot_y, goal_x - robot_x)

        # Print debug info
        self.get_logger().info(f"🎯 Target: X={goal_x:.2f}, Y={goal_y:.2f}")
        self.get_logger().info(f"🤖 TurtleBot Position: X={robot_x:.2f}, Y={robot_y:.2f}")
        self.get_logger().info(f"📏 Distance to Waypoint: {distance:.3f} meters")

        # If close enough to waypoint, move to the next one
        if distance < 0.1:
            self.get_logger().info(f"✅ Reached waypoint {self.current_index + 1}/{len(self.current_waypoints)}")
            self.current_index += 1
            return

        # Create movement command
        twist = Twist()

        # First, rotate toward the waypoint
        angle_diff = angle_to_target  # Simplified for now; could be improved with real orientation
        if abs(angle_diff) > 0.1:  
            twist.angular.z = 0.5 if angle_diff > 0 else -0.5  # Turn toward target
        else:
            twist.linear.x = min(0.2, distance)  # Move forward only when aligned

        # Debug message before publishing command
        self.get_logger().info(f"🚀 Sending /cmd_vel: Linear={twist.linear.x:.2f}, Angular={twist.angular.z:.2f}")

        # Publish movement command
        self.cmd_vel_pub.publish(twist)


    def stop_robot(self):
        """Stop the robot when waypoints are complete."""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("🏁 All waypoints reached! Stopping.")

def main():
    rclpy.init()
    node = TurtleBotWaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
