

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

        # Publisher for movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_waypoints = []  # Store received waypoints
        self.current_index = 0       # Track which waypoint we are moving to
        self.robot_position = None
    

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

    def navigate_to_waypoint(self):
        """Move the TurtleBot to each waypoint in sequence."""
        if not self.current_waypoints or self.current_index >= len(self.current_waypoints):
            self.stop_robot()
            return

        goal_x, goal_y = self.current_waypoints[self.current_index]
        robot_x, robot_y = self.robot_position  # Replace with real odometry

        distance = math.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
        angle_to_target = math.atan2(goal_y - robot_y, goal_x - robot_x)

        if distance < 0.1:  # Close enough to waypoint
            self.get_logger().info(f"✅ Reached waypoint {self.current_index + 1}/{len(self.current_waypoints)}")
            self.current_index += 1
            return

        # Create movement command
        twist = Twist()
        twist.linear.x = min(0.2, distance)  # Move forward
        twist.angular.z = min(0.5, angle_to_target)  # Turn towards waypoint

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
