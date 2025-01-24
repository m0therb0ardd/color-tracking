import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.interpolate import splprep, splev


class WaypointNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator')

        # Subscriptions
        self.create_subscription(Float32MultiArray, 'path_points', self.path_callback, 10)
        self.create_subscription(Image, 'image', self.image_callback, 10)

        # Publishers
        self.waypoint_publisher = self.create_publisher(Path, 'waypoints', 10)
        self.image_publisher = self.create_publisher(Image, 'waypoint_image', 10)

        # ROS utilities
        self.bridge = CvBridge()
        self.latest_image = None  # Store the latest image

        # Parameters for B-spline
        self.declare_parameter('num_waypoints', 25)
        self.declare_parameter('smoothness', 10)

    def image_callback(self, msg):
        """Store the latest image for waypoint annotation."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def path_callback(self, msg):
        """Generate waypoints and annotate the image with them."""
        # Convert the flat array back to a list of (x, y) points
        path = [(msg.data[i], msg.data[i + 1]) for i in range(0, len(msg.data), 2)]

        # Generate waypoints using B-spline
        num_waypoints = self.get_parameter('num_waypoints').get_parameter_value().integer_value
        smoothness = self.get_parameter('smoothness').get_parameter_value().double_value
        waypoints = self.create_b_spline_waypoints(path, num_waypoints, smoothness)

        # Publish the waypoints as a ROS Path message
        self.publish_b_spline_path(waypoints)

        # Annotate and publish the image with waypoints
        if self.latest_image is not None:
            self.annotate_and_publish_image(self.latest_image, waypoints)

    def create_b_spline_waypoints(self, path, num_waypoints=25, smoothness=10):
        """Generate waypoints using a B-spline curve."""
        if len(path) < 4:  # B-splines require at least 4 points
            self.get_logger().warn("Path must have at least 4 points for B-spline.")
            return path

        # Extract x and y coordinates from the path
        x = [p[0] for p in path]
        y = [p[1] for p in path]

        # Fit a B-spline to the path
        try:
            tck, _ = splprep([x, y], s=smoothness)
        except Exception as e:
            self.get_logger().error(f"Failed to fit B-spline: {e}")
            return path

        # Generate evenly spaced waypoints along the B-spline
        u = np.linspace(0, 1, num_waypoints)
        x_smooth, y_smooth = splev(u, tck)

        # Convert to a list of (x, y) waypoints
        waypoints = [(int(xi), int(yi)) for xi, yi in zip(x_smooth, y_smooth)]
        return waypoints

    def publish_b_spline_path(self, waypoints):
        """Publish the waypoints as a ROS Path message."""
        path_msg = Path()
        path_msg.header.frame_id = "map"  # Set the appropriate frame
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x =float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.waypoint_publisher.publish(path_msg)

    def annotate_and_publish_image(self, image, waypoints):
        """Annotate the image with waypoints and publish it."""
        annotated_image = image.copy()

        # Draw each waypoint as a red circle
        for x, y in waypoints:
            cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)  # Red circle

        # Convert the annotated image back to a ROS Image message
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')

        # Publish the annotated image
        self.image_publisher.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointNode()
    rclpy.spin(node)
    rclpy.shutdown()
