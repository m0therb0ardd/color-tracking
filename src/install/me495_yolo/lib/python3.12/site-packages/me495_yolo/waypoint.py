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
from visualization_msgs.msg import Marker, MarkerArray  # Import for visualization
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped
import tf_transformations


class WaypointNode(Node):
    def __init__(self):
        super().__init__('waypoint_generator')

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        # Subscriptions
        self.create_subscription(Float32MultiArray, 'path_points', self.path_callback, 10)
        self.create_subscription(Image, 'image', self.image_callback, 10)

        # Publishers
        self.waypoint_publisher = self.create_publisher(Path, 'waypoints', 10)
        self.image_publisher = self.create_publisher(Image, 'waypoint_image', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'waypoint_markers', 10)


        # ROS utilities
        self.bridge = CvBridge()
        self.latest_image = None  # Store the latest image

        # Parameters for B-spline
        self.declare_parameter('num_waypoints', 50)
        self.declare_parameter('smoothness', 10)

    def image_callback(self, msg):
        """Store the latest image for waypoint annotation."""
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def process_waypoints(self, path):
        """Process waypoints, transform them into the TurtleBot's frame, and publish."""
        
        self.get_logger().info("Processing waypoints...")

        # if len(path) < 4:
        #     self.get_logger().warn("Path must have at least 4 waypoints for B-spline. Not processing.")
        #     return

        self.get_logger().info(f"Processing {len(path)} waypoints: {path}")

        num_waypoints = self.get_parameter('num_waypoints').get_parameter_value().integer_value
        smoothness = self.get_parameter('smoothness').get_parameter_value().double_value
        waypoints = self.create_b_spline_waypoints(path, num_waypoints, smoothness)

        if not waypoints:
            self.get_logger().warn("No waypoints generated!")
            return

        # Transform waypoints to TurtleBot's frame
        transformed_waypoints = self.transform_waypoints_to_turtlebot(waypoints)

        if not transformed_waypoints:
            self.get_logger().warn("No waypoints successfully transformed to TurtleBot frame!")
            return

        self.get_logger().info(f"✅ Transformed Waypoints: {transformed_waypoints}")

        # Publish waypoints for the TurtleBot
        self.publish_b_spline_path(transformed_waypoints)
        self.publish_waypoint_markers(transformed_waypoints)

        if self.latest_image is not None:
            self.annotate_and_publish_image(self.latest_image, transformed_waypoints)

    # def transform_waypoints_to_turtlebot(self, waypoints):
    #     """Transform waypoints from camera frame to TurtleBot's frame."""
    #     transformed_waypoints = []

    #     for x, y, z in waypoints:
    #         try:
    #             # Ensure TF buffer is initialized
    #             if not hasattr(self, "tf_buffer") or self.tf_buffer is None:
    #                 self.get_logger().error("❌ TF buffer not initialized!")
    #                 return []

    #             # Check if the transform is available BEFORE looking it up
    #             if not self.tf_buffer.can_transform("turtlebot_base", "camera_color_optical_frame", rclpy.time.Time()):
    #                 self.get_logger().error("🚨 Transform from camera → turtlebot_base is NOT available!")
    #                 return []

    #             # Lookup the transform ONCE before using it
    #             transform = self.tf_buffer.lookup_transform("turtlebot_base", "camera_color_optical_frame", rclpy.time.Time())

    #             # Debugging: Print the transform
    #             self.get_logger().info(f"📌 Using transform: {transform}")

    #             # Create PoseStamped object for transformation
    #             pose_stamped = PoseStamped()
    #             pose_stamped.header.frame_id = "camera_color_optical_frame"
    #             pose_stamped.header.stamp = transform.header.stamp

    #             pose_stamped.pose.position.x = float(x)
    #             pose_stamped.pose.position.y = float(y)
    #             pose_stamped.pose.position.z = float(z)
    #             pose_stamped.pose.orientation.w = 1.0  # Identity quaternion

    #             # 🚨 Perform the transformation
    #             try:
    #                 transformed_pose = do_transform_pose(pose_stamped, transform)

    #                 # 🛑 Force Debugging: Print what do_transform_pose() returned
    #                 if transformed_pose is None:
    #                     self.get_logger().error("🚨 do_transform_pose() returned None!")
    #                     continue  # Skip this waypoint

    #                 self.get_logger().info("\n🔍🔍🔍 TRANSFORMED POSE DEBUG 🔍🔍🔍")
    #                 self.get_logger().info(f"🟢 Transformed pose TYPE: {type(transformed_pose)}")
    #                 self.get_logger().info(f"🟢 Transformed pose CONTENT: {transformed_pose}")
    #                 self.get_logger().info("🔍🔍🔍 END DEBUG 🔍🔍🔍\n")

    #             except Exception as e:
    #                 self.get_logger().error(f"🚨 ERROR inside do_transform_pose(): {e}")
    #                 continue  # Skip this waypoint

    #             # Check if transformed_pose is valid
    #             if not isinstance(transformed_pose, PoseStamped):
    #                 self.get_logger().error(f"🚨 Transformed pose is not a PoseStamped! Got: {type(transformed_pose)}")
    #                 continue  # Skip this waypoint

    #             if not hasattr(transformed_pose.pose, "position"):
    #                 self.get_logger().error(f"🚨 Transformed pose is missing position attribute! {transformed_pose}")
    #                 continue  # Skip this waypoint

    #             # Extract transformed position
    #             transformed_x = transformed_pose.pose.position.x
    #             transformed_y = transformed_pose.pose.position.y
    #             transformed_z = transformed_pose.pose.position.z

    #             transformed_waypoints.append((transformed_x, transformed_y, transformed_z))

    #         except Exception as e:
    #             self.get_logger().error(f"🚨 Failed to transform waypoint: {e}")

    #     return transformed_waypoints


    def transform_waypoints_to_turtlebot(self, waypoints):
        """Manually transform waypoints from camera frame to TurtleBot's frame."""
        transformed_waypoints = []

        for x, y, z in waypoints:
            try:
                # Ensure TF buffer is initialized
                if not hasattr(self, "tf_buffer") or self.tf_buffer is None:
                    self.get_logger().error("❌ TF buffer not initialized!")
                    return []

                # Check if the transform is available BEFORE looking it up
                if not self.tf_buffer.can_transform("turtlebot_base", "camera_color_optical_frame", rclpy.time.Time()):
                    self.get_logger().error("🚨 Transform from camera → turtlebot_base is NOT available!")
                    return []

                # Lookup the transform ONCE before using it
                transform = self.tf_buffer.lookup_transform("turtlebot_base", "camera_color_optical_frame", rclpy.time.Time())

                # Debugging: Print the transform
                self.get_logger().info(f"📌 Using transform: {transform}")

                # Extract translation from transform
                trans = transform.transform.translation
                rot = transform.transform.rotation

                # Convert quaternion to rotation matrix
                quat = [rot.x, rot.y, rot.z, rot.w]
                rot_matrix = tf_transformations.quaternion_matrix(quat)

                # Convert point to homogeneous coordinates
                point = np.array([x, y, z, 1]).reshape(4, 1)

                # Apply transformation (Rotation + Translation)
                transformed_point = np.dot(rot_matrix, point)
                transformed_x = transformed_point[0, 0] + trans.x
                transformed_y = transformed_point[1, 0] + trans.y
                transformed_z = transformed_point[2, 0] + trans.z

                self.get_logger().info(f"✅ Manually Transformed: X={transformed_x:.3f}, Y={transformed_y:.3f}, Z={transformed_z:.3f}")

                transformed_waypoints.append((transformed_x, transformed_y, transformed_z))

            except Exception as e:
                self.get_logger().error(f"🚨 Failed to transform waypoint: {e}")

        return transformed_waypoints




    def path_callback(self, msg):
        """Callback for processing received path points."""
        
        # Ensure there are at least 4 points for B-spline
        if len(msg.data) < 8:  # (4 points × 2 coordinates per point)
            self.get_logger().warn("Path must have at least 4 points for B-spline. Received fewer points.")
            return
        
        # Ensure msg.data has an even number of elements (x, y pairs)
        if len(msg.data) % 2 != 0:
            self.get_logger().error("Received path data has an odd number of elements. Ignoring message.")
            return
        
        # Convert data into (x, y) pairs
        path = [(msg.data[i], msg.data[i + 1]) for i in range(0, len(msg.data), 2)]

        # Log received path points
        self.get_logger().info(f"Received {len(path)} waypoints: {path}")

        # Now process the path (e.g., generate smoothed waypoints, publish to TurtleBot)
        self.process_waypoints(path)


    def create_b_spline_waypoints(self, path, num_waypoints=50, smoothness=10):
        """Generate waypoints using a B-spline curve."""
        if len(path) < 4:  # B-splines require at least 4 points
            self.get_logger().warn("Path must have at least 4 points for B-spline.")
            return path

        # Extract x, y, and z coordinates from the path
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        z = [p[2] if len(p) == 3 else 0.0 for p in path]  # Default Z = 0 if missing

        # Fit a B-spline to the path
        try:
            tck, _ = splprep([x, y, z], s=smoothness)
        except Exception as e:
            self.get_logger().error(f"Failed to fit B-spline: {e}")
            return path

        # Generate evenly spaced waypoints along the B-spline
        u = np.linspace(0, 1, num_waypoints)
        x_smooth, y_smooth, z_smooth = splev(u, tck)

        # Convert to a list of (x, y, z) waypoints
        waypoints = [(xi, yi, zi) for xi, yi, zi in zip(x_smooth, y_smooth, z_smooth)]
        return waypoints


    def publish_b_spline_path(self, waypoints):
        """Publish transformed waypoints as a ROS Path message for TurtleBot."""
        
        if not waypoints:
            self.get_logger().warn("❌ No waypoints to publish!")
            return

        path_msg = Path()
        path_msg.header.frame_id = "turtlebot_base"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for i, (x, y, z) in enumerate(waypoints):
            pose = PoseStamped()
            pose.header.frame_id = "turtlebot_base"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        # Debugging Logs
        self.get_logger().info(f"📤 Publishing {len(waypoints)} waypoints to /waypoints: {waypoints}")

        num_subs = self.waypoint_publisher.get_subscription_count()
        self.get_logger().info(f"🔎 Subscribers on /waypoints: {num_subs}")

        if num_subs == 0:
            self.get_logger().warn("⚠️ No subscribers to /waypoints!")

        self.waypoint_publisher.publish(path_msg)




    def annotate_and_publish_image(self, image, waypoints):
        """Annotate the image with waypoints and publish it."""
        annotated_image = image.copy()
        height, width, _ = annotated_image.shape

        for x, y in waypoints:
            u, v = self.real_world_to_pixel(x, y)  # Convert real-world coords back to pixel
            cv2.circle(annotated_image, (u, v), 5, (0, 0, 255), -1)  # Red circle


        # Convert the annotated image back to a ROS Image message
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')

        # Publish the annotated image
        self.image_publisher.publish(annotated_msg)

    def real_world_to_pixel(self, x, y, z=1.0):
        """Convert real-world (X, Y, Z) coordinates to image pixel coordinates using camera intrinsics."""
        if self.fx is None or self.fy is None:
            return 0, 0  # If intrinsics aren't set, return (0,0)
        
        u = int((x * self.fx / z) + self.cx)
        v = int((y * self.fy / z) + self.cy)
    
        return u, v  # Pixel coordinates

    def world_to_image(self, x, y, width, height):
        """Convert real-world coordinates to image pixel coordinates."""
        u = int((x + 1.0) * (width / 2.0))  # Scale to image width
        v = int((y + 1.0) * (height / 2.0))  # Scale to image height
        return max(0, min(width - 1, u)), max(0, min(height - 1, v))  # Clamp to image size


    def publish_waypoint_markers(self, waypoints):
        """Publish individual waypoints as visualization markers in RViz."""
        marker_array = MarkerArray()

        for i, (x, y, z) in enumerate(waypoints):
            marker = Marker()
            marker.header.frame_id = "camera_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1  # Size of the marker
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0  # Red
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Fully visible

            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)





def main(args=None):
    rclpy.init(args=args)
    node = WaypointNode()
    rclpy.spin(node)
    rclpy.shutdown()
