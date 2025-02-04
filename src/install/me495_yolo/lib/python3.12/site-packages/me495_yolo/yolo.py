# import rclpy
# from ultralytics import YOLO
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np

# class YoloNode(Node):
#     """
#     Use Yolo to identify scene objects

#     Subscribes
#     ----------
#     image (sensor_msgs/msg/Image) - The input image

#     Publishes
#     ---------
#     new_image (sensor_msgs/msg/Image) - The image with the detections

#     Parameters
#     model (string) - The Yolo model to use: see docs.ultralytics.org for available values. Default is yolo11n.pt
#     """
#     def __init__(self):
#         super().__init__("pose")
#         self.bridge = CvBridge()
#         self.declare_parameter("model",
#                                value="yolo11n.pt")
#         self.model = YOLO(self.get_parameter("model").get_parameter_value().string_value)
#         self.create_subscription(Image, 'image', self.yolo_callback, 10)
#         self.pub = self.create_publisher(Image, 'new_image', 10)

 
 

#     def yolo_callback(self, image):
#         """Identify all the purple objects in the scene"""
#         # Convert to OpenCV
#         cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

#         # Convert to HSV and apply a mask for purple objects
#         hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
#         lower_purple = np.array([100, 150, 50])  # Adjust as needed
#         upper_purple = np.array([140, 255, 255])  # Adjust as needed
#         mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

#         # Filter the image to only show purple areas
#         purple_objects = cv2.bitwise_and(cv_image, cv_image, mask=mask)

#         # Run the YOLO model on the filtered image
#         results = self.model(purple_objects)

#         # Annotate the detections on the image
#         frame = results[0].plot()

#         # Convert back to ROS Image message
#         new_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

#         # Publish the new image
#         self.pub.publish(new_msg)


# def main():
#     rclpy.init()
#     node = YoloNode()
#     rclpy.spin(node)
#     rclpy.shutdown()

################################################################################################

# import rclpy
# from ultralytics import YOLO
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge


# class YoloNode(Node):
#     """
#     Use Yolo to identify scene objects

#     Subscribes
#     ----------
#     image (sensor_msgs/msg/Image) - The input image

#     Publishes
#     ---------
#     new_image (sensor_msgs/msg/Image) - The image with the detections

#     Parameters
#     model (string) - The Yolo model to use: see docs.ultralytics.org for available values. Default is yolo11n.pt
#     """
#     def __init__(self):
#         super().__init__("pose")
#         self.bridge = CvBridge()
#         self.declare_parameter("model",
#                                value="yolo11n.pt")
#         self.model = YOLO(self.get_parameter("model").get_parameter_value().string_value)
#         self.create_subscription(Image, 'image', self.yolo_callback, 10)
#         self.pub = self.create_publisher(Image, 'new_image', 10)

#     def yolo_callback(self, image):
#         """Identify all the objects in the scene"""
#         # Convert to OpenCV
#         cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
#         # Run the model
#         results = self.model(cv_image)
#         # Get the result and draw it on an OpenCV image
#         frame = results[0].plot()
#         new_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
#         # publish
#         self.pub.publish(new_msg)

# def main():
#     rclpy.init()
#     node = YoloNode()
#     rclpy.spin(node)
#     rclpy.shutdown()

#######################################################################################################################
# import rclpy
# from ultralytics import YOLO
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np


# class YoloNode(Node):
#     """
#     Use Yolo to identify and track specific colors (pink, orange, white, green, blue, yellow).

#     Subscribes
#     ----------
#     image (sensor_msgs/msg/Image) - The input image

#     Publishes
#     ---------
#     new_image (sensor_msgs/msg/Image) - The image with the detections

#     Parameters
#     ----------
#     model (string) - The Yolo model to use. Default is yolo11n.pt
#     """

#     def __init__(self):
#         super().__init__("pose")
#         self.bridge = CvBridge()
#         self.declare_parameter("model", value="yolo11n.pt")
#         self.model = YOLO(self.get_parameter("model").get_parameter_value().string_value)
#         self.create_subscription(Image, 'image', self.yolo_callback, 10)
#         self.pub = self.create_publisher(Image, 'new_image', 10)

#     def yolo_callback(self, image):
#         """Identify and track specific colors in the scene"""
#         # Convert ROS Image to OpenCV image
#         cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

#         # Convert to HSV for color segmentation
#         hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

#         # Define HSV ranges for colors
#         color_ranges = {
#             "pink": ((145, 50, 50), (165, 255, 255)),
#             "orange": ((5, 50, 50), (15, 255, 255)),
#             "white": ((0, 0, 200), (180, 30, 255)),
#             "green": ((35, 50, 50), (85, 255, 255)),
#             "blue": ((90, 50, 50), (130, 255, 255)),
#             "yellow": ((25, 50, 50), (35, 255, 255)),
#         }

#         # Create masks for each color and combine them
#         combined_mask = None
#         for color, (lower, upper) in color_ranges.items():
#             mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
#             combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

#         # Filter the original image to show only specified colors
#         filtered_image = cv2.bitwise_and(cv_image, cv_image, mask=combined_mask)

#         # Run YOLO model on the filtered image
#         results = self.model(filtered_image)

#         # Annotate the detections on the image
#         frame = results[0].plot()

#         # Convert back to ROS Image message
#         new_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

#         # Publish the annotated image
#         self.pub.publish(new_msg)


# def main():
#     rclpy.init()
#     node = YoloNode()
#     rclpy.spin(node)
#     rclpy.shutdown()


###################################################################################################################################
# import rclpy
# from ultralytics import YOLO
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np


# class YoloNode(Node):
#     """
#     Identify and track specific colors (pink, orange, white, green, blue, yellow).

#     Subscribes
#     ----------
#     image (sensor_msgs/msg/Image) - The input image

#     Publishes
#     ---------
#     new_image (sensor_msgs/msg/Image) - The image with bounding boxes around detected colors
#     """

#     def __init__(self):
#         super().__init__("pose")
#         self.bridge = CvBridge()
#         self.create_subscription(Image, 'image', self.yolo_callback, 10)
#         self.pub = self.create_publisher(Image, 'new_image', 10)

#         #store path 
#         self.path = []

#         #define the c

#     def yolo_callback(self, image):
#         """Identify and track specific colors in the scene."""
#         # Convert ROS Image to OpenCV image
#         cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

#         # Convert to HSV for color segmentation
#         hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

#         # Define HSV ranges for colors
#         color_ranges = {
#             "pink": ((145, 50, 50), (165, 255, 255)),
#             "orange": ((5, 50, 50), (15, 255, 255)),
#             "white": ((0, 0, 200), (180, 30, 255)),
#             "green": ((35, 50, 50), (85, 255, 255)),
#             "blue": ((90, 50, 50), (130, 255, 255)),
#             "yellow": ((25, 50, 50), (35, 255, 255)),
#         }

#         # Create a copy of the image for drawing annotations
#         annotated_image = cv_image.copy()

#         # Loop through each color and detect regions
#         for color_name, (lower, upper) in color_ranges.items():
#             # Create a mask for the current color
#             mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

#             # Find contours of the detected regions
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # calculate centroid and label the detected colors
#             for contour in contours:
#                 if cv2.contourArea(contour) > 500:  # Filter small regions by area
#                     # Calculate the centroid
#                     moments = cv2.moments(contour)
#                     if moments["m00"] != 0:  # Avoid division by zero
#                         cX = int(moments["m10"] / moments["m00"])
#                         cY = int(moments["m01"] / moments["m00"])
#                         # Annotate the centroid on the image
#                         cv2.circle(annotated_image, (cX, cY), 5, (0, 255, 0), -1)
#                         cv2.putText(
#                             annotated_image,
#                             f"{color_name} ({cX}, {cY})",
#                             (cX + 10, cY - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.5,
#                             (0, 255, 0),
#                             2,
#                         )

#         # Convert the annotated image back to ROS Image message
#         new_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')

#         # Publish the annotated image
#         self.pub.publish(new_msg)


# def main():
#     rclpy.init()
#     node = YoloNode()
#     rclpy.spin(node)
#     rclpy.shutdown()

########################################################################## reallly good color tracking
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import time
# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge
# from scipy.interpolate import splprep, splev
# from std_msgs.msg import Float32MultiArray
# from nav_msgs.msg import Path
# from scipy.interpolate import splprep, splev
# from visualization_msgs.msg import Marker, MarkerArray  # Import this


# class YoloNode(Node):
#     """
#     Identify and track a specific color (e.g., yellow),
#     and create a path based on the movement of the detected color.

#     Subscribes
#     ----------
#     image (sensor_msgs/msg/Image) - The input image

#     Publishes
#     ---------
#     new_image (sensor_msgs/msg/Image) - Annotated image with the path
#     """

#     def __init__(self):
#         super().__init__("color_tracker")
#         self.bridge = CvBridge()
#         self.create_subscription(Image, 'image', self.yolo_callback, 10)
#         #self.create_subscription(Path, 'waypoints', self.waypoint_callback, 10)
#         self.create_subscription(MarkerArray, 'waypoint_markers', self.marker_callback, 10)
#         #self.create_subscription(Image, 'depth', self.depth_callback, 10)  # Depth image
#         self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
#         self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

#         self.pub = self.create_publisher(Image, 'new_image', 10)
#         self.path_publisher = self.create_publisher(Float32MultiArray, 'path_points', 10)


#         # Store the path as a list of centroids
#         self.path = []
#         self.waypoints = []
#         self.waypoint_markers = []  # Initialize an empty list

#         self.depth_image = None  # Store the latest depth frame
#         self.fx, self.fy, self.cx, self.cy = None, None, None, None  # Camera intrinsics


#         # Define the color to track (yellow)
#         #self.color_name = "yellow"
#         #self.color_range = ((25, 50, 50), (35, 255, 255))  # HSV range for yellow

#         self.color_name = "pink"
#         self.color_range = ((145, 50, 50), (165, 255, 255))  # HSV range for pink

#         # self.color_name = "orange"
#         # self.color_range =  ((5, 50, 50), (15, 255, 255))

#         # self.color_name = "white"
#         # self.color_range =  ((0, 0, 200), (180, 30, 255))

#         # self.color_name = "green"
#         # self.color_range =  ((0, 0, 200), (180, 30, 255))

#         # self.color_name = "blue"
#         # self.color_range =  ((90, 50, 50), (130, 255, 255))

#         # Timer for path recording
#         self.start_time = None
#         #self.time_limit = 15 # end path recordign after 15 seconds 

#     def camera_info_callback(self, msg):
#         """Extract camera intrinsics from CameraInfo topic."""
#         self.fx = msg.k[0]  # Focal length x
#         self.fy = msg.k[4]  # Focal length y
#         self.cx = msg.k[2]  # Optical center x
#         self.cy = msg.k[5]  # Optical center y
#         self.get_logger().info(f"✅ ✅ Camera Info Received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

#     def depth_callback(self, msg):
#         """Store the latest depth frame."""
#         self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
#         self.get_logger().info(f"✅✅ Depth Image Received: First Depth Value = {self.depth_image[240, 320]}")  # Sample at center pixel

#     def waypoint_callback(self, msg):
#         """Callback to receive waypoints from the waypoint generation node."""
#         self.waypoints = [(int(pose.pose.position.x), int(pose.pose.position.y)) for pose in msg.poses]


#     def yolo_callback(self, image):
#         """Track the specified color and create a smoothed path."""


#         #logging to make sure i have camera intrisnics to calculate depth 
#         if self.depth_image is None or self.fx is None:
#             self.get_logger().warn("Waiting for depth image and camera intrinsics!")
#             return
        
#         # Convert ROS Image to OpenCV image
#         cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

#         # Convert to HSV for color segmentation
#         hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

#         # Create a mask for the specified color
#         mask = cv2.inRange(hsv_image, np.array(self.color_range[0]), np.array(self.color_range[1]))

#         # Find contours of the detected regions
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Annotate the image and update the path
#         annotated_image = cv_image.copy()
#         merged_contour = None

#         # Merge nearby contours --> this helps with the blank band on the center of my color bands 
#         for contour in contours:
#             if cv2.contourArea(contour) > 500:  # Filter small regions by area
#                 if merged_contour is None:
#                     merged_contour = contour
#                 else:
#                     merged_contour = np.vstack((merged_contour, contour))

#         # If a merged contour exists, calculate its centroid
#         if merged_contour is not None:
#             moments = cv2.moments(merged_contour)
#             if moments["m00"] != 0:  # Avoid division by zero
#                 cX = int(moments["m10"] / moments["m00"])
#                 cY = int(moments["m01"] / moments["m00"])
                
#                 # Convert pixel coordinates to real-world (X, Y, Z)
#                 real_world_coords = self.pixel_to_world(cX, cY)
#                 if real_world_coords is not None:
#                     X, Y, Z = real_world_coords
#                     self.get_logger().info(f"Real-World Coordinates: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

#                     # Add to path only if it moves significantly
#                     if len(self.path) == 0 or (abs(X - self.path[-1][0]) > 0.05 or abs(Y - self.path[-1][1]) > 0.05):
#                         self.path.append((X, Y, Z))

#                     # Draw the detected point
#                     cv2.circle(cv_image, (cX, cY), 5, (0, 255, 0), -1)
#                     cv2.putText(cv_image, f"{X:.2f}, {Y:.2f}, {Z:.2f}",
#                                 (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 # # Add the centroid to the path
#                 # # self.path.append((cX, cY))
#                 # # Add the centroid to the path ONLY if it moves significantly
#                 # if len(self.path) == 0 or (abs(cX - self.path[-1][0]) > 5 or abs(cY - self.path[-1][1]) > 5):
#                 #     self.path.append((cX, cY))

#                 # # Draw the current centroid
#                 # cv2.circle(annotated_image, (cX, cY), 5, (0, 255, 0), -1)
#                 # cv2.putText(
#                 #     annotated_image,
#                 #     f"{self.color_name} ({cX}, {cY})",
#                 #     (cX + 10, cY - 10),
#                 #     cv2.FONT_HERSHEY_SIMPLEX,
#                 #     0.5,
#                 #     (0, 255, 0),
#                 #     2,
#                 # )


#         # Draw the smoothed path in pixel coordinates 
#         if len(self.path) > 1:
#             for i in range(1, len(self.path)):
#                 cv2.line(annotated_image, self.path[i - 1], self.path[i], (0, 255, 0), 2)

#         # Publish the raw path
#         path_msg = Float32MultiArray()
#         path_msg.data = [coord for point in self.path for coord in point]
#         self.path_publisher.publish(path_msg)

#         # Draw waypoints as red dots from waypoint_markers
#         for marker in self.waypoint_markers:
#             x = int(marker.pose.position.x)
#             y = int(marker.pose.position.y)
#             cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)  # Red dot



#         # Convert the annotated image back to ROS Image message
#         new_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
#         # Publish the annotated image
#         self.pub.publish(new_msg)

#     def marker_callback(self, msg):
#         """Callback to receive waypoint markers and store them for visualization."""
#         self.get_logger().info(f"Received {len(msg.markers)} markers")
#         self.waypoint_markers = msg.markers


# def main():
#     rclpy.init()
#     node = YoloNode()
#     rclpy.spin(node)
#     rclpy.shutdown()

###################################################### x y z
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped


class YoloNode(Node):
    def __init__(self):
        super().__init__("color_tracker")
        self.bridge = CvBridge()

        # Subscribe to RGB image, depth image, and camera info
        self.create_subscription(Image, 'image', self.yolo_callback, 10)
        # self.create_subscription(Image, 'depth', self.depth_callback, 10)
        # self.create_subscription(CameraInfo, 'camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_timer = self.create_timer(0.1, self.broadcast_static_tf)  # Publish every 100ms


        self.create_subscription(MarkerArray, 'waypoint_markers', self.marker_callback, 10)

        self.last_valid_dancer_pose = None  # Store last valid dancer position


        # Publishers
        self.pub = self.create_publisher(Image, 'new_image', 10)
        self.path_publisher = self.create_publisher(Float32MultiArray, 'path_points', 10)

        # Color Tracking Settings (Pink marker)
        # Color Tracking Settings
        self.pink_range = ((145, 50, 50), (165, 255, 255))  # HSV range for pink (dancer)
        self.blue_range = ((90, 50, 50), (130, 255, 255))   # HSV range for blue (TurtleBot marker)

        # self.color_name = "pink"
        # self.color_range = ((145, 50, 50), (165, 255, 255))  # HSV range for pink

        # # self.color_name = "blue"
        # # self.color_range =  ((90, 50, 50), (130, 255, 255))


        self.depth_image = None  # Store the latest depth frame
        self.fx, self.fy, self.cx, self.cy = None, None, None, None  # Camera intrinsics

        # Path storage
        self.path = []
        self.waypoint_markers = []
        self.dancer_path = []
        self.turtlebot_path = []

    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo topic."""
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]  # Focal lengths
            self.cx, self.cy = msg.k[2], msg.k[5]  # Optical center
            self.get_logger().info(f"✅ Camera Info Received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def depth_callback(self, msg):
        """Store the latest depth frame and print a sample value."""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.get_logger().info(f"✅ Depth Image Received: First Depth Value = {self.depth_image[240, 320]}")  # Sample at center pixel


    # def yolo_callback(self, image):
    #     """Track color, merge contours, and convert to real-world coordinates."""
        
    #     if self.depth_image is None or self.fx is None:
    #         self.get_logger().warn("Waiting for depth image and camera intrinsics!")
    #         return

    #     # Convert ROS Image to OpenCV
    #     cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    #     hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    #     # Create a mask for the selected color
    #     mask = cv2.inRange(hsv_image, np.array(self.color_range[0]), np.array(self.color_range[1]))

    #     # Find and merge contours
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     merged_contour = None

    #     for contour in contours:
    #         if cv2.contourArea(contour) > 500:
    #             if merged_contour is None:
    #                 merged_contour = contour
    #             else:
    #                 merged_contour = np.vstack((merged_contour, contour))

    #     if merged_contour is not None:
    #         moments = cv2.moments(merged_contour)
    #         if moments["m00"] != 0:
    #             cX = int(moments["m10"] / moments["m00"])
    #             cY = int(moments["m01"] / moments["m00"])

    #             # Convert pixel coordinates to real-world (X, Y, Z)
    #             real_world_coords = self.pixel_to_world(cX, cY)
    #             if real_world_coords is not None:
    #                 X, Y, Z = real_world_coords
    #                 self.get_logger().info(f"Real-World Coordinates: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

    #                 # Broadcast TF transform from camera to dancer
    #                 self.broadcast_camera_to_dancer(X, Y, Z)

    #                 # Add to path only if it moves significantly
    #                 if len(self.path) == 0 or (abs(X - self.path[-1][0]) > 0.05 or abs(Y - self.path[-1][1]) > 0.05):
    #                     self.path.append((X, Y, Z))

    #                 # Draw the detected point
    #                 cv2.circle(cv_image, (cX, cY), 5, (0, 255, 0), -1)
    #                 cv2.putText(cv_image, f"{X:.2f}, {Y:.2f}, {Z:.2f}",
    #                             (cX + 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     # Publish the real-world path
    #     path_msg = Float32MultiArray()
    #     path_msg.data = [coord for point in self.path for coord in point]
    #     self.path_publisher.publish(path_msg)

    #     # Publish the annotated image
    #     new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    #     self.pub.publish(new_msg)

    #     # Draw the smoothed path
    #     if len(self.path) > 1:
    #         for i in range(1, len(self.path)):
    #             p1 = self.image_to_pixel(self.path[i - 1])
    #             p2 = self.image_to_pixel(self.path[i])
    #             cv2.line(cv_image, p1, p2, (0, 255, 0), 2)

    #     # Draw waypoints as red dots
    #     for marker in self.waypoint_markers:
    #         x = int(marker.pose.position.x)
    #         y = int(marker.pose.position.y)
    #         cv2.circle(cv_image, (x, y), 5, (0, 0, 255), -1)  # Red dot for waypoints

    #     # Publish the real-world waypoints
    #     path_msg = Float32MultiArray()
    #     path_msg.data = [coord for point in self.path for coord in point]
    #     self.path_publisher.publish(path_msg)

    #     # Publish the annotated image
    #     new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    #     self.pub.publish(new_msg)

    # def yolo_callback(self, image):
    #     """Track pink (dancer) and blue (TurtleBot) markers, draw paths, and publish TF transforms."""

    #     if self.depth_image is None or self.fx is None:
    #         self.get_logger().warn("Waiting for depth image and camera intrinsics!")
    #         return

    #     # Convert ROS Image to OpenCV
    #     cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    #     hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    #     ###  Detect Pink (Dancer)
    #     pink_mask = cv2.inRange(hsv_image, np.array(self.pink_range[0]), np.array(self.pink_range[1]))
    #     pink_cX, pink_cY = self.find_contour_center(pink_mask)
        
    #     if pink_cX is not None and pink_cY is not None:
    #         pink_world_coords = self.pixel_to_world(pink_cX, pink_cY)
    #         if pink_world_coords:
    #             Xp, Yp, Zp = pink_world_coords
    #             self.broadcast_camera_to_dancer(Xp, Yp, Zp)
    #             self.get_logger().info(f" Dancer Position: X={Xp:.3f}, Y={Yp:.3f}, Z={Zp:.3f}")

    #             #  Draw pink centroid and label it
    #             cv2.circle(cv_image, (pink_cX, pink_cY), 5, (255, 0, 255), -1)  # Pink dot for dancer
    #             cv2.putText(cv_image, "Dancer", (pink_cX + 10, pink_cY - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    #             #  Store dancer position for path tracking
    #             self.dancer_path.append((pink_cX, pink_cY))

    #     ###  Detect Blue (TurtleBot)
    #     blue_mask = cv2.inRange(hsv_image, np.array(self.blue_range[0]), np.array(self.blue_range[1]))
    #     blue_cX, blue_cY = self.find_contour_center(blue_mask)
        
    #     if blue_cX is not None and blue_cY is not None:
    #         blue_world_coords = self.pixel_to_world(blue_cX, blue_cY)
    #         if blue_world_coords:
    #             Xt, Yt, Zt = blue_world_coords
    #             self.broadcast_camera_to_turtlebot(Xt, Yt, Zt)
    #             self.get_logger().info(f" TurtleBot Position: X={Xt:.3f}, Y={Yt:.3f}, Z={Zt:.3f}")

    #             #  Draw blue centroid and label it
    #             cv2.circle(cv_image, (blue_cX, blue_cY), 5, (255, 0, 0), -1)  # Blue dot for TurtleBot
    #             cv2.putText(cv_image, "TurtleBot", (blue_cX + 10, blue_cY - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    #             #  Store TurtleBot position for path tracking
    #             self.turtlebot_path.append((blue_cX, blue_cY))

    #     # Draw paths for dancer and TurtleBot
    #     self.draw_paths(cv_image)

    #     # Publish the annotated image with centroids, labels, and paths
    #     new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    #     self.pub.publish(new_msg)

    # def yolo_callback(self, image):
    #     """Track pink (dancer) and blue (TurtleBot) markers, merge contours, and publish real-world path."""

    #     if self.depth_image is None or self.fx is None:
    #         self.get_logger().warn("Waiting for depth image and camera intrinsics!")
    #         return

    #     # Convert ROS Image to OpenCV
    #     cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
    #     hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    #     ###  Detect and Merge Pink Contours (Dancer)
    #     pink_mask = cv2.inRange(hsv_image, np.array(self.pink_range[0]), np.array(self.pink_range[1]))
    #     contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     merged_contour = None
    #     for contour in contours:
    #         if cv2.contourArea(contour) > 500:  # Ignore small noise
    #             if merged_contour is None:
    #                 merged_contour = contour
    #             else:
    #                 merged_contour = np.vstack((merged_contour, contour))

    #     if merged_contour is not None:
    #         moments = cv2.moments(merged_contour)
    #         if moments["m00"] != 0:
    #             pink_cX = int(moments["m10"] / moments["m00"])
    #             pink_cY = int(moments["m01"] / moments["m00"])

    #             # Convert pixel coordinates to real-world (X, Y, Z)
    #             pink_world_coords = self.pixel_to_world(pink_cX, pink_cY)
    #             if pink_world_coords:
    #                 Xp, Yp, Zp = pink_world_coords
    #                 self.broadcast_camera_to_dancer(Xp, Yp, Zp)
    #                 self.get_logger().info(f"✅ Dancer Position: X={Xp:.3f}, Y={Yp:.3f}, Z={Zp:.3f}")

    #                 # Add to path only if it moves significantly
    #                 if len(self.dancer_path) == 0 or (abs(Xp - self.dancer_path[-1][0]) > 0.05 or abs(Yp - self.dancer_path[-1][1]) > 0.05):
    #                     self.dancer_path.append((Xp, Yp, Zp))
    #                     self.get_logger().info(f"🟢 Added Path Point: ({Xp:.3f}, {Yp:.3f}, {Zp:.3f})")

    #                 # Draw the detected centroid
    #                 cv2.circle(cv_image, (pink_cX, pink_cY), 5, (255, 0, 255), -1)  # Pink dot
    #                 cv2.putText(cv_image, "Dancer", (pink_cX + 10, pink_cY - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    #     # Draw green path from real-world coordinates
    #     if len(self.dancer_path) > 1:
    #         for i in range(1, len(self.dancer_path)):
    #             p1 = self.image_to_pixel(self.dancer_path[i - 1])
    #             p2 = self.image_to_pixel(self.dancer_path[i])
    #             cv2.line(cv_image, p1, p2, (0, 255, 0), 2)

    #     # Publish the real-world path
    #     if len(self.dancer_path) > 0:
    #         path_msg = Float32MultiArray()
    #         path_msg.data = [coord for point in self.dancer_path for coord in point]
    #         self.get_logger().info(f"📤 Publishing Path: {path_msg.data}")
    #         self.path_publisher.publish(path_msg)

    #     if len(self.dancer_path) == 0 or (abs(X - self.path[-1][0]) > 0.05 or abs(Y - self.dancer_path[-1][1]) > 0.05):
    #         self.path.append((X, Y, Z))  # Always store (x, y, z)


    #     # Publish the annotated image with centroids, labels, and paths
    #     new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
    #     self.pub.publish(new_msg)
    def yolo_callback(self, image):
        """Track pink (dancer) and blue (TurtleBot) markers, merge contours, update paths, and publish real-world coordinates."""

        if self.depth_image is None or self.fx is None:
            self.get_logger().warn("Waiting for depth image and camera intrinsics!")
            return

        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        ### **Step 1: Detect & Merge Pink Contours (Dancer)**
        pink_mask = cv2.inRange(hsv_image, np.array(self.pink_range[0]), np.array(self.pink_range[1]))
        pink_contour = self.merge_contours(pink_mask)

        if pink_contour is not None:
            moments = cv2.moments(pink_contour)
            if moments["m00"] != 0:
                pink_cX = int(moments["m10"] / moments["m00"])
                pink_cY = int(moments["m01"] / moments["m00"])

                ### **Step 2: Convert Pink Centroid to Real-World Coordinates**
                pink_world_coords = self.pixel_to_world(pink_cX, pink_cY)

                if pink_world_coords:
                    Xp, Yp, Zp = pink_world_coords  # Ensure these are always defined
                    self.broadcast_camera_to_dancer(Xp, Yp, Zp)
                    self.get_logger().info(f"🩷 Dancer Position: X={Xp:.3f}, Y={Yp:.3f}, Z={Zp:.3f}")

                    ### **Step 3: Add to Path Only if Movement is Significant**
                    if len(self.dancer_path) == 0 or (
                        abs(Xp - self.dancer_path[-1][0]) > 0.05 or abs(Yp - self.dancer_path[-1][1]) > 0.05
                    ):
                        self.dancer_path.append((Xp, Yp, Zp))  # Store (x, y, z)

                    ### **Step 4: Draw Pink Centroid**
                    cv2.circle(cv_image, (pink_cX, pink_cY), 5, (255, 0, 255), -1)
                    cv2.putText(cv_image, "Dancer", (pink_cX + 10, pink_cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        ### **Step 5: Detect & Merge Blue Contours (TurtleBot)**
        blue_mask = cv2.inRange(hsv_image, np.array(self.blue_range[0]), np.array(self.blue_range[1]))
        blue_contour = self.merge_contours(blue_mask)

        if blue_contour is not None:
            moments = cv2.moments(blue_contour)
            if moments["m00"] != 0:
                blue_cX = int(moments["m10"] / moments["m00"])
                blue_cY = int(moments["m01"] / moments["m00"])

                ### **Step 6: Convert Blue Centroid to Real-World Coordinates**
                blue_world_coords = self.pixel_to_world(blue_cX, blue_cY)

                if blue_world_coords:
                    Xt, Yt, Zt = blue_world_coords  # Ensure these are always defined
                    self.broadcast_camera_to_turtlebot(Xt, Yt, Zt)
                    self.get_logger().info(f"💙 TurtleBot Position: X={Xt:.3f}, Y={Yt:.3f}, Z={Zt:.3f}")

                    ### **Step 7: Add to TurtleBot Path Only if Movement is Significant**
                    if len(self.turtlebot_path) == 0 or (
                        abs(Xt - self.turtlebot_path[-1][0]) > 0.05 or abs(Yt - self.turtlebot_path[-1][1]) > 0.05
                    ):
                        self.turtlebot_path.append((Xt, Yt, Zt))  # Store (x, y, z)

                    ### **Step 8: Draw Blue Centroid**
                    cv2.circle(cv_image, (blue_cX, blue_cY), 5, (255, 0, 0), -1)
                    cv2.putText(cv_image, "TurtleBot", (blue_cX + 10, blue_cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        ### **Step 9: Publish the Real-World Paths**
        if len(self.dancer_path) > 0 and all(len(p) == 3 for p in self.dancer_path):
            path_msg = Float32MultiArray()
            path_msg.data = [coord for point in self.dancer_path for coord in point]
            self.get_logger().info(f"📤 Publishing Dancer Path: {path_msg.data}")
            self.path_publisher.publish(path_msg)

        if len(self.turtlebot_path) > 0 and all(len(p) == 3 for p in self.turtlebot_path):
            path_msg = Float32MultiArray()
            path_msg.data = [coord for point in self.turtlebot_path for coord in point]
            self.get_logger().info(f"📤 Publishing TurtleBot Path: {path_msg.data}")
            self.path_publisher.publish(path_msg)

        ### **Step 10: Publish the Annotated Image**
        new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.pub.publish(new_msg)



    def merge_contours(self, mask):
        """Find and merge nearby contours to avoid gaps in tracking."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_contour = None

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small areas
                if merged_contour is None:
                    merged_contour = contour
                else:
                    merged_contour = np.vstack((merged_contour, contour))

        return merged_contour



    def draw_paths(self, image):
        """Draw green paths for the dancer and TurtleBot."""
        
        # Draw the dancer's path
        if len(self.dancer_path) > 1:
            for i in range(1, len(self.dancer_path)):
                cv2.line(image, self.dancer_path[i - 1], self.dancer_path[i], (0, 255, 0), 2)

        # # Draw the TurtleBot's path
        # if len(self.turtlebot_path) > 1:
        #     for i in range(1, len(self.turtlebot_path)):
        #         cv2.line(image, self.turtlebot_path[i - 1], self.turtlebot_path[i], (0, 255, 0), 2)



    def find_contour_center(self, mask):
        """Find the largest contour in the mask and return its centroid (cX, cY)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                return cX, cY
        return None, None

    def image_to_pixel(self, world_point):
        """Convert real-world (X, Y, Z) to pixel coordinates for visualization."""
        x, y, z = world_point
        u = int((x * self.fx / z) + self.cx)
        v = int((y * self.fy / z) + self.cy)
        return (u, v)
    

    def pixel_to_world(self, u, v):
        """Convert pixel coordinates (u, v) and depth to real-world coordinates."""
        if self.depth_image is None or self.fx is None:
            return None  # Wait until we have depth and camera intrinsics

        # Ensure pixel coordinates are within valid image bounds
        h, w = self.depth_image.shape  # Get depth image size
        u = np.clip(u, 0, w - 1)  # Clamp x-coordinate
        v = np.clip(v, 0, h - 1)  # Clamp y-coordinate

        # Get depth at the pixel
        depth = self.depth_image[v, u] * 0.001  # Convert mm to meters
        if depth <= 0:  # Invalid depth
            return None

        # Convert to real-world coordinates using intrinsics
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth  # Depth is the Z coordinate

        return X, Y, Z
    
    def broadcast_camera_to_dancer(self, X, Y, Z):
        """Broadcast transformation from camera frame to detected pink object (dancer)."""
        try:
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_color_optical_frame'  # Camera's frame
            t.child_frame_id = 'dancer_pink_object'  # New frame for dancer

            t.transform.translation.x = float(X)
            t.transform.translation.y = float(Y)
            t.transform.translation.z = float(Z)

            # Set rotation to identity quaternion (no rotation assumed)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            # Broadcast transformation
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().error(f"Failed to publish TF transform: {e}")

    def broadcast_camera_to_turtlebot(self, X, Y, Z):
        """Broadcast transformation from camera frame to TurtleBot's blue marker."""
        try:
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_color_optical_frame'  # Camera's reference frame
            t.child_frame_id = 'turtlebot_base'  # Name of TurtleBot's frame

            t.transform.translation.x = float(X)
            t.transform.translation.y = float(Y)
            t.transform.translation.z = float(Z)

            # Identity rotation (assuming no rotation)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().error(f"Failed to publish TurtleBot TF transform: {e}")

    # def compute_dancer_in_turtlebot_frame(self):
    #     """Compute dancer's position in the TurtleBot's reference frame using TF."""
    #     try:
    #         tf_buffer = tf2_ros.Buffer()
    #         listener = tf2_ros.TransformListener(tf_buffer, self)

    #         transform = tf_buffer.lookup_transform('turtlebot_base', 'dancer_pink_object', rclpy.time.Time())

    #         x = transform.transform.translation.x
    #         y = transform.transform.translation.y
    #         z = transform.transform.translation.z

    #         self.get_logger().info(f" Dancer in TurtleBot Frame: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    #         return x, y, z
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to compute dancer in TurtleBot frame: {e}")
    #         return None
    

    def compute_dancer_in_turtlebot_frame(self):
        """Compute dancer's position in the TurtleBot's reference frame using TF."""
        try:
            tf_buffer = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(tf_buffer, self)

            transform = tf_buffer.lookup_transform('turtlebot_base', 'dancer_pink_object', rclpy.time.Time())

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            self.last_valid_dancer_pose = (x, y, z)  # Store last known good pose
            self.get_logger().info(f"✅ Dancer in TurtleBot Frame: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

            return x, y, z

        except Exception as e:
            self.get_logger().warn(f"⚠️ Failed to compute dancer in TurtleBot frame: {e}")

            # If no new detection, return the last known valid position
            if self.last_valid_dancer_pose:
                self.get_logger().info("♻️ Using last known dancer pose instead.")
                return self.last_valid_dancer_pose

            return None  # No valid data yet

    
    def broadcast_static_tf(self):
        """Continuously publish the transform between camera and TurtleBot."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_color_optical_frame'  # Camera frame
        t.child_frame_id = 'turtlebot_base'  # TurtleBot frame

        # Use a default fixed transform (adjust as needed)
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.2
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0  # Identity quaternion

        self.tf_broadcaster.sendTransform(t)





    def marker_callback(self, msg):
        """Receive waypoint markers for visualization.""" 
        self.get_logger().info(f"Received {len(msg.markers)} markers")
        self.waypoint_markers = msg.markers

def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()


###################################################

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CameraInfo
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# from std_msgs.msg import Float32MultiArray
# from visualization_msgs.msg import MarkerArray


# class YoloNode(Node):
#     def __init__(self):
#         super().__init__("color_tracker")
#         self.bridge = CvBridge()

#         # Subscriptions
#         self.create_subscription(Image, 'image', self.yolo_callback, 10)
#         self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
#         self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
#         self.create_subscription(MarkerArray, 'waypoint_markers', self.marker_callback, 10)

#         # Publishers
#         self.pub = self.create_publisher(Image, 'new_image', 10)
#         self.path_publisher = self.create_publisher(Float32MultiArray, 'path_points', 10)

#         # Path storage
#         self.path = []  # Now storing real-world coordinates (X, Y, Z)
#         self.waypoint_markers = []  # Waypoints received

#         # Camera intrinsics
#         self.depth_image = None
#         self.fx, self.fy, self.cx, self.cy = None, None, None, None

#         # Color to track (pink)
#         self.color_name = "pink"
#         self.color_range = ((145, 50, 50), (165, 255, 255))  # HSV range for pink

#     def camera_info_callback(self, msg):
#         """Extract camera intrinsics."""
#         self.fx, self.fy = msg.k[0], msg.k[4]
#         self.cx, self.cy = msg.k[2], msg.k[5]
#         self.get_logger().info(f"✅ Camera Info: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

#     def depth_callback(self, msg):
#         """Store the latest depth frame."""
#         self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

#     def yolo_callback(self, image):
#         """Track the color and store path in real-world coordinates."""
#         if self.depth_image is None or self.fx is None:
#             self.get_logger().warn("Waiting for depth image and camera intrinsics!")
#             return

#         # Convert ROS Image to OpenCV
#         cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
#         hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

#         # Create a mask for the specified color
#         mask = cv2.inRange(hsv_image, np.array(self.color_range[0]), np.array(self.color_range[1]))

#         # Find and merge contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         merged_contour = None

#         for contour in contours:
#             if cv2.contourArea(contour) > 500:  # Ignore small noise
#                 merged_contour = np.vstack((merged_contour, contour)) if merged_contour is not None else contour

#         # Process merged contour
#         if merged_contour is not None:
#             moments = cv2.moments(merged_contour)
#             if moments["m00"] != 0:
#                 cX = int(moments["m10"] / moments["m00"])
#                 cY = int(moments["m01"] / moments["m00"])

#                 # Convert pixel coordinates to real-world (X, Y, Z)
#                 real_world_coords = self.pixel_to_world(cX, cY)
#                 if real_world_coords is not None:
#                     X, Y, Z = real_world_coords
#                     self.get_logger().info(f"Real-World Coordinates: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

#                     # Store path in real-world coordinates
#                     if len(self.path) == 0 or (abs(X - self.path[-1][0]) > 0.05 or abs(Y - self.path[-1][1]) > 0.05):
#                         self.path.append((X, Y, Z))

#         # Publish the path
#         path_msg = Float32MultiArray()
#         path_msg.data = [coord for point in self.path for coord in point]
#         self.path_publisher.publish(path_msg)

#         # Create an annotated image for visualization
#         self.visualize_tracked_path(cv_image)

#     def visualize_tracked_path(self, cv_image):
#         """Draw the green path and red waypoints in pixel coordinates on `new_image`."""
#         annotated_image = cv_image.copy()

#         # Draw the tracked path (green line)
#         if len(self.path) > 1:
#             for i in range(1, len(self.path)):
#                 p1 = self.image_to_pixel(self.path[i - 1])
#                 p2 = self.image_to_pixel(self.path[i])
#                 cv2.line(annotated_image, p1, p2, (0, 255, 0), 2)  # Green path

#         # Draw waypoints (red dots)
#         for marker in self.waypoint_markers:
#             real_x = marker.pose.position.x
#             real_y = marker.pose.position.y
#             real_z = marker.pose.position.z  # If using Z

#             # Convert waypoints back to pixel coordinates
#             pixel_coords = self.image_to_pixel((real_x, real_y, real_z))
#             cv2.circle(annotated_image, pixel_coords, 5, (0, 0, 255), -1)  # Red dot

#         # Publish the annotated image
#         new_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
#         self.pub.publish(new_msg)

#     def image_to_pixel(self, world_point):
#         """Convert real-world (X, Y, Z) to pixel coordinates."""
#         x, y, z = world_point
#         u = int((x * self.fx / z) + self.cx)
#         v = int((y * self.fy / z) + self.cy)
#         return (u, v)

#     def pixel_to_world(self, u, v):
#         """Convert pixel coordinates to real-world (X, Y, Z)."""
#         if self.depth_image is None or self.fx is None:
#             return None

#         # Ensure pixel coordinates are within bounds
#         h, w = self.depth_image.shape
#         u = np.clip(u, 0, w - 1)
#         v = np.clip(v, 0, h - 1)

#         # Get depth at pixel
#         depth = self.depth_image[v, u] * 0.001  # Convert mm to meters
#         if depth <= 0:  # Invalid depth
#             return None

#         # Convert to real-world coordinates
#         X = (u - self.cx) * depth / self.fx
#         Y = (v - self.cy) * depth / self.fy
#         Z = depth

#         return X, Y, Z

#     def marker_callback(self, msg):
#         """Receive waypoint markers for visualization."""
#         self.get_logger().info(f"Received {len(msg.markers)} markers")
#         self.waypoint_markers = msg.markers


# def main():
#     rclpy.init()
#     node = YoloNode()
#     rclpy.spin(node)
#     rclpy.shutdown()
