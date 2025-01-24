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


####################################################################################################################################
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

############################################################################
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from scipy.interpolate import splprep, splev
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path



class YoloNode(Node):
    """
    Identify and track a specific color (e.g., yellow),
    and create a path based on the movement of the detected color.

    Subscribes
    ----------
    image (sensor_msgs/msg/Image) - The input image

    Publishes
    ---------
    new_image (sensor_msgs/msg/Image) - Annotated image with the path
    """

    def __init__(self):
        super().__init__("color_tracker")
        self.bridge = CvBridge()
        self.create_subscription(Image, 'image', self.yolo_callback, 10)
        self.create_subscription(Path, 'waypoints', self.waypoint_callback, 10)
        self.pub = self.create_publisher(Image, 'new_image', 10)
        self.path_publisher = self.create_publisher(Float32MultiArray, 'path_points', 10)


        # Store the path as a list of centroids
        self.path = []
        self.waypoints = []

        # Define the color to track (yellow)
        self.color_name = "yellow"
        self.color_range = ((25, 50, 50), (35, 255, 255))  # HSV range for yellow

        # self.color_name = "pink"
        # self.color_range = ((145, 50, 50), (165, 255, 255))  # HSV range for pink

        # self.color_name = "orange"
        # self.color_range =  ((5, 50, 50), (15, 255, 255))

        # self.color_name = "white"
        # self.color_range =  ((0, 0, 200), (180, 30, 255))

        # self.color_name = "green"
        # self.color_range =  ((0, 0, 200), (180, 30, 255))

        # self.color_name = "blue"
        # self.color_range =  ((90, 50, 50), (130, 255, 255))

        # Timer for path recording
        self.start_time = None
        #self.time_limit = 15 # end path recordign after 15 seconds 

    def waypoint_callback(self, msg):
        """Callback to receive waypoints from the waypoint generation node."""
        self.waypoints = [(int(pose.pose.position.x), int(pose.pose.position.y)) for pose in msg.poses]

    
    def smooth_new_point(self, path, window_size=10):
        """Smooth the latest point using only its nearest neighbors."""
        if len(path) < 2:  # No need to smooth with less than 2 points
            return path

        # Define the sliding window for smoothing the current point
        start = max(0, len(path) - window_size)
        neighbors = path[start:]

        # Compute the average position of the neighbors
        x_avg = sum(p[0] for p in neighbors) // len(neighbors)
        y_avg = sum(p[1] for p in neighbors) // len(neighbors)

        # Replace the latest point with its smoothed position
        path[-1] = (x_avg, y_avg)

        return path
    
    def smooth_path_ema(self, path, alpha=0.2):
        """Smooth the path using an exponential moving average."""
        if len(path) < 2:  # No need to smooth with less than 2 points
            return path

        smoothed_path = [path[0]]
        for i in range(1, len(path)):
            prev_x, prev_y = smoothed_path[-1]
            curr_x, curr_y = path[i]
            smoothed_path.append((
                int(alpha * curr_x + (1 - alpha) * prev_x),
                int(alpha * curr_y + (1 - alpha) * prev_y),
            ))
        return smoothed_path
    

    def smooth_path_spline(self, path, s=2):
        """Smooth the path using cubic spline interpolation."""
        if len(path) < 4:  # Spline requires at least 4 points
            return path

        x = [p[0] for p in path]
        y = [p[1] for p in path]

        # Fit a spline to the points
        tck, _ = splprep([x, y], s=s)
        x_smooth, y_smooth = splev(np.linspace(0, 1, len(path)), tck)

        smoothed_path = [(int(xi), int(yi)) for xi, yi in zip(x_smooth, y_smooth)]
        return smoothed_path



    def yolo_callback(self, image):
        """Track the specified color and create a smoothed path."""

        # Begin timer clock 
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        # elapsed_time = current_time - self.start_time
        # if elapsed_time > self.time_limit:
        #     # Stop updating the path after the time limit
        #     self.get_logger().info("Path recording time limit reached.")
        #     return

        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # Convert to HSV for color segmentation
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Create a mask for the specified color
        mask = cv2.inRange(hsv_image, np.array(self.color_range[0]), np.array(self.color_range[1]))

        # Find contours of the detected regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Annotate the image and update the path
        annotated_image = cv_image.copy()
        merged_contour = None

        # Merge nearby contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small regions by area
                if merged_contour is None:
                    merged_contour = contour
                else:
                    merged_contour = np.vstack((merged_contour, contour))

        # If a merged contour exists, calculate its centroid
        if merged_contour is not None:
            moments = cv2.moments(merged_contour)
            if moments["m00"] != 0:  # Avoid division by zero
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])

                # Add the centroid to the path
                # self.path.append((cX, cY))
                # Add the centroid to the path ONLY if it moves significantly
                if len(self.path) == 0 or (abs(cX - self.path[-1][0]) > 5 or abs(cY - self.path[-1][1]) > 5):
                    self.path.append((cX, cY))

                # Draw the current centroid
                cv2.circle(annotated_image, (cX, cY), 5, (0, 255, 0), -1)
                cv2.putText(
                    annotated_image,
                    f"{self.color_name} ({cX}, {cY})",
                    (cX + 10, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Smooth the path
        #self.path = self.smooth_path_local(self.path, window_size=5)

        # Smooth the path NOT HAPPY WITH 
        #self.path = self.smooth_new_point(self.path, window_size=10)

        #ema path NOT HAPPY WITH 
        #self.path = self.smooth_path_ema(self.path, alpha=0.2)

        #cubic spline NOT HAPPY WITH
        #self.path = self.smooth_path_spline(self.path, s=2)

        # Draw the smoothed path
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                cv2.line(annotated_image, self.path[i - 1], self.path[i], (0, 255, 0), 2)

        # Publish the raw path
        path_msg = Float32MultiArray()
        path_msg.data = [coord for point in self.path for coord in point]
        self.path_publisher.publish(path_msg)

        # Draw waypoints as red dots
        for x, y in self.waypoints:
            cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)  # Red dot for waypoints


        # Convert the annotated image back to ROS Image message
        new_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        # Publish the annotated image
        self.pub.publish(new_msg)





def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()
