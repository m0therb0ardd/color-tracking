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
import rclpy
from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class YoloNode(Node):
    """
    Identify and track specific colors (pink, orange, white, green, blue, yellow).

    Subscribes
    ----------
    image (sensor_msgs/msg/Image) - The input image

    Publishes
    ---------
    new_image (sensor_msgs/msg/Image) - The image with bounding boxes around detected colors
    """

    def __init__(self):
        super().__init__("pose")
        self.bridge = CvBridge()
        self.create_subscription(Image, 'image', self.yolo_callback, 10)
        self.pub = self.create_publisher(Image, 'new_image', 10)

    def yolo_callback(self, image):
        """Identify and track specific colors in the scene."""
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # Convert to HSV for color segmentation
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define HSV ranges for colors
        color_ranges = {
            "pink": ((145, 50, 50), (165, 255, 255)),
            "orange": ((5, 50, 50), (15, 255, 255)),
            "white": ((0, 0, 200), (180, 30, 255)),
            "green": ((35, 50, 50), (85, 255, 255)),
            "blue": ((90, 50, 50), (130, 255, 255)),
            "yellow": ((25, 50, 50), (35, 255, 255)),
        }

        # Create a copy of the image for drawing annotations
        annotated_image = cv_image.copy()

        # Loop through each color and detect regions
        for color_name, (lower, upper) in color_ranges.items():
            # Create a mask for the current color
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

            # Find contours of the detected regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # calculate centroid and label the detected colors
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Filter small regions by area
                    # Calculate the centroid
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:  # Avoid division by zero
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                        # Annotate the centroid on the image
                        cv2.circle(annotated_image, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.putText(
                            annotated_image,
                            f"{color_name} ({cX}, {cY})",
                            (cX + 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

        # Convert the annotated image back to ROS Image message
        new_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')

        # Publish the annotated image
        self.pub.publish(new_msg)


def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()
