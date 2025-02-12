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
from me495_yolo.waypoint_follower import TurtleBotWaypointFollower

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
        self.freq = 10
        # self.tf_timer = self.create_timer(1.0/self.freq, self.broadcast_static_tf)  # Publish every 100ms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.waypoint_follower = TurtleBotWaypointFollower()
        


        # Timer variables
        self.start_time = None  # Store start time of movement
        self.time_limit = 8 # Stop recording waypoints after 5 seconds


        self.create_subscription(MarkerArray, 'waypoint_markers', self.marker_callback, 10)

        self.last_valid_dancer_pose = None  # Store last valid dancer position


        # Publishers
        self.pub = self.create_publisher(Image, 'new_image', 10)
        self.path_publisher = self.create_publisher(Float32MultiArray, 'path_points', 10)
        self.turtlebot_position_pub = self.create_publisher(Float32MultiArray, '/turtlebot_position', 10)


        # Color Tracking Settings (Pink marker)
        # Color Tracking Settings
        self.pink_range = ((145, 50, 50), (165, 255, 255))  # HSV range for pink (dancer)
        self.blue_range = ((35, 50, 50), (85, 255, 255)) #green but not changign nathign 
        #self.blue_range = ((90, 50, 50), (130, 255, 255))   # HSV range for blue (TurtleBot marker)

        # self.color_name = "pink"
        # self.color_range = ((145, 50, 50), (165, 255, 255))  # HSV range for pink

        # # self.color_name = "blue"
        # # self.color_range =  ((90, 50, 50), (130, 255, 255))


        self.depth_image = None  # Store the latest depth frame
        self.fx, self.fy, self.cx, self.cy = None, None, None, None  # Camera intrinsics
        self.pinkavg = np.zeros((int(self.freq*1.0), 2))

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


    def yolo_callback(self, image):
        """Track pink (dancer) and blue (TurtleBot) markers, merge contours, update paths, and publish real-world coordinates."""

        elapsed_time = 0  # ✅ Ensure elapsed_time is always defined at the start

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
                curr_pink_cX = int(moments["m10"] / moments["m00"])
                curr_pink_cY = int(moments["m01"] / moments["m00"])

                if (curr_pink_cX !=0 and curr_pink_cY !=0):
                    # Running average: Shift all elements left
                    self.pinkavg[:-1] = self.pinkavg[1:]

                    # Insert new position at the last index
                    self.pinkavg[-1] = np.array([curr_pink_cX, curr_pink_cY])

                # Compute smoothed centroid
                pink_cX, pink_cY = np.mean(self.pinkavg, axis=0)

                pink_cX = int(pink_cX)
                pink_cY = int(pink_cY) 

                ### **Step 2: Convert Pink Centroid to Real-World Coordinates**
                pink_world_coords = self.pixel_to_world(pink_cX, pink_cY)

                if pink_world_coords:
                    Xp, Yp, Zp = pink_world_coords
                    self.broadcast_camera_to_dancer(Xp, Yp, Zp)
                    self.get_logger().info(f"🩷 Dancer Position: X={Xp:.3f}, Y={Yp:.3f}, Z={Zp:.3f}")

                    ### ✅ **Step 3: Start Timer on First Detection**
                    if self.start_time is None:
                        self.start_time = time.time()  # Start tracking when dancer is first detected
                        self.get_logger().info("⏳ Started tracking dancer path")

                    # ✅ Always define elapsed_time
                    elapsed_time = time.time() - self.start_time
                    self.get_logger().info(f"⏳ Timer Check: {elapsed_time:.2f}s elapsed")

                    ### **Step 5: Add to Path Only if Movement is Significant**
                    if len(self.dancer_path) == 0 or (
                        abs(Xp - self.dancer_path[-1][0]) > 0.05 or abs(Yp - self.dancer_path[-1][1]) > 0.05
                    ):
                        self.dancer_path.append((Xp, Yp, Zp))  
                        self.get_logger().info(f"📍 Added Waypoint: ({Xp:.3f}, {Yp:.3f}, {Zp:.3f})")

                    ### **Step 6: Draw Pink Centroid**
                    cv2.circle(cv_image, (pink_cX, pink_cY), 5, (255, 0, 255), -1)
                    cv2.putText(cv_image, "Dancer", (pink_cX + 10, pink_cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        ### **Step 7: Publish the Final Waypoints Once After 5 Seconds**
        if elapsed_time > self.time_limit and len(self.dancer_path) > 0:
            path_msg = Float32MultiArray()
            path_msg.data = [coord for point in self.dancer_path for coord in point]
            self.get_logger().info(f"📤 Publishing Final 5-Second Dancer Path: {path_msg.data}")
            self.path_publisher.publish(path_msg)

            # **Clear the dancer path to prevent further updates in RViz**
            self.dancer_path = []  

        ### **Step 8: Publish the Annotated Image**
        new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.pub.publish(new_msg)

        ##blue mask 
        ### **Step 2: Detect & Merge Blue Contours (TurtleBot)**
        blue_mask = cv2.inRange(hsv_image, np.array(self.blue_range[0]), np.array(self.blue_range[1]))
        blue_contour = self.merge_contours(blue_mask)

        if blue_contour is not None:
            moments = cv2.moments(blue_contour)
            if moments["m00"] != 0:
                blue_cX = int(moments["m10"] / moments["m00"])
                blue_cY = int(moments["m01"] / moments["m00"])

                blue_world_coords = self.pixel_to_world(blue_cX, blue_cY)

                if blue_world_coords:
                    Xb, Yb, Zb = blue_world_coords

                    # Publish TurtleBot's position
                    turtlebot_msg = Float32MultiArray()
                    turtlebot_msg.data = [Xb, Yb]  # Only X, Y since TurtleBot moves on a plane
                    self.turtlebot_position_pub.publish(turtlebot_msg)

                    # update turtlebot position 
                    self.waypoint_follower.update_turtlebot_position(Xb, Yb)
                    
                     # Log the published position
                    self.get_logger().info(f"📡 Published TurtleBot Position: X={Xb:.3f}, Y={Yb:.3f}")


                    self.broadcast_camera_to_turtlebot(Xb, Yb, Zb)
                    self.get_logger().info(f"🔵 TurtleBot Position: X={Xb:.3f}, Y={Yb:.3f}, Z={Zb:.3f}")
                    self.get_logger().info(f"🔵 TurtleBot Detected at: X={Xb:.3f}, Y={Yb:.3f}, Z={Zb:.3f}")

                    # **Draw blue centroid on image**
                    cv2.circle(cv_image, (blue_cX, blue_cY), 5, (255, 255, 0), -1)
                    cv2.putText(cv_image, "TurtleBot", (blue_cX + 10, blue_cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        ### **Step 3: Publish the Annotated Image**
        new_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.pub.publish(new_msg)


    def merge_contours(self, mask):
        """Find and merge nearby contours to avoid gaps in tracking."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged_contour = None

        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small areas
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


        self.get_logger().info(f" Dancer in Camera Frame: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
        self.get_logger().info(f"Depth at ({u},{v}) = {depth:.3f} meters")


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
        """Broadcast transformation from camera frame to detected blue object (TurtleBot)."""
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'camera_color_optical_frame'  # Camera's frame
            t.child_frame_id = 'turtlebot_blue_object'  # New frame for TurtleBot marker


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
            self.get_logger().info(f"📡 Broadcasted TurtleBot TF: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

        except Exception as e:
            self.get_logger().error(f"Failed to publish TF transform for TurtleBot: {e}")



    def get_turtlebot_position(self):
        """Get the real-time TurtleBot position from TF."""
        try:
            transform = self.tf_buffer.lookup_transform('map', 'turtlebot_base', rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z
            self.get_logger().info(f"✅ Real TurtleBot Position: X={x}, Y={y}")
            return x, y, z
        except Exception as e:
            self.get_logger().warn(f"⚠️ Failed to get TurtleBot position: {e}")
            return None

    def compute_dancer_in_turtlebot_frame(self):
        """Compute dancer’s position in the TurtleBot’s reference frame using TF."""
        try:
            # Transform dancer position into TurtleBot's frame
            #transform = self.tf_buffer.lookup_transform('turtlebot_base', 'dancer_pink_object', rclpy.time.Time())
            transform = self.tf_buffer.lookup_transform('turtlebot_blue_object', 'dancer_pink_object', rclpy.time.Time())

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            self.last_valid_dancer_pose = (x, y, z)
            self.get_logger().info(f"✅ Dancer in TurtleBot Frame: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

            return x, y, z
        except Exception as e:
            self.get_logger().warn(f"⚠️ Failed to compute dancer in TurtleBot frame: {e}")
            return self.last_valid_dancer_pose if self.last_valid_dancer_pose else None



    
    def get_camera_to_turtlebot_transform(self):
        """Get real-time transform from camera to TurtleBot."""
        try:
            transform = self.tf_buffer.lookup_transform('turtlebot_base', 'camera_color_optical_frame', rclpy.time.Time())
            self.get_logger().info(f"✅ Real Camera->TurtleBot Transform: X={transform.transform.translation.x}, Y={transform.transform.translation.y}")
            return transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z
        except Exception as e:
            self.get_logger().warn(f"⚠️ Failed to get camera-to-TurtleBot transform: {e}")
            return None




    def marker_callback(self, msg):
        """Receive waypoint markers for visualization.""" 
        self.get_logger().info(f"Received {len(msg.markers)} markers")
        self.waypoint_markers = msg.markers

def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    rclpy.shutdown()
