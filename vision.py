import depthai as dai
import cv2
import numpy as np
import os
import sys
import math

from datetime import datetime
from networktables import NetworkTables

# Create pipeline
pipeline = dai.Pipeline()

# Initialize networktables
NetworkTables.initialize(server="PLACEHOLDER")
rpiTable = NetworkTables.getTable("raspberrypi")

# Variables
nn_path = "models/model.blob"
img_width = 640
img_height = 640

# Check file path integrity
if not os.path.isfile(nn_path):
    print(f"ERROR: File path {nn_path} does not exist.")
    sys.exit(1)

# Sources and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nn = pipeline.create(dai.node.YoloDetectionNetwork)

left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)

stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

left.out.link(stereo.left)
right.out.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_nn = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")
xout_nn.setStreamName("nn")

# Properties
cam_rgb.setPreviewSize(img_width, img_height)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Linking
cam_rgb.preview.link(nn.input)
stereo.depth.link(xout_depth.input)
cam_rgb.video.link(xout_rgb.input)
nn.out.link(xout_nn.input)

# Neural Network Configuration
try:
    nn.setBlobPath(nn_path)
except:
    print(f"ERROR: File path {nn_path} contains an empty blob.")
    sys.exit(1)

nn.setConfidenceThreshold(0.5)
nn.setNumClasses(2)
nn.setCoordinateSize(4)
nn.setIouThreshold(0.5)

# Functions
def calculate_angle(offset):
    cameraHorizontalFOV = np.deg2rad(73.5)  # ???
    depthImageWidth = 1080.0                # ???

    return math.atan(math.tan(cameraHorizontalFOV / 2.0) * offset / (depthImageWidth / 2.0))

def addText(text, location, image, color = (255, 255, 255), font = cv2.FONT_HERSHEY_TRIPLEX):
    cv2.putText(
        img = frame,
        org = location,
        text = text,
        fontFace = font,
        fontScale = 0.5,
        color = color,
        thickness = 1
    )

try:
    with dai.Device(pipeline) as device:
        # Get the rgb frames, depth frames, and nn data
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()
            in_nn = q_nn.get()

            if in_rgb is None or in_depth is None or in_nn is None:
                continue

            # Decode neural network's output, draw bounding boxes
            frame = in_rgb.getCvFrame()
            depth = in_depth.getFrame()
            detections = in_nn.detections

            for index, detection in enumerate(detections, start=1):
                obj_xmin = int(detection.xmin * img_width)
                obj_ymin = int(detection.ymin * img_height)
                obj_xmax = int(detection.xmax * img_width)
                obj_ymax = int(detection.ymax * img_height)

                obj_center_x = int((obj_xmin + obj_xmax) / 2)
                obj_center_y = int((obj_ymin + obj_ymax) / 2)
                center_depth = depth[obj_center_y, obj_center_x]

                depth_map = depth[obj_ymin:obj_ymax, obj_xmin:obj_xmax]
                averageDepth = np.mean(depth_map)

                midpointWidth = int(depth.shape[1] / 2)
                midpointHeight = int(depth.shape[0] / 2)

                obj_xOffset = obj_center_x - midpointWidth
                obj_yOffset = obj_center_y - midpointHeight

                obj_angle_x = calculate_angle(obj_xOffset)
                obj_angle_y = calculate_angle(obj_yOffset)

                z = averageDepth # averageDepth works too
                x = z * math.tan(obj_angle_x)
                y = -z * math.tan(obj_angle_y)

                distance = np.sqrt(x*x + y*y + z*z)

                time = datetime.now().strftime("%H:%M:%S")

                # Print values for debugging & status
                print(f"Time:                   {time}")
                print(f"Detection #:            {index}")
                print(f"Number of detections:   {len(detections)}") # hopefully works?
                print(f"Object center x:        {obj_center_x}")
                print(f"Object center y:        {obj_center_y}")
                print(f"Object distance:        {distance}")
                print(f"Object average depth:   {averageDepth}")
                print("\n")
                print(f"Object relative x:      {x}")
                print(f"Object relative y:      {y}")
                print(f"Object relative z:      {z}")
                print("\n")

                # Draw bounding box
                cv2.rectangle(
                    img = frame,
                    pt1 = (obj_xmin, obj_ymin),
                    pt2 = (obj_xmax, obj_ymax),
                    color = (0, 255, 0),
                    thickness = 2
                )

                # Add text to the bounding box
                # addText("X: {x} mm", (obj_xmin, obj_ymax + 10), frame)
                # addText("Y: {y} mm", (obj_xmin + 25, obj_ymax + 10), frame)
                # addText("Z: {z} mm", (obj_xmin + 50, obj_ymax + 10), frame)

                # Post to networktables
                rpiTable.getEntry("notetrans").setDoubleArray([x, y, z])

            cv2.imshow("RGB", frame)

            if cv2.waitKey(1) == ord("q"):
                break
except KeyboardInterrupt:
    print("Program interrupted by user, stopping...")
    sys.exit(1)

cv2.destroyAllWindows()
print("Resources released, program exited.")