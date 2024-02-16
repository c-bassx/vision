import depthai as dai
import cv2
import numpy as np
import os
import sys

from networktables import NetworkTables

# Create pipeline
pipeline = dai.Pipeline()

# Initialize networktables
NetworkTables.initialize(server="PLACEHOLDER")
rpiTable = NetworkTables.getTable("raspberrypi")

# Variables
nn_path = "models/model.blob"
preview_width = 300
preview_height = 300

# Check file path integrity
if not os.path.isfile(nn_path):
    print(f"ERROR: File path {nn_path} does not exist.")
    sys.exit(1)

# Sources and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nn = pipeline.create(dai.node.YoloDetectionNetwork)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_nn = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")
xout_nn.setStreamName("nn")

# Properties
cam_rgb.setPreviewSize(preview_width, preview_height)
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
def absolute_distance(x, y, z, params):
    fx, fy, cx, cy = params
    i = (x - cx) * z / fx
    j = (y - cy) * z / fy
    return i, j, z

def calculate_angles(x, y, z):
    pitch = np.arctan2(y, np.sqrt(x**2 + z**2))  # x-axis rotation
    yaw = np.arctan2(x, z)                      #  y-axis rotation
    roll = np.arctan2(np.sqrt(x**2 + y**2), z)  #  z-axis rotation

    # Convert radians to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll

# Possible future use? If not then useless
def get_object_data(detections, depth_frame, fx, fy, cx, cy):
    object_data = []
    for detection in detections:
        xmin = int(detection.xmin * 300)
        ymin = int(detection.ymin * 300)
        xmax = int(detection.xmax * 300)
        ymax = int(detection.ymax * 300)

        depth_values = depth_frame[ymin:ymax, xmin:xmax]

        if depth_values.size == 0 or np.any(depth_values <= 0):
            print(f"Warning: Invalid depth data for detection at ({xmin}, {ymin}).")
            continue

        z = np.mean(depth_values)
        i, j, k = absolute_distance(xmin, ymin, z, params=(fx, fy, cx, cy))
        distance = np.sqrt(i**2 + j**2 + k**2)
        pitch, yaw, roll = calculate_angles(i, j, k)

        object_data.append({
            "x": i,
            "y": j,
            "z": k,
            "distance": distance,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        })

    return object_data

try:
    with dai.Device(pipeline) as device:
        # Camera intrinsics
        intrinsics = device.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.RGB)
        fx, fy, cx, cy = (
            intrinsics[0][0],
            intrinsics[1][1],
            intrinsics[0][2],
            intrinsics[1][2],
        )

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
            depth_frame = in_depth.getFrame()
            detections = in_nn.detections
            for detection in detections:
                xmin = int(detection.xmin * 300)
                ymin = int(detection.ymin * 300)
                xmax = int(detection.xmax * 300)
                ymax = int(detection.ymax * 300)

                depth_values = depth_frame[ymin:ymax, xmin:xmax]

                if depth_values.size == 0 or np.any(depth_values <= 0):
                    print(f"Warning: Invalid depth data for detection at ({xmin}, {ymin}).")
                    continue

                z = np.mean(depth_values)

                i, j, k = absolute_distance(xmin, ymin, z, params=(fx, fy, cx, cy))
                distance = np.sqrt(i**2 + j**2 + k**2)
                pitch, yaw, roll = calculate_angles(i, j, k)

                cv2.rectangle(
                    frame, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2
                )

                cv2.putText( 
                    frame, f"3D Pos: ({i:.2f}, {j:.2f}, {k:.2f}), Distance: {distance:.2f} mm",
                    (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                )

                cv2.putText(
                    frame, f"Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Roll: {roll:.2f}°",
                    (xmin, ymin - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255)
                )

            cv2.imshow("RGB", frame)

            # Post to networktables
            rpiTable.getEntry("notetrans").setDoubleArray([i, j, k])

            if cv2.waitKey(1) == ord("q"):
                break
except KeyboardInterrupt:
    print("Program interrupted by user, stopping...")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)


cv2.destroyAllWindows()
print("Resources released, program exited.")