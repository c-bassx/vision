import depthai as dai
import cv2
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Variables
nn_path = ""

# Define sources and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nn = pipeline.create(dai.node.NeuralNetwork)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_nn = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")
xout_nn.setStreamName("nn")

# Properties
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Linking
cam_rgb.preview.link(nn.input)
stereo.depth.link(xout_depth.input)
cam_rgb.video.link(xout_rgb.input)
nn.out.link(xout_nn.input)

# Neural Network
nn.setBlobPath(nn_path)

# Functions
def absolute_distance(x, y, z, params):
    fx, fy, cx, cy = params
    i = (x - cx) * z / fx
    j = (y - cy) * z / fy
    return i, j, z


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
            z = np.mean(depth_values)

            i, j, k = absolute_distance(xmin, ymin, z, params=(fx, fy, cx, cy))
            distance = np.sqrt(i**2 + j**2 + k**2)

            cv2.rectangle(
                frame, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2
            )

            cv2.putText( 
                frame, f"3D Pos: ({i:.2f}, {j:.2f}, {k:.2f}), Distance: {distance:.2f} mm",
                (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
            )

        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()