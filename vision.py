# Rewrite of vision code

# Necessary imports
import depthai as dai
import numpy as np
import math
import time
import cv2

from networktables import NetworkTables

# Create pipeline
pipeline = dai.Pipeline()

# Initialize networktables
NetworkTables.initialize(server="roboRIO-4159-FRC.local")
rpiTable = NetworkTables.getTable("raspberrypi")

# Global declarations
nnPath = "models/model.blob"
img_width = 384
img_height = 640

# Sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)

xOutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
xOutDepth = pipeline.create(dai.node.XLinkOut)

stereo = pipeline.create(dai.node.StereoDepth)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)

stereo.setLeftRightCheck(False)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

xOutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
xOutDepth.setStreamName("depth")

# Properties
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(30)

# Neural Network Settings
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(2)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xOutRgb.input)
detectionNetwork.out.link(nnOut.input)
stereo.depth.link(xOutDepth.input)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Connect to device, start pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    # Get FOV
    calibData = device.readCalibration()
    fovRgb = calibData.getFov(dai.CameraBoardSocket.RGB)
    fovLeft = calibData.getFov(dai.CameraBoardSocket.LEFT)
    fovRight = calibData.getFov(dai.CameraBoardSocket.RIGHT)

    def calculateAngle(offset):
        cameraHorizontalFOV = np.deg2rad(fovRgb)  
        depthImageWidth = 1080.0                

        return math.atan(math.tan(cameraHorizontalFOV / 2.0) * offset / (depthImageWidth / 2.0))

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    def scaleDown(number):
        return round(number / 1000, 1)
    
    def sendToRobot(x, y, z):
        rpiTable.getEntry("notetrans").setDoubleArray([x, y, z])

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
    
    def displayFrame(name, frame, depth):
        color = (255, 0, 0)

        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            x, y, z, dist = getCoords(frame, detection, depth)
            sendToRobot(x, y, z) # send xyz translation over networktables to robot

            cv2.putText(frame, f"Note: {x}, {y}, {z}, {dist}", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        cv2.imshow(name, frame)

    def getCoords(frame, detection, depth):
        xmin, ymin, xmax, ymax = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

        centerX = int((xmin + xmax) / 2)
        centerY = int((ymin + ymax) / 2)
        centerDepth = depth[centerX, centerY]
        print(f"centerDepth: {centerDepth}")

        depthMap = depth[ymin:ymax, xmin:xmax]
        averageDepth = np.mean(depthMap)

        midpointWidth = int(depth.shape[1] / 2)
        midpointHeight = int(depth.shape[0] / 2)

        xOffset = centerX - midpointWidth
        yOffset = centerY - midpointHeight

        angleX = calculateAngle(xOffset)
        angleY = calculateAngle(yOffset)

        z = int(averageDepth)
        x = int(z * math.tan(angleX))
        y = int(-z * math.tan(angleY))
        distance = int(np.sqrt(x*x + y*y + z*z))

        x = scaleDown(x) # TODO: Fix x coordinate
        y = scaleDown(y)
        z = scaleDown(z)
        distance = scaleDown(distance)

        return x, y, z, distance

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()
        depth = qDepth.get().getFrame() # continuously update depth data

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            
        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame, depth)

        if cv2.waitKey(1) == ord('q'):
            break