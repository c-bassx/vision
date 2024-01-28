import depthai as dai
import cv2

pipeline = dai.Pipeline()

# Create ColorCamera node
camera = pipeline.create(dai.node.ColorCamera)
camera.setBoardSocket(dai.CameraBoardSocket.RGB)
camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camera.setInterleaved(False)

# Create XLinkOut node for the camera output
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")

# Link the nodes
camera.video.link(xoutVideo.input)

# Create the DepthAI device
with dai.Device(pipeline) as device:
    # Get the output queue for the video stream
    videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        # Get the next video frame
        videoFrame = videoQueue.get()

        # Get the BGR frame from the video packet
        frame = videoFrame.getCvFrame()

        # Display the frame
        cv2.imshow("Camera Output", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# Destroy OpenCV window
cv2.destroyAllWindows()
