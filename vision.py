import depthai as dai
import cv2

pipeline = dai.Pipeline()
stereo = pipeline.create(dai.node.StereoDepth)

camera = pipeline.create(dai.node.ColorCamera)
camera.setBoardSocket(dai.CameraBoardSocket.RGB)
camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camera.setInterleaved(False)

xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")

camera.video.link(xoutVideo.input)

with dai.Device(pipeline) as device:
    videoQueue = device.getOutputQueue(name = "video", maxSize = 4, blocking = False)

    while True:
        videoFrame = videoQueue.get()
        frame = videoFrame.getCvFrame()
        cv2.imshow("Camera Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()