import depthai as dai

with dai.Device() as device:
    calibData = device.readCalibration()
    print(f"RGB FOV {calibData.getFov(dai.CameraBoardSocket.RGB)}, Mono FOV {calibData.getFov(dai.CameraBoardSocket.LEFT)}")