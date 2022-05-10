import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import cv2

ser = serial.Serial("/dev/cu.usbmodem1101",9600, timeout = 1)
frameLength, frameWidth = 96, 96
frameSize = frameLength * frameWidth
identifier = [0x55, 0xAA]


# plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

image = np.ones((frameLength, frameWidth), dtype=np.uint8)
ax.imshow(image, cmap='gray', animated=True)


def get_image():

    data = ser.read(int(frameSize * 2.1))
    # ser.reset_input_buffer()
    # ser.flushOutput()
    # while 1:
    #     data = ser.read_until(b'\x55\xAA')
    #     if len(data) == frameSize + 2:
    #         break

    idx = 0
    for idx in range(len(data) - 1):
        if data[idx] == identifier[0] and data[idx + 1] == identifier[1]:  # image starts with this 2byte identifier
            idx += 2  # move over the identifier
            break

    frame = data[idx:idx + frameSize]  # get the image
    frame = np.frombuffer(frame, dtype=np.uint8)  # convert into numpy array
    image = np.reshape(frame, (frameLength, frameWidth))  # reshape
    return image


def updatefig(*args):
    image = get_image()
    cv2.imwrite("image.pgm", image) # save image to local
    ax.imshow(image,cmap='gray')
    return ax,

anim = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
plt.show()







