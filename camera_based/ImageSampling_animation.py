import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

ser = serial.Serial("/dev/cu.usbmodem11401",9600, timeout = 1)
frameLength, frameWidth = 64, 64
frameSize = frameLength * frameWidth


# plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

image = np.ones((frameLength, frameWidth), dtype=np.uint8)
ax.imshow(image, cmap='gray', animated=True)


def get_image():

    data = ser.read(int(frameSize * 2.1))
    idx = 0
    for idx in range(len(data) - 1):
        if data[idx] == 0x55 and data[idx + 1] == 0xAA:  # image starts with this 2byte identifier
            idx += 2  # move over the identifier
            break

    print(idx)
    frame = data[idx:idx + frameSize]  # get the image
    frame = np.frombuffer(frame, dtype=np.uint8)  # convert into numpy array
    image = np.reshape(frame, (frameLength, frameWidth))  # reshape
    return image


def updatefig(*args):
    image = get_image()
    print(image[30,:10])
    ax.imshow(image,cmap='gray')
    return ax,

anim = animation.FuncAnimation(fig, updatefig, interval=1, blit=True)
plt.show()

# while 1:







# state, read, result, startbyte = 0, 0, 0, 0
# while 1:
#     if read == 0:
#         startbyte = ser.read()
#         if (startbyte == 0x55):
#             state = 1
#
#         if (startbyte == 0xAA and state == 1):
#             read = 1
#
#         if (startbyte == 0xBB and state == 1):
#             result = 1
#     if result == 1:
#         ser.read(2)
#
#     if read == 1:
#         res = ser.read(frameSize)
#         read = 0



