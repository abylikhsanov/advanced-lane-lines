import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import os
import glob

master = Tk()
file_path = filedialog.askopenfilename()
directory = os.path.split(file_path)[0]
filename = os.path.basename(file_path)
file_path = directory + '/' + filename
# Make a list of calibration images
images = glob.glob(directory + '\*.jpg')
index = images.index(file_path)

img = cv2.imread(file_path)
img_copy = np.copy(img)
cv2.namedWindow('Image')
cv2.namedWindow('Threshold viewer')

channel = img_copy[:, :, 0]
small = cv2.resize(img_copy, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('Image', small)


def threshold(x):
    # get current positions of trackbars
    min = cv2.getTrackbarPos('Min', 'Threshold viewer')
    max = cv2.getTrackbarPos('Max', 'Threshold viewer')

    binary = np.zeros_like(channel)
    binary[(channel >= min) & (channel <= max)] = 1

    final_image_RGB = np.dstack((binary, binary, binary)) * 255

    cv2.imshow('Threshold viewer', final_image_RGB)


# create trackbars for color change
cv2.createTrackbar('Min', 'Threshold viewer', 0, 255, threshold)
cv2.createTrackbar('Max', 'Threshold viewer', 255, 255, threshold)

x = 0
threshold(x)


def sel():
    global channel

    selection = v.get()

    if 0 <= selection <= 2:
        img_copy = np.copy(img)
    if 3 <= selection <= 5:
        img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if 6 <= selection <= 8:
        img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if 9 <= selection <= 11:
        img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if 12 <= selection <= 14:
        img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    small = cv2.resize(img_copy, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Image', small)
    selection = selection % 3
    channel = img_copy[:, :, selection]
    threshold(x)


def prev_image():
    global index, img
    index -= 1
    if index == 0:
        prev_button.config(state=DISABLED)
    img = cv2.imread(images[index])
    sel()
    next_button.config(state=NORMAL)


def next_image():
    global index, img
    index += 1
    if index == (len(images) - 1):
        next_button.config(state=DISABLED)
    img = cv2.imread(images[index])
    sel()
    prev_button.config(state=NORMAL)


v = IntVar()
v_image = IntVar()

Radiobutton(master, text="BGR B channel", variable=v, value=0, command=sel).pack(anchor=W)
Radiobutton(master, text="BGR G channel", variable=v, value=1, command=sel).pack(anchor=W)
Radiobutton(master, text="BGR R channel", variable=v, value=2, command=sel).pack(anchor=W)
Radiobutton(master, text="HLS H channel", variable=v, value=3, command=sel).pack(anchor=W)
Radiobutton(master, text="HLS L channel", variable=v, value=4, command=sel).pack(anchor=W)
Radiobutton(master, text="HLS S channel", variable=v, value=5, command=sel).pack(anchor=W)
Radiobutton(master, text="HSV H channel", variable=v, value=6, command=sel).pack(anchor=W)
Radiobutton(master, text="HSV S channel", variable=v, value=7, command=sel).pack(anchor=W)
Radiobutton(master, text="HSV V channel", variable=v, value=8, command=sel).pack(anchor=W)
Radiobutton(master, text="YUV Y channel", variable=v, value=9, command=sel).pack(anchor=W)
Radiobutton(master, text="YUV U channel", variable=v, value=10, command=sel).pack(anchor=W)
Radiobutton(master, text="YUV V channel", variable=v, value=11, command=sel).pack(anchor=W)
Radiobutton(master, text="YCrCb Y channel", variable=v, value=12, command=sel).pack(anchor=W)
Radiobutton(master, text="YCrCb Cr channel", variable=v, value=13, command=sel).pack(anchor=W)
Radiobutton(master, text="YCrCb Cb channel", variable=v, value=14, command=sel).pack(anchor=W)

prev_button = Button(master, text="Prev image", command=prev_image)
prev_button.pack()
next_button = Button(master, text="Next image", command=next_image)
next_button.pack()

if index == 0:
    prev_button.config(state=DISABLED)
elif index == (len(images) - 1):
    next_button.config(state=DISABLED)

mainloop()

cv2.destroyAllWindows()