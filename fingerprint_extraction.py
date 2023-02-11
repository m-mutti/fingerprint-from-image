import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
# Load the image
img = cv.imread(sys.argv[1])

# Convert to HSV color space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define the lower and upper bounds for skin color in HSV
lower = np.array([0, 20, 70], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

# Threshold the image to extract only the skin color pixels
mask = cv.inRange(hsv, lower, upper)

# Perform morphological transformations to remove noise
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

# Detect contours in the binary image
contours, _ = cv.findContours(
    mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Sort the contours according to area
contours = sorted(contours, key=cv.contourArea, reverse=True)

# Find the largest 4 contour
contours = contours[:4]

# Create a binary mask using the contour information
mask = np.zeros_like(mask)
for i in contours:
    cv.drawContours(mask, [i], 0, 255, -1)

# Perform post-processing to further improve the quality of the segmented hand region
mask = cv.GaussianBlur(mask, (5, 5), 0)

# Segment the hand region from the rest of the image
hand = cv.bitwise_and(img, img, mask=mask)

# Adaptive thresholding
hand = cv.cvtColor(hand, cv.COLOR_BGR2GRAY)
hand = cv.adaptiveThreshold(
    hand, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# Separate fingers
fingers = []
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    fingers.append(hand[y:y + h, x:x + w])

# Show fingers
fig = plt.figure(figsize=(10, 10))
for index, finger in enumerate(fingers, start=1):
    fig.add_subplot(2, 2, index)
    plt.imshow(finger, cmap="gray")
    plt.title(f"Finger {index}")

plt.show()

