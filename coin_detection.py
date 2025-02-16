import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('coins-2-cropped-2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.axis('off')  # To hide axis

blur = cv2.GaussianBlur(gray, (5,5), 0)
plt.imshow(blur, cmap='gray')
plt.show()

canny = cv2.Canny(blur, 1, 125)
plt.imshow(canny, cmap='gray')
plt.show()

dilated = cv2.dilate(canny, (1,1), iterations = 4)
plt.imshow(dilated, cmap='gray')
plt.show()

# Find contours in the edge-detected image
contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours on
contour_image = image.copy()


(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)

plt.imshow(rgb)


print('Coins in the image: ', len(cnt))
plt.show()