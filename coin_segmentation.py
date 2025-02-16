# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load and preprocess image
# image = cv2.imread('coins-2-cropped.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5,5), 0)

# # Edge detection
# # canny = cv2.Canny(blur, 1, 125)
# canny = cv2.Canny(blur, 30, 90)  # Reduce lower threshold to capture more edges


# dilated = cv2.dilate(canny, (1,1), iterations=4)

# # Find contours
# contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # Segmentation using Watershed
# # Create a binary mask
# _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Remove noise using morphological operations
# kernel = np.ones((3,3), np.uint8)
# opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# # Sure background area
# # sure_bg = cv2.dilate(opening, kernel, iterations=3)

# sure_bg = cv2.dilate(opening, kernel, iterations=5)  # Increase iterations


# # Distance transform for sure foreground area
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# # _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)  # Increased threshold for better separation
# _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# sure_fg = cv2.erode(sure_fg, kernel, iterations=1)  # Remove small unwanted detections


# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg, sure_fg)

# # Marker labeling
# _, markers = cv2.connectedComponents(sure_fg)
# markers = markers + 1
# markers[unknown == 255] = 0

# # Apply watershed
# image_watershed = image.copy()
# cv2.watershed(image_watershed, markers)

# # Create an output image with different colors for each coin
# segmented_image = np.zeros_like(image)
# colors = np.random.randint(0, 255, (np.max(markers) + 1, 3), dtype=np.uint8)

# for i in range(2, np.max(markers) + 1):
#     segmented_image[markers == i] = colors[i]

# # Display segmented image
# plt.figure(figsize=(10,5))
# plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
# plt.title("Region-Based Segmentation of Coins")
# plt.axis('off')
# plt.show()

# # Count the number of coins
# print(f'Total Coins Detected: {np.max(markers) - 1}')

from skimage.filters import sobel
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import canny
from scipy import ndimage as ndi

coins = data.coins()
elmap = sobel(coins)
figure, axis = plt.subplots(figsize=(6, 4))
axis.imshow(elmap, cmap=plt.cm.gray)
axis.axis('off')
axis.set_title('map elevation')
plt.show()