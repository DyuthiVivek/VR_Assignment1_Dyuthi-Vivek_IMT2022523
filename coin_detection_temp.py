import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess image
image = cv2.imread('coins-2-cropped-2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Edge detection
canny = cv2.Canny(blur, 1, 125)
dilated = cv2.dilate(canny, (1,1), iterations=4)

# Find contours
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw detected contours
contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)

# Display detected contours
plt.figure(figsize=(10,5))
plt.imshow(contour_image)
plt.title("Detected Coins")
plt.axis('off')
plt.show()

# Segmentation using Watershed
# Create a binary mask
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove noise using morphological operations
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform for sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
image_watershed = image.copy()
cv2.watershed(image_watershed, markers)
image_watershed[markers == -1] = [255, 0, 0]  # Mark boundaries in red

print(contours, len(contours))
# Draw bounding boxes around each detected coin
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image_watershed, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display segmented image
plt.figure(figsize=(10,5))
plt.imshow(cv2.cvtColor(image_watershed, cv2.COLOR_BGR2RGB))
plt.title("Segmented Coins with Bounding Boxes")
plt.axis('off')
plt.show()

# Count the number of coins
print(f'Total Coins Detected: {len(contours)}')
