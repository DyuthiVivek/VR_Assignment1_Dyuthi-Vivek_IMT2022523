import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('coins-2.jpg')
#image preprocessing
image_blur = cv2.medianBlur(image,25)

image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

image_res ,image_thresh = cv2.threshold(image_blur_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Edge detection
canny = cv2.Canny(image_blur_gray, 30, 125)

plt.imshow(canny, cmap='gray')
plt.show()

# dilated = cv2.dilate(canny, (1,1), iterations=4)

# # Find contours
# contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # Draw detected contours
# contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
# cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)

# # Display detected contours
# plt.figure(figsize=(10,5))
# plt.imshow(contour_image)
# plt.title("Detected Coins")
# plt.axis('off')
# plt.show()


# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(image_thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown ==255] = 0

markers = cv2.watershed(image,markers)
image[markers == -1] = [255,0,0]

plt.imshow(markers,cmap='gray')
plt.show()


labels = np.unique(markers)
  
coins = []
for label in labels[2:]:  
  
# Create a binary image in which only the area of the label is in the foreground 
#and the rest of the image is in the background   
    target = np.where(markers == label, 255, 0).astype(np.uint8)
    
  # Perform contour extraction on the created binary image
    contours, hierarchy = cv2.findContours(
        target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    coins.append(contours[0])

# Draw the outline
img = cv2.drawContours(image, coins, -1, color=(0, 255, 0), thickness=15)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img,cmap='gray')

plt.show()


number_of_objects_in_image= len(coins)

print ("The number of objects in this image: ", str(number_of_objects_in_image))