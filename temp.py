import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image1 = cv2.imread('1.jpeg', cv2.IMREAD_COLOR)
image2 = cv2.imread('2.jpeg', cv2.IMREAD_COLOR)
image3 = cv2.imread('4.jpeg', cv2.IMREAD_COLOR)
image4 = cv2.imread('5.jpeg', cv2.IMREAD_COLOR)

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Match keypoints using FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Store all good matches as per Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)

# Display matched keypoints
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title('Matched Keypoints')
plt.axis('off')
plt.show()

# Extract location of good matches
points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

for i, match in enumerate(good_matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Compute homography using RANSAC
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

# Project image1 into the plane of image2 using back warping
height, width, channels = image2.shape
warped_image1 = cv2.warpPerspective(image1, H, (width, height))

# Weighted blending
blended_image = cv2.addWeighted(image2, 0.5, warped_image1, 0.5, 0)

# Display blended image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
plt.title('Blended Image (Image1 + Image2)')
plt.axis('off')
plt.show()

# Repeat the process for image3 and image4
def project_image(source_image, target_image):
    gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    keypoints_source, descriptors_source = sift.detectAndCompute(gray_source, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray2, None)

    matches = flann.knnMatch(descriptors_source, descriptors_target, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    points_source = np.zeros((len(good_matches), 2), dtype=np.float32)
    points_target = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points_source[i, :] = keypoints_source[match.queryIdx].pt
        points_target[i, :] = keypoints_target[match.trainIdx].pt

    H, _ = cv2.findHomography(points_source, points_target, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(source_image, H, (width, height))
    blended_image = cv2.addWeighted(target_image, 0.5, warped_image, 0.5, 0)
    return blended_image

# Project image3 into the plane of image2
blended_image_3 = project_image(image3, blended_image)

# Display blended image with image3
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(blended_image_3, cv2.COLOR_BGR2RGB))
plt.title('Blended Image (Image1 + Image2 + Image3)')
plt.axis('off')
plt.show()

# Project image4 into the plane of image2
blended_image_4 = project_image(image4, blended_image_3)

# Display final blended image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(blended_image_4, cv2.COLOR_BGR2RGB))
plt.title('Blended Image (Image1 + Image2 + Image3 + Image4)')
plt.axis('off')
plt.show()