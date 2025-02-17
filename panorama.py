import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_image(title, img):
    if len(img.shape) == 3:  # Color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.savefig(f'images/{title}.jpg')

def find_keypoints_and_matches(img1, img2, title):
    # Find SIFT keypoints and matches between two images
    sift = cv2.SIFT_create()
    
    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save_image(f'keypoint-matches-{title}-and-reference', match_img)

    return kp1, kp2, matches

def compute_homography(kp1, kp2, matches):
    # Compute the homography matrix using RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H

def warp_images(img1, img2, H):
    # Warp img1 into the reference frame of img2 using homography and back warping
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Find the corners of img1 in the new transformed space
    corners_img1 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_img1, H)
    
    # dimensions for the final panorama
    all_corners = np.concatenate((transformed_corners, np.float32([[0, 0], [w2, 0], [0, h2], [w2, h2]]).reshape(-1, 1, 2)), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])  # Shift to avoid negative coordinates
    H_translated = translation_matrix @ H  # Adjust the homography
    
    # Warp img1 into the new panorama space
    warped_img1 = cv2.warpPerspective(img1, H_translated, (x_max - x_min, y_max - y_min))
    
    # Also shift img2 to fit into the panorama
    translated_img2 = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    translated_img2[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2

    return warped_img1, translated_img2

def blend_images(img1, img2):
    # Blend two overlapping images using weighted blending
    mask1 = (img1 > 0).astype(np.float32)
    mask2 = (img2 > 0).astype(np.float32)

    blended = (img1 * mask1 + img2 * mask2) / (mask1 + mask2 + 1e-6)  # Prevent division by zero
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def stitch_images(img1, img2, title):
    print(f"Stitching image {title} into the reference frame")
    
    # Find keypoints and matches
    kp1, kp2, matches = find_keypoints_and_matches(img1, img2, title)
    
    # Compute Homography
    H = compute_homography(kp1, kp2, matches)
    
    # Warp Image 1 into the reference frame of Image 2
    warped_img1, translated_img2 = warp_images(img1, img2, H)
    
    # Blend the images
    panorama = blend_images(warped_img1, translated_img2)

    return panorama

def create_panorama(image_files):
    # Load images
    images = [cv2.imread(img) for img in image_files]

    # Stitch images progressively
    panorama = images[1]  # Reference image (img2)
    
    for i, img in enumerate([images[0]] + images[2:]):
        panorama = stitch_images(img, panorama, title=str(i+1) if i == 0 else str(i+2))

    save_image("final-panorama", panorama)

if __name__ == "__main__":
    image_files = ["2-1.jpeg", "2-2.jpeg", "2-3.jpeg", "2-4.jpeg"]
    print('Using image 2 as the reference frame')
    create_panorama(image_files)
