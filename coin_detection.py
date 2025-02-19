import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # image preprocessing
    image_blur = cv2.medianBlur(image,25)
    image_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    image_res, image_thresh = cv2.threshold(image_blur_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return image_blur_gray, image_res, image_thresh

def edge_and_coin_detection(image_blur_gray):
    # canny edge detection
    canny = cv2.Canny(image_blur_gray, 1, 125)
    plt.imshow(canny, cmap='gray')
    plt.axis('off')
    plt.title("Canny Edge Detection")
    plt.savefig('images/coin-detection-canny.jpg')

    # detecing the coins using contours
    dilated = cv2.dilate(canny, (1,1), iterations=4)

    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # draw detected contours
    contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    cv2.drawContours(contour_image, contours, -1, (0,255,0), 6)

    # Display detected contours
    plt.figure(figsize=(10,5))
    plt.axis('off')
    plt.imshow(contour_image)
    plt.title("Detected Coins")
    plt.savefig('images/coin-detection-contours.jpg')


def region_based_segmentation(image_thresh):
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
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0
    markers += 1
    # Now, mark the region of unknown with zero
    markers[unknown ==255] = 0

    # Apply watershed
    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]

    plt.title("Segmented Coins")
    plt.imshow(markers,cmap='gray')
    plt.axis('off')
    plt.savefig('images/coin-detection-region-based-segmented.jpg')
    
    return markers

def segment_individual_coins_and_count(markers, image):
    labels = np.unique(markers)

    coins = []
    for label in labels[2:]:
        # Create a binary image - only the area of the label is in the foreground
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        
        # Perform contour extraction on the created binary image
        contours, _ = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # For each coin (contour), extract and save the segmented coin
        for i, contour in enumerate(contours):
            # mask for the current coin
            mask = np.zeros_like(target)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Extract the coin from the original
            coin = cv2.bitwise_and(image, image, mask=mask)
            
            # bounding box around the coin
            x, y, w, h = cv2.boundingRect(contour)
            coin_cropped = coin[y:y+h, x:x+w]
            
            coin_filename = f'coin_{label-1}.jpg'
            cv2.imwrite(f'images/{coin_filename}', coin_cropped)
            
            coins.append(coin_cropped)

    print("The number of objects in this image: ", len(coins))


if __name__ == "__main__":
    image = cv2.imread('coins.jpg')
    image_blur_gray, image_res, image_thresh = preprocess_image(image)
    edge_and_coin_detection(image_blur_gray)
    markers = region_based_segmentation(image_thresh)
    segment_individual_coins_and_count(markers, image)
