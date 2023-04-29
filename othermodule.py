import cv2 
import re
import numpy
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

def seperate(image):
    # getting the size of image
    size=image.shape
    # Converting into the numpy array
    data=numpy.array(image)
    # Manuplating the Values of Pixels 
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            if data[i][j]<100:
                data[i][j]=1
    # Updating the original image
    image=data
    return image

def histogram_equalization(image):
# Calculate histogram
    hist = np.zeros(256, dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
# Calculate cumulative distribution function
    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
# Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
# Apply equalization
    image_equalized = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_equalized[i, j] = cdf_normalized[image[i, j]]

    return image_equalized.astype(np.uint8)

def Finding_FrontPage_Keypoints(img1,img2):
    # Find keypoints and descriptors for each image
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    des1 = des1.astype('float32')
    des2 = des2.astype('float32')
    # Match keypoints between the two images
    bf = cv2.BFMatcher()
    kp1_pts = np.array([kp1[i].pt for i in range(len(kp1))], dtype=np.float32)
    kp2_pts = np.array([kp2[i].pt for i in range(len(kp2))], dtype=np.float32)

    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9* n.distance:
            good_matches.append(m)

    # Get the coordinates of matching keypoints in each image
    pts1 = [kp1[m.queryIdx].pt for m in good_matches]
    pts2 = [kp2[m.trainIdx].pt for m in good_matches]

    # Find homography between the two images
    H, mask = cv2.findHomography(np.float32(pts2), np.float32(pts1), cv2.RANSAC)

    # Warp image2 to align with image1
    h, w = img1.shape
    img2_aligned = cv2.warpPerspective(img2, H, (w, h))

    # Crop the aligned image to a specific region
    #     x, y= 0, 0 
    x, y, width, height = 0, 0, w, h

    # Image are cropped    
    img1_cropped = img1[y:y+height, x:x+width]
    img2_cropped = img2_aligned[y:y+height, x:x+width]
    
    
    # Apply thresholding to convert the image to binary
    thresh = cv2.threshold(img1_cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Apply dilation to make the text thicker
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    dilate = cv2.dilate(thresh, kernel, iterations=10)
    
    # Apply thresholding to convert the image to binary
    thresh3 = cv2.threshold(img2_cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Apply dilation to make the text thicker
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    dilate3 = cv2.dilate(thresh3, kernel3, iterations=1)
    # Finding the Adaptive threshold
    thresh1 = cv2.adaptiveThreshold(seperate(histogram_equalization(dilate)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,11,2)
    # Finding the Adaptive threshold
    thresh2 = cv2.adaptiveThreshold(seperate(histogram_equalization(dilate3)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,11,2)
        

    # Find keypoints and descriptors for the cropped images
    kp1_cropped, des1_cropped = sift.detectAndCompute(thresh1, None)
    kp2_cropped, des2_cropped = sift.detectAndCompute(thresh2, None)

    kp1_cropped_pts = np.array([kp1_cropped[i].pt for i in range(len(kp1_cropped))], dtype=np.float32)
    kp2_cropped_pts = np.array([kp2_cropped[i].pt for i in range(len(kp2_cropped))], dtype=np.float32)

    matches_cropped = bf.knnMatch(des1_cropped.astype('float32'), des2_cropped.astype('float32'), k=2)

    # Apply ratio test to filter good matches
    good_matches_cropped = []
    for m, n in matches_cropped:
        if m.distance < 0.3 * n.distance:
            good_matches_cropped.append(m)

    # Boolean
    Matched = False
    
    print( "lenght ",len(good_matches_cropped))
    # Check if the images are the same
    if len(good_matches_cropped) >= 10:
        Matched = True
    
    # Draw matches on a new image
    result = cv2.drawMatches(thresh1, kp1_cropped,thresh2, kp2_cropped, good_matches_cropped, None)
    
    # Show result image
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return Matched



def Finding_BackPage_Keypoints(img1,img2):
    # Find keypoints and descriptors for each image
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match keypoints between the two images
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9* n.distance:
            good_matches.append(m)

    # Get the coordinates of matching keypoints in each image
    pts1 = [kp1[m.queryIdx].pt for m in good_matches]
    pts2 = [kp2[m.trainIdx].pt for m in good_matches]

    # Find homography between the two images
    H, mask = cv2.findHomography(np.float32(pts2), np.float32(pts1), cv2.RANSAC)

    # Warp image2 to align with image1
    h, w = img1.shape
    img2_aligned = cv2.warpPerspective(img2, H, (w, h))

    # Crop the aligned image to a specific region
    # x, y, width, height = 10, 0, 150, 70
    x, y, width, height = 500, 150, 300, 600

    # Image are cropped    
    img1_cropped = img1[y:y+height, x:x+width]
    img2_cropped = img2_aligned[y:y+height, x:x+width]
    
    
    # Apply thresholding to convert the image to binary
    thresh = cv2.threshold(img1_cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Apply dilation to make the text thicker
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    dilate = cv2.dilate(thresh, kernel, iterations=10)
    
    # Apply thresholding to convert the image to binary
    thresh3 = cv2.threshold(img2_cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Apply dilation to make the text thicker
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    dilate3 = cv2.dilate(thresh3, kernel3, iterations=10)
    # Finding the Adaptive threshold
    thresh1 = cv2.adaptiveThreshold(seperate(histogram_equalization(dilate)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,11,2)
    # Finding the Adaptive threshold
    thresh2 = cv2.adaptiveThreshold(seperate(histogram_equalization(dilate3)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,11,2)
        

    # Find keypoints and descriptors for the cropped images
    kp1_cropped, des1_cropped = sift.detectAndCompute(thresh1, None)
    kp2_cropped, des2_cropped = sift.detectAndCompute(thresh2, None)

    # Match descriptors between the two cropped images
    matches_cropped = bf.knnMatch(des1_cropped, des2_cropped, k=2)

    # Apply ratio test to filter good matches
    good_matches_cropped = []
    for m, n in matches_cropped:
        if m.distance < 0.7 * n.distance:
            good_matches_cropped.append(m)

    # Boolean
    Matched = False
    
    # Check if the images are the same
    if len(good_matches_cropped) >= 10:
        Matched = True
    
    # Draw matches on a new image
    result = cv2.drawMatches(thresh1, kp1_cropped,thresh2, kp2_cropped, good_matches_cropped, None)
    
    # Show result image
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return Matched

def mainfunction(img1path ):
    # Decode the image data into OpenCV format
    # img1 = cv2.imread("files/10-front.jpg", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imdecode(np.frombuffer(img1path, np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # return Finding_FrontPage_Keypoints(img1, img2)

    output=""
    imagelist=['files/10-front.jpg','files/100-front.jpg','files/500-front.jpg','files/1000-front.jpg','files/5000-front.jpg']
    # imagelist=['files/100-back.jpg','files/500-back.jpg','files/1000-back.jpg','files/5000-back.jpg']

    img2 = cv2.imdecode(np.frombuffer(img1path, np.uint8), cv2.IMREAD_GRAYSCALE)
    for i in imagelist:
        img1 = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        
        # Finding the Front-Page Keypoints
        Matched = Finding_FrontPage_Keypoints(img1,img2)
        
        if Matched == True:
            output=i
            break
            
    if output != "":        
        numbers = re.findall(r'\d+', output)
        result = int(''.join(numbers))
    else:
        result="UPLOAD GOOD QUALITY IMAGE"

    return result