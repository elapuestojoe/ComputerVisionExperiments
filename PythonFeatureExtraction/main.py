import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Joe\\Pictures\\CircleSquare.png", 0)

#Threshold image
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Connected components
# connectivity = 4
# output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# num_labels = output[0]
# print("Number of objects: ", num_labels)

# labels = output[1]
# stats = output[2]
# centroids = output[3]

# for stat in stats:
#     print(stat)
# for label in range(num_labels):
#     print("Element: ", label)
#     stat = stats[label]
#     print("Area: ", stats[label][cv2.CC_STAT_AREA])

# Contour approach
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(thresh, contours, -1, (128,255,128), 3)

print("Number of objects: ", len(contours)-1)
for i in range(1, len(contours)):
    currentContour = contours[i]
    print("Element: ", i)
    print("Area: ", cv2.contourArea(currentContour))
    print("Perimenter: ", cv2.arcLength(currentContour, True))
    moments = cv2.moments(currentContour)
    print("Moments: ", moments)
    cx = int(moments["m10"]/moments["m00"])
    cy = int(moments["m01"]/moments["m00"])
    print("Centroid: ", [cx, cy])
    print("Hu moments: ", cv2.HuMoments(moments).flatten())

    cv2.fourierDescriptor(currentContour)
cv2.imshow("Contour Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()