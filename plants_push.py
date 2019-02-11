import numpy as np
import cv2
import matplotlib.pyplot as plt


# read in image
img = cv2.imread('house_plant3.jpg')
n_height = 200
n_width = 200
img = cv2.resize(img, (n_width, n_height), interpolation=cv2.INTER_LINEAR)
img = cv2.GaussianBlur(img, (5, 5), 3)

#get image shape
height = img.shape[0]
width = img.shape[1]

pixels = []
# extract RGB pixel values
for i in range(height):
    for j in range(width):
        pixels.append(img[i, j])


green = []

lum = []
green_lum = []

#iterate through 
for i in range(len(pixels)):
    total = (pixels[i][0] + pixels[i][1] + pixels[i][2])
    
    green.append(pixels[i][1] / (total + 1))
    green_temp = pixels[i][1] / (total + 1)
    
    
    lum.append((0.33 * pixels[i][0]) + (0.5 * pixels[i][1]) +
               (0.16 * pixels[i][2]))
   



# plot brightness against green-ness
plt.scatter(lum, green, edgecolors='black')

plt.xlabel('brightness')
plt.ylabel('green-ness')
plt.show()