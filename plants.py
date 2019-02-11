import numpy as np
import cv2
import matplotlib.pyplot as plt
# from sklearn import svm
from sklearn.svm import LinearSVC
import pickle


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# read in test image
t_img = cv2.imread('house_plant2.jpg')

# read in image
img = cv2.imread('house_plant3.jpg')
n_height = 200
n_width = 200
img = cv2.resize(img, (n_width, n_height), interpolation=cv2.INTER_LINEAR)
img = cv2.bilateralFilter(img, 9, 75, 75)
# img = cv2.GaussianBlur(img, (5, 5), 3)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# make labels for image
sensitivity = 25
color_green = 60
lower_green = np.array([color_green - sensitivity, 100, 100])
upper_green = np.array([color_green + sensitivity, 255, 255])

mask = cv2.inRange(img_hsv, lower_green, upper_green)


def gen_labels(mask):
    height = mask.shape[0]
    length = mask.shape[1]
    labels = []
    for i in range(height):
        for j in range(length):
            if mask[i, j] == 0:
                labels.append(0)
            else:
                labels.append(1)

    return labels


labels = gen_labels(mask)

#get image shape
height = img.shape[0]
width = img.shape[1]

pixels = []
# extract RGB pixel values
for i in range(height):
    for j in range(width):
        pixels.append(img[i, j])

red = []
green = []
blue = []
lum = []
green_lum = []

for i in range(len(pixels)):
    total = (pixels[i][0] + pixels[i][1] + pixels[i][2])
    blue.append(pixels[i][0] / (total + 1))
    green.append(pixels[i][1] / (total + 1))
    green_temp = pixels[i][1] / (total + 1)
    red.append(pixels[i][2] / (total + 1))
    lum_temp = (
        (0.33 * pixels[i][0]) + (0.5 * pixels[i][1]) + (0.16 * pixels[i][2]))
    lum.append((0.33 * pixels[i][0]) + (0.5 * pixels[i][1]) +
               (0.16 * pixels[i][2]))
    green_lum.append([float(lum_temp), float(green_temp)])

green_lum = np.array(green_lum)
plt.imshow(mask)
plt.show()
# fig, ax = plt.subplots()

# clf = LinearSVC()
# clf.fit(green_lum, labels)
# print(clf.predict([[100, 100], [0, 0], [200, 200]]))
# # xx, yy = make_meshgrid(np.array(lum), np.array(green))
# # plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# plt.scatter(lum, green, edgecolors='black')

# plt.xlabel('brightness')
# plt.ylabel('green-ness')
# plt.show()