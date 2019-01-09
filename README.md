# Color-Based-Classification
Color based classification of plant images in order to track growth over time.

This program uses a linear support vector machine to recognize the color green in a picture of a plant. The use case of this program is in a automated garden, where it can track the growth of the plant over time, based on the size of the leaves and also possibly detect environmental stresses such as lack of water, which would be inicated by changes in the color of the leaves over time. This work is a loose implementation of this paper --> (

# Running the program

# Training the SVM
The SVM was trained by feeding in pixel data from the following plant. 

![](https://github.com/rohin-dasari/Color-Based-Classification/blob/master/images/house_plant3.jpg)

A mask was made by converting the image from the BGR to the HSV color space and finding all the green regions. The resulting binary mask served the purpose of acting as labels for the training data. 

![](

In order to attain the training data, the BGR (OpenCV uses BGR rather than RGB) pixel values were extracted from the image and the green values were taken and divided by the sum of the blue, green, and red values to attain a value that represents the amount of green in a pixel relative to the other pixel intensity values. By plotting this against the brightness of the pixel intensities, it becomes clear where there are bright green areas, dull green areas, and areas where there isn't much green at all. Our goal is to isolate as much of the green as we can and calculate the bounding area around our green region. If the plant is growing, over time, this bounding area should grow. If the plant is dying or experiencing drought stress, this bounding area should be decreasing. The effects of gravitropism are ignored for the most part because the current system in place for the garden involves a constant source of sunlight in a fixed place.  
