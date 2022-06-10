import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
ggblur = cv2.GaussianBlur(img, (3, 3),0)
gblur = cv2.GaussianBlur(img, (5, 5), 0)



titles = ['image', 'GaussianBlur 3x3', 'GaussianBlur 5x5']
images = [img, ggblur, gblur,]

for i in range(3):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()