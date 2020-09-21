'''
|-----------------------------------------------
| Statistical Descriptions of Image
|-----------------------------------------------
| - Histogram Equalization
| - Plotting
|-----------------------------------------------
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert dari BGR jadi GRAY
h = img.shape[0]
w = img.shape[1]

gray_counter = np.zeros(256, dtype=int)
for i in range(h):
    for j in range(w):
        gray_counter[gray[i][j]] += 1

equ = cv2.equalizeHist(gray) # Histogram equalization
equ_counter = np.zeros(256, dtype=int)
for i in range(h):
    for j in range(w):
        equ_counter[equ[i][j]] += 1

# Plotting with matplotlib
plt.figure(1, figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(gray_counter, 'r', label='Before')
plt.legend(loc='upper left')
plt.ylabel('quantity')
plt.xlabel('intensity')
plt.axis([0, 256, 0, gray_counter.max() if (gray_counter.max() > equ_counter.max()) else equ_counter.max()])

plt.subplot(2, 1, 2)
plt.plot(equ_counter, 'b', label='After')
plt.legend(loc='upper left')
plt.ylabel('quantity')
plt.xlabel('intensity')
plt.axis([0, 256, 0, gray_counter.max() if (gray_counter.max() > equ_counter.max()) else equ_counter.max()])
plt.show()

# Show image result
res = np.hstack((gray, equ))
cv2.imshow('Image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()