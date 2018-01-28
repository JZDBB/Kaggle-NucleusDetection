import cv2
import os
import numpy as np

a = os.listdir('.\\masks')
add_img = []
for i in range(len(a)):
    path = '.\\masks\\' + a[i]
    img = cv2.imread(path)
    img.astype(np.float32)
    add_img.append(img)
    # if i == 1:
    #     sum_img = img
    # else:
    #     sum_img = sum_img + img
sum_img = sum(k for k in add_img)
# for m in range(len(sum_img)):
#     for n in range(len(sum_img[m])):
#         if sum_img[m][n] > 255:
#             print("= =")
print(np.max(sum_img))