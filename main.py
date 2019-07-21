#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt

#----------PARÁMETROS-------
frames = 200
secuencia = []

#------------LOAD----------------
for i in range(90000,90000+frames):
	secuencia.append(cv2.imread( str('./inputs/') + str(i) + str('.png')))

#=============================================================================
#=============================================================================
#Primero trabajo con una imagen
img = cv2.imread('./inputs/images.png')
img = secuencia[0]
_ , img = cv2.threshold(cv2.cvtColor(secuencia[0], cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


#Parametros del detector
params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 200

#Creo instancia de detector
detector = cv2.SimpleBlobDetector_create()
#Detecto blobs
keypoints = detector.detect(img)
print(keypoints)

keypoints_draw = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(keypoints_draw)
plt.show()
