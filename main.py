#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

#----------PARÁMETROS-------
frames = 200
secuencia = []

#------------LOAD----------------
for i in range(90000,90000+frames):
	secuencia.append(cv2.imread( str('./inputs/') + str(i) + str('.png')))



#=============================================================================
#   DETECCION
#=============================================================================
#Devuelvo lista de blobs detectados.
#=============================================================================
def find_blobs(img):
	img = secuencia[0]

	#Filtro de media para sacar puntos ruidosos
	img_filtered = cv2.blur(img, (10,10))
	_ , segmented = cv2.threshold(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	#parametros
	params = cv2.SimpleBlobDetector_Params()
	#instancia
	detector = cv2.SimpleBlobDetector_create()

	#Detecto
	keypoints = detector.detect(255 - segmented)

	#out = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return keypoints





#==============================================================================
#Primero trabajo con una imagen - PRIMERA IMPLEMENTACION SIN FILTRO DE KALMAN # 
#==============================================================================
img = secuencia[0]
key_points = find_blobs(img)
coordinates_list_actual = []
for keypoint in key_points:
	coordinates_list_actual.append(np.array(keypoint.pt))

#Tomo frame nuevo
img = secuencia[1]
key_points = find_blobs(img)
coordinates_list_future = []
for keypoint in key_points:
	coordinates_list_future.append(np.array(keypoint.pt))


