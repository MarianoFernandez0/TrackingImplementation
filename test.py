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
def find_blobs(img,diam):
	#img = secuencia[0]

	#Filtro de media para sacar puntos ruidosos
	img_filtered = cv2.blur(img, (diam,diam))
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
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
diam = 14
key_points = find_blobs(img,diam)

key_points_draw = key_points
detected_blobs = cv2.drawKeypoints(img, key_points_draw, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Blobs',detected_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

coordinates_list_actual = []
#for keypoint in key_points:
#	coordinates_list_actual.append(np.array(keypoint.pt))
for i in range(len(key_points)):
	coordinates_list_actual.append(np.array(key_points[i].pt))



#Tomo frame nuevo
img = secuencia[1]

key_points_1 = find_blobs(img,diam)
coordinates_list_future = []

key_points_draw = key_points_1
detected_blobs = cv2.drawKeypoints(img, key_points_1, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Blobs',detected_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
#for keypoint in key_points:
#	coordinates_list_future.append(np.array(keypoint.pt))

for i in range(len(key_points)):
	coordinates_list_future.append(np.array(key_points[i].pt))

Cost = np.zeros([len(coordinates_list_actual),len(coordinates_list_future)])

for i in range(len(coordinates_list_actual)):
    for j in range(len(coordinates_list_future)):
        Cost[i,j] = np.linalg.norm(coordinates_list_actual[i]-coordinates_list_future[j],2)
        i 
print(Cost.shape)