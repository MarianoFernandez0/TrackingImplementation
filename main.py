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
	#img = secuencia[0]

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

def track(coordinates_list_actual,coordinates_list_future):
    #Calculo la matiz costo según la norma 2
    cost = np.zeros([len(coordinates_list_actual),len(coordinates_list_future)])
    tracks = np.zeros([len(coordinates_list_actual),2])
    for i in range(len(coordinates_list_actual)):
        tracks[i,0] = i
        for j in range(len(coordinates_list_future)):
            cost[i,j] = np.linalg.norm(coordinates_list_actual[i]-coordinates_list_future[j],2)
        if min(cost[i,:])>30:
            tracks[i,1] = -1
        else:
            tracks[i,1] = np.argmin(cost[i,:]) 
    
    return tracks

#==============================================================================
#Primero trabajo con una imagen - PRIMERA IMPLEMENTACION SIN FILTRO DE KALMAN # 
#==============================================================================
img = secuencia[0]
key_points = find_blobs(img)

detected_blobs = cv2.drawKeypoints(img, key_points, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

coordinates_list_actual = []
for keypoint in key_points:
	coordinates_list_actual.append(np.array(keypoint.pt))

#Tomo frame nuevo
img = secuencia[1]
key_points = find_blobs(img)

detected_blobs_2 = cv2.drawKeypoints(detected_blobs, key_points, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

coordinates_list_future = []
for keypoint in key_points:
	coordinates_list_future.append(np.array(keypoint.pt))


#dibujo las trtacks de las primeras dos imagenes
tracks = track(coordinates_list_actual,coordinates_list_future)    

for i in range(len(coordinates_list_actual)):
    if tracks[i,1]>0:
        cv2.line(detected_blobs_2, (int(coordinates_list_actual[i][0]),int(coordinates_list_actual[i][1])), (int(coordinates_list_future[int(tracks[i,1])][0]),int(coordinates_list_future[int(tracks[i,1])][1])),(0,255,0))
cv2.imshow('Tracks',detected_blobs_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
