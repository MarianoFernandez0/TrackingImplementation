#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from  file_functions import find_blobs,get_assignments
#----------PARÁMETROS-------
frames = 200
secuencia = []

#------------LOAD----------------
for i in range(90000,90000+frames):
	secuencia.append(cv2.imread( str('./inputs/') + str(i) + str('.png')))
#==============================================================================
#Primero trabajo con una imagen - PRIMERA IMPLEMENTACION SIN FILTRO DE KALMAN # 
#==============================================================================
img = secuencia[1]
key_points = find_blobs(img)

detected_blobs = cv2.drawKeypoints(img, key_points, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

coordinates_list_actual = []
for keypoint in key_points:
	coordinates_list_actual.append(np.array(keypoint.pt))

#Tomo frame nuevo
img = secuencia[6]
key_points = find_blobs(img)

detected_blobs_2 = cv2.drawKeypoints(detected_blobs, key_points, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

coordinates_list_future = []
for keypoint in key_points:
	coordinates_list_future.append(np.array(keypoint.pt))


#dibujo las trtacks de las primeras dos imagenes
assignments = get_assignments(coordinates_list_actual,coordinates_list_future)    
print(assignments)

for i in range(len(coordinates_list_actual)):
    if assignments[i,1]>0:
        cv2.line(detected_blobs_2, (int(coordinates_list_actual[i][0]),int(coordinates_list_actual[i][1])), (int(coordinates_list_future[int(assignments[i,1])][0]),int(coordinates_list_future[int(assignments[i,1])][1])),(0,255,0))
cv2.imshow('Tracks',detected_blobs_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
