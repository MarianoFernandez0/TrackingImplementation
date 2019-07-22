#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from functions import file_functions

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

	#instancia
	detector = cv2.SimpleBlobDetector_create()

	#Detecto
	keypoints = detector.detect(255 - segmented)

	#out = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return keypoints


#==============================================================================
#                                 Devuelvo                                    # 
#==============================================================================
def get_assignments(coordinates_list_actual, coordinates_list_future, maxCost = 50):
    #Calculo la matriz costo según la norma 2
    cost = np.zeros([len(coordinates_list_actual),len(coordinates_list_future)])
    for i in range(len(coordinates_list_actual)):
        for j in range(len(coordinates_list_future)):
            cost[i,j] = np.linalg.norm(coordinates_list_actual[i]-coordinates_list_future[j],2)
    #Resulevo asignaciones con matriz de costos
    row_ind, col_ind = linear_sum_assignment(cost) 
    #Tengo que discriminar aquellas que sean mayores al costo minimo
    final_assignation = []

    for i in range(len(row_ind)):
    	if(cost[row_ind[i],col_ind[i]] < maxCost):
    		final_assignation.append([row_ind[i], col_ind[i]])

    return final_assignation

#==============================================================================
#Primero trabajo con una imagen - PRIMERA IMPLEMENTACION SIN FILTRO DE KALMAN # 
#==============================================================================
track_list = []

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


#Obtengo lista de asignaciones
assignments = get_assignments(coordinates_list_actual,coordinates_list_future)    

#Guardo en la lista de tracks
#track_list = update_tracks(track_list)


#Pruebo track
for assignment in assignments:
    cv2.line(detected_blobs, (int(coordinates_list_actual[assignment[0]][0]),int(coordinates_list_actual[assignment[0]][1])), (int(coordinates_list_future[int(assignment[1])][0]),int(coordinates_list_future[int(assignment[1])][1])),(0,255,0))
cv2.imshow('Tracks',detected_blobs_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
