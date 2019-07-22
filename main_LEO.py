#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from functions import file_functions


#----------PARÁMETROS-------
frames = 200
secuencia = []

#==============================================================================
#------------                   LOAD                           ----------------
#==============================================================================

for i in range(90000,90000+frames):
    secuencia.append(cv2.imread( str('./inputs/microscopio/') + str(i) + str('.png')))

'''
for i in range(frames):
    secuencia.append(cv2.imread( str('./inputs/simulated') + str(i) + str('.png')))'''
#==============================================================================
#------------                  ITERACIONES                   ----------------
#==============================================================================

#Primer Frame
frame = secuencia[0]
#Obtengo los blobs
key_points = file_functions.find_blobs(frame, 0, True)
#Guardo el frame con los puntos detectados
detected_blobs = cv2.drawKeypoints(frame, key_points, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(str('./outs/detected_blobs/frame0.png'), detected_blobs)
coordinates_list_past = []
for keypoint in key_points:
    coordinates_list_past.append(np.array(keypoint.pt))

#Inicializo lista de tracks

tracks = []
for i in range(len(coordinates_list_past)):     # Agrego una track por cada coordenada inicial    
    tracks.append(np.ones([frames,2])*-1)   # Cada track es un numpy array con filas como frames y columnas x e y 
    tracks[i][0,:] = coordinates_list_past[i]   # Las coordenadas en el resto de los frame son -1 -1
    

#ITERO
for it in range(1, frames):
    #estudio el frame actual
    frame = secuencia[it]
    #Obtengo los blobs
    key_points = file_functions.find_blobs(frame, it, True)
    
    #Guardo el frame con los puntos detectados
    detected_blobs = cv2.drawKeypoints(frame, key_points, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str('./outs/detected_blobs/frame') + str(it) + str('.png'), detected_blobs)
    coordinates_list_actual = []
    for keypoint in key_points:
        coordinates_list_actual.append(np.array(keypoint.pt))

    #Obtengo lista de asignaciones
    assignments = file_functions.get_assignments(coordinates_list_past,coordinates_list_actual)    
    #Guardo en la lista de tracksack
    tracks = file_functions.update_tracks(tracks, coordinates_list_past, coordinates_list_actual, assignments, it)
    coordinates_list_past = coordinates_list_actual.copy()
#==============================================================================
#------------                PROCESO INFO OBTENIDA             ----------------
#==============================================================================  

for it in range(frames):
    for i in range(it):
        for track in tracks:
            if track[i,0]>0 and track[i-1,0]>0:
                cv2.line(secuencia[it], (int(track[i-1,0]),int(track[i-1,1]) ), (int(track[i,0]),int(track[i,1]) ), (255,255,255))
    cv2.imwrite(str('./outs/tracking/frame') + str(it) + str('.png'), secuencia[it])
    
