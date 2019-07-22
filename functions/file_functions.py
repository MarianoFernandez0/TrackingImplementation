#!/usr/bin/python
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

#=============================================================================
#   DETECCION
#=============================================================================
#Devuelvo lista de blobs detectados.
#=============================================================================
def find_blobs(img, index, save = False ):
    #Filtro de media para sacar puntos ruidosos
    img_filtered = cv2.blur(img, (10,10))
    _ , segmented = cv2.threshold(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Guardo segmentacion
    if save:
        cv2.imwrite(str('./outs/segmentation/frame') + str(index) + str('.png'), segmented)
    #instancia
    detector = cv2.SimpleBlobDetector_create()    
    #Detecto
    keypoints = detector.detect(255 - segmented)
        
    return keypoints


#==============================================================================
#                                 Devuelvo                                    # 
#==============================================================================
def get_assignments(coordinates_list_past, coordinates_list_actual, maxCost = 30):
    #Calculo la matriz costo según la norma 2
    cost = np.zeros([len(coordinates_list_past),len(coordinates_list_actual)])
    for i in range(len(coordinates_list_past)):
        for j in range(len(coordinates_list_actual)):
            cost[i,j] = np.linalg.norm(coordinates_list_past[i]-coordinates_list_actual[j],2)
    #Resulevo asignaciones con matriz de costos
    row_ind, col_ind = linear_sum_assignment(cost) 
    #Tengo que discriminar aquellas que sean mayores al costo minimo
    final_assignation = []

    for i in range(len(row_ind)):
        if(cost[row_ind[i],col_ind[i]] < maxCost):
            final_assignation.append([row_ind[i], col_ind[i]])

    return final_assignation




#==============================================================================
#                               Actualizo lista de tracks                     # 
#==============================================================================
def update_tracks(tracks, coordinates_list_past, coordinates_list_actual, assignments, frame):
#Toma la lista de tracks y actualiza los tracks viejos o agrega nuevos 
    
    for k in range(len(coordinates_list_actual)):           #para cada blob detectado en el futuro lo agrego a un track viejo o nuevo según corresponda
        col_assignments = [row[1] for row in assignments]   #Segunda columna de assignments
        if k in col_assignments:
            coord_actual = coordinates_list_past[assignments[col_assignments.index(k)][0]]    #Coordenada actual a la que corresponde la coordenada futura
            
            col_coord_actual = np.zeros([len(tracks),2])     #un array con todos las coordenadas de los puntos actuales
            for i in range(len(tracks)):
                col_coord_actual[i,:] = tracks[i][frame-1,:] 
            if len(np.where(col_coord_actual==coord_actual)[0]):
                index_coord_actual = np.where(col_coord_actual==coord_actual)[0][0] #index de la track que está la coordenada buscada
                tracks[index_coord_actual][frame,:] = coordinates_list_actual[k]    #agrega la coordenada del frame futuro a la track
        else:
            tracks.append(np.ones(tracks[0].shape)*-1)                          #crea una track nueva si no hay assignemt
            tracks[len(tracks)-1][frame,:] = coordinates_list_actual[k]
    return tracks