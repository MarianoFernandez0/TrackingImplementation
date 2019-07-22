import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment


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



def get_assignments(coordinates_list_actual, coordinates_list_future, minCost = 30):
    #Calculo la matriz costo según la norma 2
    cost = np.zeros([len(coordinates_list_actual),len(coordinates_list_future)])
    tracks = np.zeros([len(coordinates_list_actual),2])
    for i in range(len(coordinates_list_actual)):
        tracks[i,0] = i
        for j in range(len(coordinates_list_future)):
            cost[i,j] = np.linalg.norm(coordinates_list_actual[i]-coordinates_list_future[j],2)
        if min(cost[i,:])>minCost:
            tracks[i,1] = -1
        else:
            tracks[i,1] = np.argmin(cost[i,:]) 
    
    return tracks