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