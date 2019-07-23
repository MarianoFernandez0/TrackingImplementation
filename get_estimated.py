def get_estimated(coordinates_list_past,tracks,frame):       
    coordinates_list_estimated = []
    
    for k in range(len(coordinates_list_past)):
        col_track_past = np.zeros([len(tracks),2])     #un array con todos las coordenadas de los puntos actuales
        for i in range(len(tracks)):
            col_track_past[i,:] = tracks[i][frame-1,:] 
        if len(np.where(col_track_past==coordinates_list_past[k])[0]):
            index_track_past = np.where(col_track_past==coordinates_list_past[k])[0][0] #index de la track que está la coordenada buscada

        if tracks[index_track_past][frame-2,0]>0:

            d = tracks[index_track_past][frame-1,:]-tracks[index_track_past][frame-2,:]   
                
            coordinates_list_estimated.append(tracks[index_track_past][frame-1,:]+d)
            
        else:
            coordinates_list_estimated.append(tracks[index_track_past][frame-1,:])
    return coordinates_list_estimated
