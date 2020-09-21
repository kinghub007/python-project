import face_recognition
import os
import math
from pathlib import Path
import itertools
import numpy

def calculate_facial_ratio(f_coord, s_coord, t_coord, fth_coord):
	dist1 = math.sqrt((f_coord[0]-s_coord[0])**2 + (f_coord[1]-s_coord[1])**2)
	dist2 = math.sqrt((t_coord[0]-fth_coord[0])**2 + (t_coord[1]-fth_coord[1])**2)

	ratio = dist1/dist2

	return ratio

#Global variables that define the directories of this program
HOT_FACES_DIR = Path("hot_stuff_images")
HOT_FACES_DATA_DIR = Path("hot_stuff_data")

ratios_list = list()

#Create empty file to save facial landmarks coordinates
hs_facial_extractor_log = "hs_facial_ext_log.txt"
hs_facial_ratios_filename = "hs_facial_ratios.txt"
open(HOT_FACES_DATA_DIR/hs_facial_ratios_filename, 'w+').close()
open(HOT_FACES_DATA_DIR/hs_facial_extractor_log, 'w+').close()

#A simple print to indicate that the program is working
print("Loading faces")
#print(sorted(os.listdir(HOT_FACES_DIR)))

log_file = open(HOT_FACES_DATA_DIR/hs_facial_extractor_log, "a+")
#A loop to load all the images located in a specific directory
for hot_filename in sorted(os.listdir(HOT_FACES_DIR)):
    hs_landmarks_coord = list()
    #A print to show the current file being processed
    #print("Processing: "+str(HOT_FACES_DIR/hot_filename))
    log_file.write("\nProcessing: "+str(HOT_FACES_DIR/hot_filename))

    #The image is loaded using the load_image_file method
    hs_image = face_recognition.load_image_file(f'{HOT_FACES_DIR}/{hot_filename}')
    #print("Extracting facial landmarks for "+str(HOT_FACES_DIR/hot_filename))
    log_file.write("\nExtracting facial landmarks for "+str(HOT_FACES_DIR/hot_filename))
    #The facial landmarks(location of chin, nose, etc) are loaded and stored in a list
    hs_landmarks = face_recognition.face_landmarks(hs_image, model="large")
    
    if not hs_landmarks:
        #print("!!!!!!!!!!!!!!!!!!!!!!! Unable to obtain the facial landmarks from "+str(HOT_FACES_DIR/hot_filename))
        log_file.write("\n!!!!!!!!!!!!!!!!!!!!!!! Unable to obtain the facial landmarks from "+str(HOT_FACES_DIR/hot_filename))
    else:
        temp_list = list()
        # The process for extracting all the facial landmarks coordinates and storing them in a list
        #print("Calculating facial ratios for "+str(HOT_FACES_DIR/hot_filename))
        log_file.write("\nCalculating facial ratios for "+str(HOT_FACES_DIR/hot_filename))

        for keys in hs_landmarks[0]:
            values = hs_landmarks[0][keys]
            values_combos = itertools.combinations(values, 4)

            for combo in values_combos:
                temp_list.append(calculate_facial_ratio(combo[0], combo[1], combo[2], combo[3]))
        
        # Calculate the ratios for all the facial landmarks coordinates
        #facial_combinations = itertools.combinations(hs_landmarks_coord, 4)
        
        #for combination in facial_combinations:
        #    temp_list.append(calculate_facial_ratio(combination[0], combination[1], combination[2], combination[3]))
        ratios_list.append(temp_list)

print("Data processing done, saving results")
log_file.write("\nData processing done, saving results")
log_file.close()
numpy.savetxt(HOT_FACES_DATA_DIR/hs_facial_ratios_filename, ratios_list, delimiter=',', fmt = '%.04f')


