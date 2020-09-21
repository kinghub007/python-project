import face_recognition
import os
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import itertools
import pickle
from sklearn import linear_model
from sklearn import decomposition
import numpy

def calculate_facial_ratio(f_coord, s_coord, t_coord, fth_coord):
	dist1 = math.sqrt((f_coord[0]-s_coord[0])**2 + (f_coord[1]-s_coord[1])**2)
	dist2 = math.sqrt((t_coord[0]-fth_coord[0])**2 + (t_coord[1]-fth_coord[1])**2)

	if dist2 == 0:
		dist2 = 1
 
	ratio = dist1/dist2

	return ratio

def save_result(hs_image, facial_landmarks, hs_score, hs_filename):
    #This is to view graphically the points in the images
    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(hs_image)
    hs_size = pil_image.size
    image_draw = ImageDraw.Draw(pil_image)
    
    # Drawing lines for each facial landmark
    for facial_feature in facial_landmarks.keys():
        # First, the points are drawn
        for point in facial_landmarks[facial_feature]:
            image_draw.point(point, fill=(255,0,0,0))
        # Second, the line joining them
        image_draw.line(facial_landmarks[facial_feature], width=5)

    # Load the font to write the text
    font = ImageFont.truetype("OpenSans-Regular.ttf", size=32)
    image_draw.text((5,hs_size[1]-50), "Predicted Rating: "+ str(hs_score), fill=(0,0,0), font=font)

    # Save the new image
    pil_image.save(hs_filename)
    print("Image saved")

#Global variables that define the directories and filenames
HOT_FACES_TEST_DIR = Path("hot_stuff_test")
HOT_FACES_DATA_DIR = Path("hot_stuff_data")
hs_ml_model_filename = "hs_ml_model.sav"
hs_ml_model_pca_filename = "hs_ml_model_pca.sav"

#Loading the machine learning model
print("Loading machine learning model")
hs_ml_model = linear_model.LinearRegression()
hs_ml_pca = decomposition.PCA(n_components=25)
hs_ml_model = pickle.load(open(HOT_FACES_DATA_DIR/hs_ml_model_filename, 'rb'))
hs_ml_pca = pickle.load(open(HOT_FACES_DATA_DIR/hs_ml_model_pca_filename, 'rb'))

print("Loading test faces")

#A loop to load all the images located the test directory
for hot_filename in os.listdir(HOT_FACES_TEST_DIR):
    #Current file being processed
    print("Processing: "+str(HOT_FACES_TEST_DIR/hot_filename))

    #Image is loaded
    hs_image = face_recognition.load_image_file(f'{HOT_FACES_TEST_DIR}/{hot_filename}')

    #The facial landmarks(location of chin, nose, etc) are obtained
    hs_landmarks = face_recognition.face_landmarks(hs_image, model="large")
    print("Facial landmarks extracted")

    facial_ratios = list()
    # The facial ratios are calculated
    print("Calculating facial ratios for "+str(HOT_FACES_TEST_DIR/hot_filename))
    for keys in hs_landmarks[0]:
        values = hs_landmarks[0][keys]
        values_combos = itertools.combinations(values, 4)

        for combo in values_combos:
            facial_ratios.append(calculate_facial_ratio(combo[0], combo[1], combo[2], combo[3]))

    # The list is transformed to an array
    facial_ratios = numpy.asarray(facial_ratios).reshape(1,-1)

    # The ratios are reduced using PCA
    facial_ratios = hs_ml_pca.transform(facial_ratios)

    # The prediction from the ml is calculated
    hs_result = hs_ml_model.predict(facial_ratios)
    hs_result = round(hs_result[0], 2)

    # The results are saved to the image
    save_result(hs_image, hs_landmarks[0], hs_result, f'{HOT_FACES_TEST_DIR}/{"pred_"+hot_filename}')
    print("Done, saving results")