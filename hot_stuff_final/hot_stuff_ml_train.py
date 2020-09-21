from pathlib import Path
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import decomposition
import pickle
import matplotlib.pyplot as plt


# This is to track the execution time
start_time = time.time()

#Global variables that define the directories of this program
HOT_FACES_DATA_DIR = Path("hot_stuff_data")

# Variables for the filenames
hs_facial_ratios_filename = "hs_facial_ratios.txt"
hs_facial_labels = "hs_facial_ratings.txt"
hs_ml_model = "hs_ml_model.sav"
hs_ml_model_pca = "hs_ml_model_pca.sav"

# Using numpy, the files are loaded
hs_all_facial_ratio = np.loadtxt(HOT_FACES_DATA_DIR/hs_facial_ratios_filename, delimiter=',')
hs_all_ratings = np.loadtxt(HOT_FACES_DATA_DIR/hs_facial_labels, delimiter=',')

# Split the facial ratios and respective ratings into two sets using train_test_split: training and testing
# hs_fr_training_set hs_ratings_training_set => training data
# hs_fr_testing_set hs_ratings_testing_set => testing data
hs_fr_training_set, hs_fr_testing_set, hs_ratings_training_set, hs_ratings_testing_set = train_test_split(hs_all_facial_ratio, hs_all_ratings, test_size=0.05, random_state=1)

# 
pca = decomposition.PCA(n_components=20)
#pca.fit(hs_fr_training_set)
#hs_fr_training_set = pca.transform(hs_fr_training_set)
hs_fr_training_set = pca.fit_transform(hs_fr_training_set)
hs_fr_testing_set = pca.transform(hs_fr_testing_set)

# 
regr = linear_model.LinearRegression()
regr.fit(hs_fr_training_set, hs_ratings_training_set)
ratings_predict = regr.predict(hs_fr_testing_set)

pickle.dump(regr, open(HOT_FACES_DATA_DIR/hs_ml_model, "wb"))
pickle.dump(pca, open(HOT_FACES_DATA_DIR/hs_ml_model_pca, "wb"))

print("--- %s seconds ---" % (time.time() - start_time))

corr = np.corrcoef(ratings_predict, hs_ratings_testing_set)[0, 1]
print(corr)

residue = np.mean((ratings_predict - hs_ratings_testing_set) ** 2)
print(residue)

rangeArray = np.arange(1, len(hs_fr_testing_set)+1)
plt.plot(rangeArray, hs_ratings_testing_set, 'r', rangeArray, ratings_predict, 'b')
plt.show()