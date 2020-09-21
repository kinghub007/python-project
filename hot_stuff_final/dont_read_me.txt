Welcome to Hot Stuff v1.0
This python program uses the face_recognition and sklearn libraries to analyze face and predict how 'how' or attractive they are.
HS is composed of 4 programs, that need to be executed in the following order:
1. hot_stuff_fextractor: this program extracts the facial ratios from the images in the hot_stuff_images folder. Note: some images cannot be processed by the face_recognition library. Check the log in the hot_stuff_ datafolder to remove such images before executing hot_stuff_rextractor.
2. hot_stuff_rextractor: calculates the average ratings for the faces from the spreadsheet located in the SCUT-FBP5500_v2 folder and saves the results in the hot_stuff_data folder.
3. hot_stuff_ml_train: uses PCA and linear regression to build a ML model using the inputs from the previous two programs  
4. hot_stuff_train: predicts the attractiveness of the faces stored in the hot_stuff_test folder.