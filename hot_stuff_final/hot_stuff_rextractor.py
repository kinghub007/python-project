import os, fnmatch
import pandas as pd
from pathlib import Path

#Defining directory to save the ratings

HOT_FACES_DATA_DIR = Path("hot_stuff_data")
RATINGS_TXT = "hs_facial_ratings.txt"

#Declaring the file path and reading the main file.
DATA_DIR = Path('SCUT-FBP5500_v2')
DATA_FILE = Path('All_Ratings.xlsx')

print("Processing started")
xls = pd.ExcelFile(DATA_DIR/DATA_FILE)


#For matching string
lst = []
list_files = os.listdir('hot_stuff_images')
pat = "*.jpg"
for files in list_files:
    if fnmatch.fnmatch(files, pat):
        img = files
        for elem in [img]:
            lst.append(elem)
str1 = '|'.join(lst)

#For Asian Female
#Read n numbers of images with Filename and Rating
df1 = pd.read_excel(xls, 'Asian_Female', usecols = 'B,C')
sp1 = df1[df1['Filename'].str.match(str1)]

#Sort according to the names
af = sp1.sort_values('Filename')
#Getting Average and rounding to 2 decimal place
r1 = af.groupby('Filename').mean().round(2)
#print(r1)
#Saving the avg into txt file
avg_af = r1['Rating'].tolist()
with open(HOT_FACES_DATA_DIR/RATINGS_TXT, 'w+') as af_file:
    af_file.write('\n'.join(str(item) for item in avg_af))
    af_file.write('\n')
    
#For Asian Male
df2 = pd.read_excel(xls, 'Asian_Male', usecols = 'B,C')
sp2 = df2[df2['Filename'].str.match(str1)]
am = sp2.sort_values('Filename')
r2 = am.groupby('Filename').mean().round(2)
#print(r2)
avg_am = r2['Rating'].tolist()
with open(HOT_FACES_DATA_DIR/RATINGS_TXT, 'a') as am_file:
    am_file.write('\n'.join(str(item) for item in avg_am))
    am_file.write('\n')
    
#For Caucasian Female
df3 = pd.read_excel(xls, 'Caucasian_Female', usecols = 'B,C')
sp3 = df3[df3['Filename'].str.match(str1)]
cf = sp3.sort_values('Filename')
r3 = cf.groupby('Filename').mean().round(2)
#print(r3)
avg_cf = r3['Rating'].tolist()
with open(HOT_FACES_DATA_DIR/RATINGS_TXT, 'a') as cf_file:
    cf_file.write('\n'.join(str(item) for item in avg_cf))
    cf_file.write('\n')
    
#For Caucasian Male
df4 = pd.read_excel(xls, 'Caucasian_Male', usecols = 'B,C')
sp4 = df4[df4['Filename'].str.match(str1)]
cm = sp4.sort_values('Filename')
r4 = cm.groupby('Filename').mean().round(2)
#print(r4)
avg_cm = r4['Rating'].tolist()
with open(HOT_FACES_DATA_DIR/RATINGS_TXT, 'a') as cm_file:
    cm_file.write('\n'.join(str(item) for item in avg_cm))

print("Images processed and calculated average ratings are saved into a text file.")



