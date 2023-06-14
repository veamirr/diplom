import pickle
import symbol_classification
import cv2
import os

data = pickle.load(open('classification-model.pickle', 'rb'))

path = r'C:\Users\Vera Mironova\Desktop\diplom2\initials\class1_bin'
arr = os.listdir(path)

img_list = []
for file in arr:
   img = cv2.imread(f'C:\\Users\\Vera Mironova\\Desktop\\diplom2\\initials\\class1_bin\\{file}', cv2.IMREAD_GRAYSCALE)
   img = cv2.bitwise_not(img)
   img_list.append(img)


predictions = symbol_classification.predict_many(clf=data, binaries=img_list, n=1)
#print(predictions)
print(data.classes_[predictions])
