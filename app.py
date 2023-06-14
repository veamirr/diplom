# uvicorn app:app --reload
import base64
import io
import numpy as np
import cv2
import symbol_classification
import features
import class2_preprocessing
import pickle
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

class Item(BaseModel):
    id: int
    name: str

app = FastAPI()
data = pickle.load(open('classification-model.pickle', 'rb'))
prepr = pickle.load(open('pre_classes.pickle', 'rb'))

@app.post("/images/get_initial_predictions/", response_model=list[Item])
async def get_initial_area(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img_features = np.array(features.features(img))

    if prepr.predict([img_features]) == 2:
        cntrs, y_max, y_min = class2_preprocessing.contour_segmentation(img)
        img = class2_preprocessing.hough_straight_lines(img, cntrs, y_max, y_min)

    predictions, scores = symbol_classification.predict_many(clf=data, binaries=[img], n=3)
    categories = [{"id": i, "name": name} for i, name in enumerate(data.classes_)][0:3]

    return categories
