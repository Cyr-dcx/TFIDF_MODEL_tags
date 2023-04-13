#pour l'api
# import os
# from pathlib import Path
# from typing import List
import joblib
from fastapi import FastAPI #HTTPException
from pydantic import BaseModel
import sys

# pour le preprocessing de la question
from utils_package.functions import final_cleaning

encoder_file = "./target_encoder.sav"
tfidf_file = "./tfidf_encoder.sav"
model_file = "./tfidf_lightGBM.sav"

target_encoder = joblib.load(encoder_file)
tfidf = joblib.load(tfidf_file)
model = joblib.load(model_file)


# def preprocess_pipeline(question):
#     question_list = []
#     preprocessed_question = final_cleaning(question, token=False)
#     question_list.append(str(preprocessed_question))
#     X_tfidf = tfidf.transform(question_list)
#     return X_tfidf

# def generate_prediction(preprocessed_question, my_model=model):
#     tags = my_model.predict(preprocessed_question)
#     return tags

# def predict_pipeline(question,target_encoder=target_encoder):
#     preprocessed_question = preprocess_pipeline(question)
#     predictions = generate_prediction(preprocessed_question)
#     tags = target_encoder.inverse_transform(predictions)
#     return tags
    
app = FastAPI()

class Phrase(BaseModel):
    phrase: str

# class Tags(BaseModel):
#     tags: List[str]

sentence_test="I've been making Python scripts for simple tasks at work and never really bothered packaging them for others to use. Now I have been assigned to make a Python wrapper for a REST API. I have absolutely no idea on how to start and I need help.What I have:(Just want to be specific as possible) I have the virtualenv ready, it's also up in github, the .gitignore file for python is there as well, plus, the requests library for interacting with the REST API. That's it.Here's the current directory tree"

@app.get("/")
def index():
    return {"tags": "Faisons une prédiction"}

# @app.post("/predict", status_code=200)
# def read_item(one_phrase: Phrase):
#     question = one_phrase.phrase
#     preprocessed_question = preprocess_pipeline(question)
#     predictions = generate_prediction(preprocessed_question)
#     tags = target_encoder.inverse_transform(predictions)

#     return {"tags": tags}
