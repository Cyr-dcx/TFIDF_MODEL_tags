#pour l'api
# import os
# from pathlib import Path
# from typing import List
from fastapi import FastAPI #HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# pour le preprocessing de la question
from utils_package.functions import final_cleaning, read_list

encoder_file = "./target_encoder.sav"
tfidf_file = "./tfidf_encoder.sav"
model_file = "./pipe_OnevsRest.sav"

target_encoder = joblib.load(encoder_file)
pipe = joblib.load(model_file)

def preprocess_pipeline(question):
    question_list = []
    preprocessed_question = final_cleaning(question, token=False)
    question_list.append(str(preprocessed_question))
    return question_list

def generate_prediction(preprocessed_question):
    best_thresholds_1 = read_list('best_thresholds_1')
    y_pred_test_proba = pipe.predict(preprocessed_question)
    y_pred_log_reg = np.empty_like(y_pred_test_proba)
    for i, thresh in enumerate(best_thresholds_1):
        y_pred_log_reg[:, i] = (y_pred_test_proba[:, i] > thresh) * 1
    return y_pred_log_reg

def predict_pipeline(question,target_encoder=target_encoder):
    preprocessed_question = preprocess_pipeline(question)
    predictions = generate_prediction(preprocessed_question)
    tags = target_encoder.inverse_transform(predictions)
    return tags
    
app = FastAPI()

class Phrase(BaseModel):
    phrase: str

sentence_test="I've been making Python scripts for simple tasks at work and never really bothered packaging them for others to use. Now I have been assigned to make a Python wrapper for a REST API. I have absolutely no idea on how to start and I need help.What I have:(Just want to be specific as possible) I have the virtualenv ready, it's also up in github, the .gitignore file for python is there as well, plus, the requests library for interacting with the REST API. That's it.Here's the current directory tree"

@app.get("/")
async def index():
    return {"tags": "Faisons une prédiction"}

@app.post("/predict", status_code=200)
def read_item(one_phrase: Phrase):
    question = one_phrase.phrase
    preprocessed_question = preprocess_pipeline(question)
    predictions = generate_prediction(preprocessed_question)
    tags = target_encoder.inverse_transform(predictions)

    return {"tags": tags}

preprocessed_question = preprocess_pipeline(sentence_test)
predictions = generate_prediction(preprocessed_question)
tags = target_encoder.inverse_transform(predictions)
print(tags)