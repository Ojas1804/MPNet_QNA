from fastapi import FastAPI
from pydantic import BaseModel
from model import Qna_System
from fastapi.middleware.cors import CORSMiddleware
from model import __version__ as model_version
import json

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

class TextOut(BaseModel):
    text: str

@app.get("/")
def home():
    return {"health_check": "OK", "version": model_version}

@app.post("/answer", response_model=TextOut)
def answer(text: TextIn):
    input_data = text.json()
    input_dictionary = json.loads(input_data)
    qna = Qna_System("abstractive-question-answering")
    query = input_dictionary["text"]
    print(query)
    answer = qna.generate_answer(query)
    return {"text": answer}