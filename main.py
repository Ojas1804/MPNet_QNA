from fastapi import FastAPI
from pydantic import BaseModel
from model import Qna_System
from fastapi.middleware.cors import CORSMiddleware
from model import __version__ as model_version
import json
import uvicorn
from CosineSimilarity import CosineSimilarity


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
    similar_docs: list[str] = []

@app.get("/")
def home():
    return {"health_check": "OK", "version": model_version}

@app.post("/answer", response_model=TextOut)
async def answer(text: TextIn):
    input_data = text.json()
    input_dictionary = json.loads(input_data)
    query = input_dictionary["text"]
    # print(query)
    qna = Qna_System("abstractive-question-answering")
    cs = CosineSimilarity(query)

    ls = []
    for doc in cs.get_similar_documents():
        ls.append(doc[0]+" : "+doc[1])
    answer = qna.generate_answer(query)
    return {"text": answer, "similar_docs": ls}


if __name__ == "__main__":
    uvicorn.run("app:main", host="0.0.0.0", port=8000, reload=True)