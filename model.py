import pinecone
import torch
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
import pprint as pprint


__version__="0.0.1"
# Connect to Pinecone
API_KEY="API_KEY"
ENV="us-west4-gcp"
pinecone.init(api_key=API_KEY, environment=ENV)

class Qna_System:
    def __init__(self, index_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.index_name = "insurance-question-answering"
        self.index = pinecone.Index(index_name=self.index_name)
        self.retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=self.device)
        self.tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
        self.generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(self.device)


    def generate_answer(self, query):
        # tokenize the query to get input_ids
        inputs = self.tokenizer([query], max_length=1024, return_tensors="pt", padding='max_length', truncation=True)
        # use generator to predict output ids
        ids = self.generator.generate(inputs["input_ids"], num_beams=2, min_length=10, max_length=40)
        # use tokenizer to decode the output ids
        answer = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer


    def query_pinecone(self, query, top_k=1):
        # generate embeddings for the query
        xq = self.retriever.encode([query]).tolist()
        # search pinecone index for context passage with the answer
        xc = self.index.query(xq, top_k=top_k, include_metadata=True)
        return xc


# Query the index and get similar vectors
if __name__ == "__main__":
    qna = Qna_System("abstractive-question-answering")
    # query = "Is theft covered in home insurance?"
    query="Is car insurance mandatory in the US?"

    print(f"structured answer: {qna.generate_answer(query)}")
