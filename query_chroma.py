from langchain.embeddings import SentenceTransformerEmbeddings, __all__
from langchain.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# Directory paths
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = AzureOpenAIEmbeddings()  # Assuming default settings

db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

query = "How does Alice meet Mad Hatter?"
matching_docs = db.similarity_search(PROMPT_TEMPLATE, k=3)

for i in matching_docs:
    print(i)
