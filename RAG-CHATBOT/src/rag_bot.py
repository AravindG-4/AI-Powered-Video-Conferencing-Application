from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

def generate_response(query, retrieved_docs):
    context = " ".join([doc["text"] for doc in retrieved_docs])
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    response = llm({"prompt": f"Context: {context}\n\nQuestion: {query}\nAnswer:"})
    return response["choices"][0]["text"]
