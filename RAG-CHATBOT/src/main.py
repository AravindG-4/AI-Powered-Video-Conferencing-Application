from transcriber import transcribe_audio
from vector_store import add_to_vector_db, search_vector_db
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def generate_response(query, retrieved_docs):
    """
    Generates a response using the Groq API based on the provided query and retrieved documents.
    
    :param query: The user's query.
    :param retrieved_docs: List of retrieved documents containing context.
    :return: Generated response as a string.
    """
    context = " ".join([doc["text"] for doc in retrieved_docs])

    try:
        # Perform a chat completion request
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Error communicating with the Groq API: {e}")
        return "Error generating response."

# Main function to run the chatbot
def main():
    audio_path = "audio/speech.wav"  # Path to your audio file
    
    # Step 1: Transcribe audio to text
    print("Transcribing audio...")
    text = transcribe_audio(audio_path)
    print("Transcribed Text:", text)
    
    # Step 2: Add transcribed text to vector database
    print("Adding text to vector database...")
    add_to_vector_db(text, {"source": audio_path})
    
    # Step 3: Query the chatbot
    query = input("Ask a question: ")
    results = search_vector_db(query)
    response = generate_response(query, results)
    print("Chatbot Response:", response)

if __name__ == "__main__":
    main()
  
