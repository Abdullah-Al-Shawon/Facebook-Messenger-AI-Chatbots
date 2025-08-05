import os
import requests
from dotenv import load_dotenv
from flask import Flask, request
import google.generativeai as genai
import pinecone

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

app = Flask(__name__)

def get_gemini_embedding(text):
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error creating Gemini Embedding: {e}")
        return None

def generate_answer(question, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Based on the provided information, give a simple and friendly answer to the following question. If you cannot find the answer from the information, say "Sorry, I don't have information on this topic." Do not provide any irrelevant answers.

        Information (Context):
        ---
        {context}
        ---

        Question: {question}
        
        Answer:
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating Gemini Answer: {e}")
        return "Sorry, I am unable to provide an answer at the moment."

def send_message(recipient_id, message_text):
    params = {
        "access_token": PAGE_ACCESS_TOKEN
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_text
        }
    }
    r = requests.post("https://graph.facebook.com/v18.0/me/messages", params=params, headers=headers, json=data)
    if r.status_code != 200:
        print(f"Failed to send message to Facebook: {r.status_code} {r.text}")

@app.route('/webhook', methods=['GET'])
def webhook_verify():
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200
    return "Hello world", 200

@app.route('/webhook', methods=['POST'])
def webhook_handle():
    data = request.get_json()
    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                if messaging_event.get("message"):
                    sender_id = messaging_event["sender"]["id"]
                    message_text = messaging_event["message"].get("text")

                    if message_text:
                        question_vector = get_gemini_embedding(message_text)
                        
                        if question_vector:
                            search_results = index.query(
                                vector=question_vector,
                                top_k=3,
                                include_metadata=True
                            )
                            
                            context = ""
                            if search_results.get('matches'):
                                for match in search_results['matches']:
                                    context += match['metadata'].get('text', '') + "\n\n"
                            
                            final_answer = generate_answer(message_text, context)
                            send_message(sender_id, final_answer)

    return "ok", 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
