from flask import Flask, render_template, request, jsonify, session
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session management

# Initialize the RAG system
def initialize_rag():
    # Settings control global defaults
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Groq(model="llama3-70b-8192", api_key="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")

    # Create a RAG tool using LlamaIndex
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    return query_engine

# Initialize the RAG system
query_engine = initialize_rag()

# Initialize chat history in session if it doesn't exist
def init_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []

@app.route('/')
def home():
    init_chat_history()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        init_chat_history()
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'response': 'Please provide a message.'}), 400
        
        # Get response from RAG system
        response = query_engine.query(user_message)
        response_text = str(response)
        
        # Add message to chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session['chat_history'].append({
            'timestamp': timestamp,
            'user': user_message,
            'bot': response_text
        })
        
        # Keep only the last 50 messages to prevent memory issues
        if len(session['chat_history']) > 50:
            session['chat_history'] = session['chat_history'][-50:]
        
        # Save the session
        session.modified = True
        
        return jsonify({
            'response': response_text,
            'history': session['chat_history']
        })
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    session.modified = True
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True) 