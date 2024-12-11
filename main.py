import json
from flask import Flask, request, jsonify
import os
from chat import query_engine
app = Flask(__name__)

# Path to the JSON file where chat history will be saved
CHAT_HISTORY_FILE = 'chat_history.json'

# Helper function to read chat history from the JSON file
def read_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

# Helper function to save chat history to the JSON file
def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(chat_history, f, indent=4)



@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()
        
        # Extract the user's input from the request data
        user_input = data.get('message')
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process the input with your query engine
        bot_response = query_engine.query(user_input)
        
        # Read the existing chat history
        chat_history = read_chat_history()
        
        # Add the new chat interaction to the history
        chat_history.append({"user": user_input, "bot": bot_response})
        
        # Save the updated chat history to the JSON file
        save_chat_history(chat_history)
        
        # Return the bot's response in JSON format
        return jsonify({'response': bot_response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chats', methods=['GET'])
def get_chats():
    try:
        # Retrieve chat history from the JSON file
        chat_history = read_chat_history()
        return jsonify({'chats': chat_history}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
