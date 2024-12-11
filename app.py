import streamlit as st
from chat import query_engine



# Function to maintain the conversation state
def run_chatbot():
    # Initialize chat history (a list of message pairs)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display the chat history
    for message in st.session_state.messages:
        message_type = message['type']
        message_text = message['text']
        
        if message_type == 'user':
            st.write(f"**You:** {message_text}")
        elif message_type == 'bot':
            st.write(f"**Chatbot:** {message_text}")
    
    # User input box
    user_input = st.text_input("You:", key="user_input")
    
    if user_input:
        # Add the user message to the session state
        st.session_state.messages.append({"type": "user", "text": user_input})
        
        # Get the response from query_engine
        bot_response = query_engine.query(user_input)
        
        # Add the bot's response to the session state
        st.session_state.messages.append({"type": "bot", "text": bot_response})
        
        # Clear the input box after sending the message
        st.session_state.user_input = ""

# Main Streamlit app layout
def main():
    st.title("Chatbot Interface")
    
    run_chatbot()

if __name__ == "__main__":
    main()
