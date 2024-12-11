from chat import query_engine
import streamlit as st

# Set Streamlit app title
st.title("RAG")



if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for the user's message
if prompt := st.chat_input("Ask something..."):
    # Add user message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show user message in chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response from OpenAI model
    with st.chat_message("assistant"):
        
        # Extract the response content
        assistant_reply = query_engine.query(prompt)
        
        # Stream the assistant's response in chunks (optional for more interaction)
        # For now, we will directly add the response to the chat history
        st.markdown(assistant_reply)
    
    # Add assistant response to message history
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
