import os  # Import os for checking folder existence
from chat import initialize_query_engine
import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime
import shutil
from ocr import process_pdf_files_in_directory
from document_embedder import create_and_save_user_indices
import tempfile
from voicebot import speak_text,recognize_audio
# Load users from the JSON file
def load_users():
    try:
        with open("users.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"admin": {"password": "password", "folder_name": "storage_key"}} 

# Get the folder name for a specific user
def get_user_folder(user_id):
    users = load_users()
    if user_id in users:
        return users[user_id]["folder_name"]
    return None
def check_user_directories(user_id: str, base_dir: str = "./user_data"):
    # Construct the paths for vector and keyword directories
    user_dir = os.path.join(base_dir, user_id)
    vector_store_dir = os.path.join(user_dir, "vector_index")
    keyword_store_dir = os.path.join(user_dir, "keyword_index")

    # Check if both directories exist
    vector_exists = os.path.exists(vector_store_dir) and os.path.isdir(vector_store_dir)
    keyword_exists = os.path.exists(keyword_store_dir) and os.path.isdir(keyword_store_dir)

    if vector_exists and keyword_exists:
        print(f"Both 'vector_index' and 'keyword_index' directories exist for user: {user_id}")
        return True  # Both directories exist
    else:
        if not vector_exists:
            print(f"'{vector_store_dir}' does not exist for user: {user_id}")
        if not keyword_exists:
            print(f"'{keyword_store_dir}' does not exist for user: {user_id}")
        return False  # One or both directories are missing

# Load chat history from the JSON file
def load_chats():
    try:
        if os.path.exists("chats.json") and os.path.getsize("chats.json") > 0:
            with open("chats.json", "r") as file:
                return json.load(file)
        else:
            return {}  # Return an empty dictionary if the file doesn't exist or is empty
    except json.JSONDecodeError:
        # In case of a corrupted JSON file, return an empty dictionary
        return {}

# Save chat history to the JSON file
def save_chats(chat_data):
    with open("chats.json", "w") as file:
        json.dump(chat_data, file, indent=4)

# Get current date in YYYY-MM-DD format
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

def save_users(users):
    with open("users.json", "w") as file:
        json.dump(users, file, indent=4)

# Set Streamlit app title
st.title("RAG")

st.markdown("""
RAG Chatbot with hybrid search.
""")

# Login system using session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None

if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        users = load_users()
        if username in users and password == users[username]["password"]:
            st.session_state.logged_in = True
            st.session_state.user_id = username
            print(username)
            st.success(f"Login successful! Welcome, {username}.")
        else:
            st.error("Invalid username or password.")
else:
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False, "user_id": None}))
    st.sidebar.success(f"Logged in as: {st.session_state.user_id}")

    # Load chat history
    chat_history = load_chats()
    user_id = st.session_state.user_id
    current_date = get_current_date()

    if user_id not in chat_history:
        chat_history[user_id] = {}
    if current_date not in chat_history[user_id]:
        chat_history[user_id][current_date] = []

    # Retrieve the folder name for the logged-in user
    # folder_name = get_user_folder(st.session_state.user_id)
    # # print(folder_name)
    # folder_exists = folder_name is not None and os.path.exists(folder_name) and os.path.isdir(folder_name)


    if check_user_directories(user_id):
        # Show the "Chat" tab only if the folder exists
        tab1, tab2, tab3 = st.tabs(["Chat", "Folder Check", "Voice Chat"])

        with tab1:
            st.header("Chat Functionality")

            # Display previous chat messages
            for message in chat_history[user_id][current_date]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Input field for the user's message
            if prompt := st.chat_input("Ask something..."):
                user_message = {"role": "user", "content": prompt}
                chat_history[user_id][current_date].append(user_message)

                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate assistant's response
                with st.chat_message("assistant"):
                    try:
                        start = time.time()
                        query_engine=initialize_query_engine(user_id=user_id)
                        assistant_reply = query_engine.query(prompt)
                        end = time.time()
                        
                        assistant_message = {"role": "assistant", "content": assistant_reply.response}
                        chat_history[user_id][current_date].append(assistant_message)

                        st.markdown(assistant_reply.response)
                    except Exception as e:
                        error_message = {"role": "assistant", "content": f"Error: {e}"}
                        chat_history[user_id][current_date].append(error_message)
                        st.markdown(f"Sorry, I encountered an error: {e}")

                # Save chat history
                save_chats(chat_history)

        with tab2:
            st.header("Embeddings Check")
            st.success(f"Embeddings for '{user_id}' exists in the database!")

            # Limit the file upload size to 5MB
            uploaded_file = st.file_uploader("Please upload a PDF to proceed.", type=["pdf"])
            
            if uploaded_file:
                # Check if the file size is less than or equal to 5MB
                if uploaded_file.size <= 5 * 1024 * 1024:  # 5MB in bytes
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

                        # Save the uploaded file to the temporary directory
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f"Uploaded {uploaded_file.name} successfully!")

                        # Process the uploaded file
                        destination_dir=process_pdf_files_in_directory(user_id=user_id, source_dir=temp_dir)

                        # Create indices for the processed files
                        create_and_save_user_indices(user_id=user_id, input_dir=destination_dir)

                        st.success("File processed and indices created successfully!. Please restart this platform.")
                else:
                    st.error("File size exceeds the 5MB limit. Please upload a smaller file.")
                # st.rerun()
        with tab3:
            st.header("Voice Chatbot")
            st.info("This tab allows you to interact with the chatbot using your voice.")

            # Check if embeddings exist for the user
            user_id = st.session_state.user_id
            if check_user_directories(user_id):
                if st.button("Start Voice Interaction"):
                    # Recognize audio input
                    user_query = recognize_audio()
                    if user_query:
                        st.success(f"You said: {user_query}")
                        with st.spinner("Generating response..."):
                            try:
                                # Initialize query engine
                                query_engine = initialize_query_engine(user_id=user_id)
                                response = query_engine.query(user_query)
                                st.markdown(f"**Response:**  {response.response}")
                                
                                # Speak the response
                                speak_text(f"  {response.response}")
                            except Exception as e:
                                st.error(f"Error generating response: {e}")
            else:
                st.warning("Embeddings for this user do not exist. Please upload a file in the 'Folder Check' tab to create embeddings.")

    else:
        st.header("Embeddings Check")
        st.success(f"Embeddings for '{user_id}' exists in the database!")
        
        # Limit the file upload size to 5MB
        uploaded_file = st.file_uploader("Please upload a PDF to proceed.", type=["pdf"])
        
        if uploaded_file:
            # Check if the file size is less than or equal to 5MB
            if uploaded_file.size <= 5 * 1024 * 1024:  # 5MB in bytes
                # file_path = os.path.join(os.getcwd(), uploaded_file.name)
                with tempfile.TemporaryDirectory() as temp_dir:
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

                        # Save the uploaded file to the temporary directory
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.success(f"Uploaded {uploaded_file.name} successfully!")

                        # Process the uploaded file
                        destination_dir=process_pdf_files_in_directory(user_id=user_id, source_dir=temp_dir)

                        # Create indices for the processed files
                        create_and_save_user_indices(user_id=user_id, input_dir=destination_dir)

                        st.success("File processed and indices created successfully!. Please restart this platform.")
            else:
                st.error("File size exceeds the 5MB limit. Please upload a smaller file.")
