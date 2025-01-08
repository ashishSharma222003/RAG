import os  # Import os for checking folder existence
from chat import query_engine, faithfullnes, correctness, relevancy
import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime
import shutil

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
Faithfulness measures if the response from a query engine matches any source nodes.\n
Relevancy measures if the response + source nodes match the query.
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
    folder_name = get_user_folder(st.session_state.user_id)
    # print(folder_name)
    folder_exists = folder_name is not None and os.path.exists(folder_name) and os.path.isdir(folder_name)


    if folder_exists and folder_name:
        # Show the "Chat" tab only if the folder exists
        tab1, tab2 = st.tabs(["Chat", "Folder Check"])

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
            st.header("Folder Check")
            st.success(f"The folder '{folder_name}' exists in the current directory!")

            # Add a button to allow the user to delete the folder
            delete_button = st.button("Delete Folder")

            if delete_button:
                # Confirm deletion action
                users = load_users()
                print(user_id)
                if user_id in users:
                            users[user_id]["folder_name"]=None
                            save_users(users)
                
                st.success(f"The folder '{folder_name} is deleted") 
                st.rerun()
    else:
        st.header("Folder Check")
        st.warning(f"The folder '{folder_name}' does not exist.")
        
        # Limit the file upload size to 5MB
        uploaded_file = st.file_uploader("Please upload a PDF to proceed.", type=["pdf"])
        
        if uploaded_file:
            # Check if the file size is less than or equal to 5MB
            if uploaded_file.size <= 5 * 1024 * 1024:  # 5MB in bytes
                file_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded {uploaded_file.name} successfully!")
            else:
                st.error("File size exceeds the 5MB limit. Please upload a smaller file.")
