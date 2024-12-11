from chat import query_engine,faithfullnes ,correctness ,relevancy# Ensure query_engine is correctly imported
import streamlit as st
import pandas as pd  # Import pandas for creating and displaying the table
import time
# Set Streamlit app title
st.title("RAG")

st.markdown("""
Faithfulness measure if the response from a query engine matches any source nodes.\n
Relevancy measure if the response + source nodes match the query.
""")
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

    # Generate response from the query engine
    with st.chat_message("assistant"):
        try:
            # Query the engine using the prompt
            start=time.time()
            assistant_reply = query_engine.query(prompt)  # Assuming this is a string response
            end=time.time()
            # time.sleep(2)
            # Add the response to chat
            st.markdown(assistant_reply.response)
            faith=faithfullnes(assistant_reply)
            relevannt=relevancy(prompt,assistant_reply)
            df = pd.DataFrame(
                [
                    {"Evaluator": "Relevancy(Pass or fail)", "rating":  "Pass" if relevannt.passing else "Fail"},
                    {"Evaluator": "Faithfulness(pass or fail)", "rating": "Pass" if faith.passing else "Fail"},
                    {"Evaluator": "Latency(sec)", "rating": end-start},
                ]
            )

            st.dataframe(df, use_container_width=True,hide_index=True)
            
        except Exception as e:
            # Handle any errors in the query process
            st.markdown(f"Sorry, I encountered an error: {e}")
    
    # Add assistant response to message history
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})


