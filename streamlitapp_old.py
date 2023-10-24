import os
import streamlit as st
import requests


# Load the base URL from the environment variable
fastapi_base_url = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Chat with our furniture finder", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with our furniture finder")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help?"}
    ]

prompt = st.chat_input("Your question")
if prompt: # Prompt for user input and save to chat history
    # Use the base URL to construct the complete URL
    chat_url = f"{fastapi_base_url}/chat"
    resp = requests.post(chat_url, json={"content": prompt}) 
    if resp.status_code == 200:
        data = resp.json()
        assistant_response = data.get("response", "No response received")
        st.session_state.messages.append({"role": "user", "content": prompt})
        # st.session_state.messages.append({"role": "assistant", "content": assistant_response})


for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("assistant_response:", assistant_response)
            response = assistant_response
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history



