import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Load the base URL from the environment variable
fastapi_base_url = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Chat with our furniture finder", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with our furniture finder")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help?"}
    ]

prompt = st.chat_input("Your question")
if prompt: # Prompt for user input and save it to chat history
    # Use the base URL to construct the complete URL
    chat_url = f"{fastapi_base_url}/chat"
    resp = requests.post(chat_url, json={"content": prompt})
    if resp.status_code == 200:
        data = resp.json()
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)

        assistant_response = {"role": "assistant", "content": data["response"]}
        # Add assistant's text response
        st.session_state.messages.append(assistant_response)

        product_details_list = data.get("product_details", [])  # Get product details as a list
        if product_details_list:
            image_messages = {"role": "image", "content": product_details_list}
            st.session_state.messages.append(image_messages)
            
        

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.write(message["content"])
    elif message["role"] == "image":
        with st.expander("Products based on your inputs:", expanded=True):  
            row_images = message["content"]
            cols = st.columns(4)  # Display images in 4 columns grid
            
            # Calculate how many images should be displayed in each column
            images_per_col = len(row_images) // 4
            remainder = len(row_images) % 4
            
            start_idx = 0
            for i, col in enumerate(cols):
                end_idx = start_idx + images_per_col + (1 if i < remainder else 0)
                for idx in range(start_idx, end_idx):
                    product_details = row_images[idx]
                    image_url = product_details.get("url", "")
                    title = product_details.get("title", "")
                    price = product_details.get("price", "")
                    caption = f"{title}: ${price}"
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        image_bytes = BytesIO(image_response.content)
                        image = Image.open(image_bytes)
                        col.image(image, caption=caption, use_column_width=True)
                    else:
                        col.write("Failed to load image response.")
                start_idx = end_idx
        # if idx == len(st.session_state.messages) - 1:
        form_key = f"feedback_form_{idx}"
        with st.form(key=form_key, clear_on_submit=True):
            st.write("Was this helpful?")
            thumbs_up_clicked = st.form_submit_button("👍 Thumbs Up")
            thumbs_down_clicked = st.form_submit_button("👎 Thumbs Down")
            if thumbs_up_clicked:
                st.write("You selected: Thumbs Up")
            elif thumbs_down_clicked:
                st.write("You selected: Thumbs Down")
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = assistant_response["content"]  # Get the assistant's response from the dictionary
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)  # Add response to message history
