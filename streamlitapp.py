import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import time 


# Load the base URL from the environment variable
fastapi_base_url = os.getenv("FASTAPI_BASE_URL", "http://52.53.163.66:80")

page_icon = Image.open("logos/Business Logo.png")

st.set_page_config(page_title="Chat with our furniture finder", page_icon=page_icon, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with our furniture finder")

chat_icon = Image.open("logos/Chatbot Logo.png")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help?"}
    ]

prompt = st.chat_input("Your question")
if prompt: # Prompt for user input and save it to chat history
    with st.spinner('Looking for products that match your description. This might take a minute...'):
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
form_id = 0 # initializing id for form key 
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.write(message["content"])
    elif message["role"] == "image":
        st.markdown('<span style="font-size: 15px;">Products based on your inputs:</span>', unsafe_allow_html=True)
        with st.expander("", expanded=True):  
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
                    href = product_details.get("href", "")
                    caption = f"{title}: ${price}"
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        image_bytes = BytesIO(image_response.content)
                        image = Image.open(image_bytes)

                        container_html = f'<div><h6>{caption}</h6><a href="{href}" target="_blank" style="display: block;"><img src="{image_url}" alt="{caption}" /><div></div></a></div><br>'
                        # Render the HTML container
                        st.markdown(container_html, unsafe_allow_html=True)
                        # col.write(f"{title}: ${price}")
                        # col.image(image, caption=caption, use_column_width=True)
                    else:
                        col.write("Failed to load image response.")
                start_idx = end_idx
        # if idx == len(st.session_state.messages) - 1:
        form_id += 1
        form_key = f"feedback_form_{form_id}"
        form = st.form(key=form_key, clear_on_submit=True)
        with form:
            form.write("Did you find what you were looking for?")
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                thumbs_up_clicked = st.form_submit_button("👍 Yep!")
            with col2:
                thumbs_down_clicked = st.form_submit_button("👎 Not quite")
                st.markdown(
                            """
                        <style>
                            div[data-testid="column"] {
                                border: 1px solid red;
                            }
                        </style>
                        """,
                            unsafe_allow_html=True,
                        )
            if thumbs_up_clicked:
                form.empty()
                form.success("Thanks for your feedback!")
            elif thumbs_down_clicked:
                form.empty()
                form.write("Sorry, that wasn't helpful. Do you mind providing more details on what you're looking for?")
    else:
        with st.chat_message(message["role"], avatar=chat_icon):
            st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = assistant_response["content"]  # Get the assistant's response from the dictionary
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)  # Add response to message history


st.markdown(
    """
<style>
    div[data-testid="column"] div[data-testid="baseButton-secondaryFormSubmit"] {
        border: 1px solid red;
    }
</style>
""",
    unsafe_allow_html=True,
)