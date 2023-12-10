import streamlit as st
from PIL import Image


page_icon = Image.open("logos/ShopBuddy.png")

st.set_page_config(page_title="Demo", page_icon=page_icon, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("How to Use ShopBuddy")

st.markdown(
    """
    **👈 Navigate to Chatbot on the sidebar** to see what ShopBuddy can do!
"""
)

video_path = "StreamlitDemo_Preso3.mov"

# Display the video
st.video(open(video_path, "rb").read())