import streamlit as st
from PIL import Image


page_icon = Image.open("logos/ShopBuddy.png")

st.set_page_config(page_title="About Us", page_icon=page_icon, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("About Us")

st.markdown(
    """
    ### Our Mission
    We are reimagining how users shop online and aim to simply the tiresome furniture shopping journey

    ### Meet the team
"""
)

image_data = [
    {"name": "Suna Leloglu", "email": "suna@berkeley.edu", "path": "logos/suna_leloglu.png"},
    {"name": "Kisha Kim", "email": "kisha.kim@berkeley.edu", "path": "logos/kisha_kim.png"},
    {"name": "Shuo Wang", "email": "shuo.wang@ischool.berkeley.edu", "path": "logos/shuo_wang.png"},
    {"name": "Harshita Shangari", "email": "hshangari@berkeley.edu ", "path": "logos/harshita_shangari.png"},
    {"name": "Jujhar Bedi", "email": "jujhar.bedi@berkeley.edu", "path": "logos/jujhar_bedi.png"},
    {"name": "Siddharth Ashokkumar", "email": "sidashok@berkeley.edu", "path": "logos/sid_ashokkumar.png"},
]

# Display images in two rows and three columns
rows = st.columns(3)

for i, data in enumerate(image_data):
    image_name = data["name"]
    email = data["email"]
    image_path = data["path"]

    # Display image name above the image
    rows[i % 3].write(f"**{image_name}**")

    # Display image
    rows[i % 3].image(Image.open(image_path), caption=email, use_column_width=True)

    # # Add a non-breaking space between the rows
    # if (i + 1) % 3 == 0 and i != len(image_data) - 1:
    #     st.markdown("&nbsp;HIIII", unsafe_allow_html=True)
    
st.markdown(
    """
    ### Who are we
    We are a group of graduate students in the MIDS program at UC Berkeley who teamed up together on a capstone project to bring Shop Buddy to life.
"""
)