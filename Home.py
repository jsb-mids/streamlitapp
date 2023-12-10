import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon=Image.open("logos/ShopBuddy.png"),
)
page_icon = Image.open("logos/ShopBuddy.png")

st.write("# Welcome to ShopBuddy!")

st.markdown(
    """
    **👈 Navigate to Chatbot on the sidebar** to see what ShopBuddy can do!
    ### ShopBuddy is a smart AI shopping expert
    Our Chatbot provides a personalized experience as it interacts with Natural Language through the power of LLM, fine tuned on industry specific furniture data and retrieves images, returns relevant images & actively changes search results based on user input through text -to-image CLIP model. We package it up with an interactive user interface that is easy to use by providing users with clickable furniture items to provide a smooth purchasing experience. We aim to solve identified gaps, such that online shoppers will: 
    - Have a consistent and simple user experience irrespective of the furniture retailer
    - Be able to converse and express their needs to quickly find what they are looking for
    
    ### Problem and Motivation 
    In the realm of artificial intelligence, the transformational impact is widespread, and one of the areas where this evolution is distinctly felt is the area of online shopping. Presently, users navigate through traditional websites using search queries and filters, often encountering a cumbersome and less-than-optimal buying experience.  

    Recognizing the potential for improvement, our project centers on harnessing the power of Large Language Models (LLMs) to elevate and streamline the furniture shopping experience. This focus is particularly pertinent given the complex nature of furniture shopping, involving multi-dimensional and multi-model search challenges.

    While there is no dearth of online furniture shopping platforms today, shoppers often have to:

    - Navigate through website experiences on different platforms of varying sophistication
    - Browse through a large inventory of items themselves to find a match
    - Walk into physical store(s) to continue their research before they finally make a purchase

"""
)