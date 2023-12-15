import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon=Image.open("logos/ShopBuddy.png"),
)
page_icon = Image.open("logos/ShopBuddy.png")

st.write("# ShopBuddy: Reimagine your shopping experience!")

st.markdown(
    """
    **👈 Navigate to Chatbot on the sidebar** to see what ShopBuddy can do!
    ### ShopBuddy is a smart AI shopping expert
    Our Chatbot provides a personalized experience as it interacts with Natural Language through the power of LLM, fine-tuned on industry-specific furniture data and image retrieval, returns relevant images & actively changes search results based on user input through text-to-image CLIP model. We package it up with an interactive user interface that is easy to use by providing users with clickable furniture items for a smooth purchasing experience. We aim to solve identified gaps, such that online shoppers will: 
    - Have a consistent and simple user experience irrespective of the furniture retailer
    - Be able to converse and express their needs to quickly find what they are looking for
    
    ### Problem and Motivation 
    In the realm of artificial intelligence, the transformational impact is widespread, and one of the areas where this evolution is distinctly felt is the area of online shopping. Presently, users navigate traditional websites using search queries and filters, often encountering a cumbersome and less-than-optimal buying experience.  
    
    We found opportunities for value adds: 
    - Enhance user shopping experience through interactive chatbot leveraging Generative AI
    - Opportunity to Integrate large language model (LLM) in a B2C market

    Our project centers on harnessing the power of Large Language Models (LLMs) to elevate and streamline the furniture shopping experience. This focus is particularly pertinent given the complex nature of furniture shopping, involving multi-dimensional and multi-model search challenges.

    

"""
)