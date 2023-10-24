# How to run:
    1. Install Poetry:
        curl -sSL https://install.python-poetry.org | python3 -

    2. Create Poetry Shell in terminal by navigating to the Chatbot dolder and running:
        poetry shell

    3. You may need to specifiy the path if you get "Poetry not found error". Just run this code if you have a Mac:
        export PATH="$HOME/.local/bin:$PATH"

    4. Install the dependencies by running the following command : 
        poetry install
    
    5. Run the Streamlit app:
        streamlit run streamlitapp.py 

 