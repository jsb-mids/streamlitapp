# Streamlit app


How to run the code

## Step 1: Install Poetry

You can install Poetry using the following `curl` command. Make sure you're in the root directory of the application. Open your Terminal and run:

```bash
curl -sSL https://install.python-poetry.org | python -
```


## Step 2: Install Dependencies
```bash
poetry install
```

## Step 3: Create Virtual Env
```bash
poetry shell
```

## Step 4 (optional): If you don't have the pickle files downloaded. Please download them and add them to a /data folder inside the root directory.

Link to the pickle files: https://ucbischool.slack.com/archives/C05QVDG4BGF/p1698040389967369

## Step 4: Run the code.
In three separate terminals run the following commands. Make sure that you create the poetry virtual env in each terminal firs via the poetry shell command.
```bash
uvicorn main:app --host 127.0.0.1 --port 8080 --reload

redis-server

streamlit run streamlitapp.py
```