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

## Step 4: Run the code.
In three separate terminals run the following commands. Make sure that you create the poetry virtual env in each terminal firs via the poetry shell command.
```bash
uvicorn main:app

redis-server

streamlit run streamlitapp.py
```