#!/bin/bash

# Start Redis Server
redis-server &

# Start UVicorn Server
uvicorn main:app &

# Function to check the readiness of the /health endpoint
check_readiness() {
  local max_retries=30
  local retry_interval=5
  for ((i = 0; i < $max_retries; i++)); do
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    if [ "$response" = "200" ]; then
      return 0  # Success, readiness confirmed
    fi
    sleep $retry_interval
  done
  return 1  # Readiness check failed
}

# Call the readiness check function
check_readiness

if [ $? -eq 0 ]; then
  # If readiness check passed, run Streamlit App
  echo "Readiness check passed. Starting Streamlit App..."
  # Replace 'streamlitapp.py' with the actual filename
  streamlit run streamlitapp.py
else
  # If readiness check failed, exit with an error message
  echo "Readiness check failed. Streamlit App not started."
  exit 1
fi
