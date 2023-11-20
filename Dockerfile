# Dockerfile
## Build venv
FROM python:3.9-slim-buster as builder

RUN apt-get update \
    && apt-get install -y \
         curl \
         build-essential \
         libffi-dev \
    && rm -rf /var/lib/apt/lists/*

#ENV POETRY_VERSION=1.1.15
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH /root/.local/bin:$PATH

WORKDIR /app
COPY pyproject.toml poetry.lock ./

RUN python -m venv --copies /app/venv
RUN . /app/venv/bin/activate \
    && poetry install --only main


# Stage 2: Final image
FROM python:3.9-slim-buster

WORKDIR /app

# Copy the built venv from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy the project files
COPY . ./


HEALTHCHECK --start-period=30s --interval=20s --timeout=15s CMD curl --fail http://localhost:8080/health || exit 1

ENV service_url=redis
ENV redis_port=6379

# Run Streamlit
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

