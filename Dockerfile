FROM  python:3.9-alpine

ENV LC_ALL en_US.utf8
ENV LANG en_US.utf8

# Update pip and setuptools
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy our code to the image
COPY ./src /app
WORKDIR /app

# Server variables
ENV SERVER_ADDRESS="0.0.0.0"
ENV SERVER_PORT=7000

EXPOSE ${SERVER_PORT}

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s\
    CMD curl --fail http://localhost:${SERVER_PORT}/healthcheck || exit 1

# Need this to run commands everytime container is started
ENTRYPOINT uvicorn main:app --reload \
    --host ${SERVER_ADDRESS} \
    --port ${SERVER_PORT} \
    --workers 8

