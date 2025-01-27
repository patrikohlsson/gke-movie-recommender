FROM python:3.12-slim

WORKDIR /app

# Install flask
RUN pip3 install Flask numpy pandas --no-cache-dir

# Install pytorch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

COPY ./ml-latest-small/movies.csv ./movies.csv
COPY ./ml-latest-small/ratings.csv ./ratings.csv
COPY ./app.py ./app.py
COPY ./baseline.pt ./baseline.pt

CMD ["python", "app.py"]
