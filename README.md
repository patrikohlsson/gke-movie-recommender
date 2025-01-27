# Dummy Movie Recommender System

## Description

This project contains a simple movie recommender system based on collaborative filtering and the MovieLens dataset.
It can optionally be deployed on Google Kubernetes Engine (GKE) and serve recommendations via an external ip.

## Usage

To run locally, simply install docker and run

```bash
docker build -t movie-recommender .
docker run -it --rm -p 8080:8080 movie-recommender
```

To deploy on GKE add the project id and region to setup.sh and run it.

```bash
./setup.sh
```

## Training

`train-baseline.py` contains a simple baseline model trained on a subset of the MovieLens dataset.

It uses matrix factorization to predict ratings for unseen movies.

