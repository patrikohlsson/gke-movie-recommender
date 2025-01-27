export PROJECT_ID=<project-id>
export REGION=europe-north1

wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip

gcloud config set project $PROJECT_ID

gcloud services enable container.googleapis.com \
     containerregistry.googleapis.com \
     cloudbuild.googleapis.com

gcloud container clusters create-auto my-python-cluster --region $REGION

docker build -t python-gke-app .

docker tag python-gke-app gcr.io/$PROJECT_ID/python-gke-app:v1

gcloud builds submit --tag gcr.io/$PROJECT_ID/python-gke-app:v1 .

# Apply configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get deployments
kubectl get pods

# Get external IP
kubectl get service python-service

# Clean up
# # Delete the cluster
# gcloud container clusters delete my-python-cluster --region $REGION

# # Delete the container image
# gcloud container images delete gcr.io/$PROJECT_ID/python-gke-app:v1
