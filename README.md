# Telesto Model API Template Project

## Prepare Model Archive

* Add weights of a model you developed to the project (see `example_model/model.pt`)
* Add code for inference (see `example_model/model.py`). The model inference class should inherit `telesto.model.ClassificationModelBase` (for classification tasks).
* Write a custom Dockerfile for the model (see `example_model/Dockerfile`) where you
    - Install all the needed dependencies
    - Copy the inference code and model weights into the image
    - If needed, override the `MODEL_CLASS` and `MODEL_PATH` environment variables
* Prepare an archive with all needed files `zip -r example_model.zip example_model`

## Build Docker Image and Start Container
Building the image containing the model:
```
docker build -t example/model-api -f example_model/Dockerfile example_model
```

Running the container:
```
docker run -p 9876:9876 --name model-api --rm example/model-api
```

## Call Model API using cURL

```
curl http://localhost:9876/

curl -X POST -H "Content-Type: application/json" --data-binary @data/example-input.json http://localhost:9876/
```

## Evaluate Model Running in Container

```
pip install -r requirements-dev.txt

python -m evaluate --api-url http://localhost:9876 --dataset-path data/test-dataset --metric accuracy
```
