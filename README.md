# telesto.ai Model API Template Project

## Prepare Model Archive

* Add weights of a model you developed to the project
* Add code for inference. The model inference class should inherit `telesto.model.ClassificationModelBase` 
(for classification tasks). For more information, see [https://github.com/telesto-ai/telesto-base](the telesto-base module),
which is where the base class is defined.
* Write a custom Dockerfile for the model where you
    - Install all the needed dependencies
    - Copy the inference code and model weights into the image
* Prepare an archive with all needed files `zip -r model.zip model`

[A concrete and worked out example can be found here.](https://github.com/telesto-ai/telesto-base/tree/develop/tests/example_model)
For a detailed guide, see [https://docs.telesto.ai](https://docs.telesto.ai)!

## Build Docker Image and Start Container
Building the image containing the model:
```
docker build -t telestoai/model-api -f model/Dockerfile model
```

Running the container:
```
docker run -p 9876:9876 --name model-api --rm telestoai/model-api
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
