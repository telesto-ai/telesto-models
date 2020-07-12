# telesto.ai Model API Template Project
This serves as a brief guide to help you submit your solutions for telesto.ai competitions. For a complete guide, see the [technical documentation](https://docs.telesto.ai/)!

## Preparing model archive

* Add weights of a model you developed to the project
* Add code for inference. The model inference class should inherit `telesto.model.ClassificationModelBase` 
(for classification tasks). For more information, see [the telesto-base module](https://github.com/telesto-ai/telesto-base),
which is where the base class is defined.
* Write a custom Dockerfile for the model where you
    - Install all the needed dependencies
    - Copy the inference code and model weights into the image
* Prepare an archive with all needed files `zip -r model.zip model`

[A concrete and worked out example can be found here.](https://github.com/telesto-ai/telesto-models/tree/master/example_model)
For a detailed guide, see [https://docs.telesto.ai](https://docs.telesto.ai)!

## Building Docker image and starting the container
Building the image containing the model:
```
docker build -t telestoai/model-api -f model/Dockerfile .
```

Running the container:
```
docker run -p 9876:9876 --name model-api --rm telestoai/model-api
```

## Calling model API using cURL

```
curl http://localhost:9876/

curl -X POST -H "Content-Type: application/json" --data-binary @example_data/example_input.json http://localhost:9876/
```

## Evaluating model running in a container

```
pip install -r requirements-dev.txt

python -m evaluate --api-url http://localhost:9876 --dataset-path example_data/test_dataset --metric accuracy
```

## An example model
A fully worked out example can be found in the `example_model` folder. You can try it out just like your model as well, using the commands above:
```
docker build -t telestoai/model-api-example -f example_model/Dockerfile .
docker run -p 9876:9876 --name model-api --rm telestoai/model-api-example
curl http://localhost:9876/
curl -X POST -H "Content-Type: application/json" --data-binary @example_data/example_input.json http://localhost:9876/
```
