# telesto.ai Model API Template Project
This serves as a brief guide to help you submit your solutions for telesto.ai competitions. For a complete guide, see the [technical documentation](https://docs.telesto.ai/)!

## Preparing model archive
These instructions are for a classification model but for other types of models the steps are the same.

* Use the `classification/model` directory for preparing a model archive.
* Add weights of a model you developed to the directory.
* Add inference code. The model inference class should inherit `telesto.models.ClassificationModelBase`. For more information, see [the telesto-base package](https://github.com/telesto-ai/telesto-base), where the base classes are defined.
* Pack an archive with all needed files using `cd classification; zip -r model.zip model`.

[A concrete and worked out example can be found here.](https://github.com/telesto-ai/telesto-models/tree/master/example_model)
For a detailed guide, see [https://docs.telesto.ai](https://docs.telesto.ai)!

## Building Docker image and starting container
Building the image containing the model:
```
docker build --pull -t telestoai/model-api classification/model
```

Running the container:
```
docker run -p 9876:9876 --name model-api --rm telestoai/model-api
```

## Testing classification model API
Send a sample input
```
curl -X POST -H "Content-Type:application/json" --data-binary @classification/example_data/example-input.json -i \
    http://localhost:9876/
...
{
    "predictions": [
        {"probs": {"cat": 0.32015, "dog": 0.67985}, "prediction": "dog"},
        {"probs": {"cat": 0.81545, "dog": 0.18455}, "prediction": "cat"}
    ]
}
```

## Testing segmentation model API
Post a sample input
```
curl -X POST -H "Content-Type:application/json" --data-binary @segmentation/example_data/example-input.json -i \
    http://localhost:9876/jobs
...
{
    "job_id": "b741bd19767441f6b7abd022744083c9"
}
```

Get the result
```
curl -H "Content-Type:application/json" -i http://localhost:9876/jobs/b741bd19767441f6b7abd022744083c9
...
{
    "mask": {
        "content": "<BASE_64_IMAGE>"
    }
}
```

## Evaluating classification model running in container
```
pip install -r requirements-dev.txt
python -m evaluate --api-url http://localhost:9876 --dataset-path classification/example_data/test_dataset --metric accuracy
```

## An example classification model
A fully worked out example of a classification model can be found in the `classification/example_model` directory. You can try it out just like your model as well, using the commands:
```
docker build --pull -t telestoai/model-api-example classification/model_example

docker run -p 9876:9876 --name model-api --rm telestoai/model-api-example

curl http://localhost:9876/

curl -X POST -H "Content-Type:application/json" --data-binary @classification/example_data/example-input.json -i \
    http://localhost:9876/
```

## An example segmentation model
A complete segmentation example model can be prepared, using the following commands:
```
curl -o segmentation/model_example/model.pt https://telesto-ai-content.fra1.digitaloceanspaces.com/example-models/segmentation/model.pt

docker build --pull -t telestoai/model-api-example segmentation/model_example

docker run -p 9876:9876 --name model-api --rm telestoai/model-api-example

curl http://localhost:9876/

curl -X POST -H "Content-Type:application/json" --data-binary @segmentation/example_data/example-input.json -i \
    http://localhost:9876/jobs

curl -H "Content-Type:application/json" -i http://localhost:9876/jobs/<job_id>
```