import argparse
import base64
from pathlib import Path

import requests
import pandas as pd
from sklearn.metrics import accuracy_score


def convert_images_to_json(image_paths):
    base64_images = []
    for fp in image_paths:
        with open(fp, "rb") as f:
            content = base64.b64encode(f.read()).decode()
            base64_images.append(content)
    doc = {"images": [{"content": content} for content in base64_images]}
    return doc


def run_model_inference(api_url, image_paths):
    i, n = 0, 32
    predictions = []
    while True:
        image_paths_slice = image_paths[i : i + n]
        if len(image_paths_slice) == 0:
            break

        print(f"Predicting {len(image_paths_slice)} images")

        in_doc = convert_images_to_json(image_paths_slice)
        resp = requests.post(api_url, json=in_doc)
        if resp.status_code != 200:
            raise Exception(resp.text)

        out_doc = resp.json()
        predictions.extend(out_doc["predictions"])
        i += n
    return predictions


def estimate_accuracy(predictions, labels):
    labels_pred = [pred["prediction"] for pred in predictions]
    return accuracy_score(labels, labels_pred)


METRIC_FUNCTIONS = {"accuracy": estimate_accuracy}


def evaluate_model(api_url, dataset_path, metric_name):
    labels_path = Path(dataset_path) / "labels.tsv"
    labels_df = pd.read_csv(labels_path, sep="\t")
    image_paths = [Path(dataset_path) / p for p in labels_df.path.values]

    predictions = run_model_inference(api_url, image_paths)

    labels = labels_df["class"].values
    score = METRIC_FUNCTIONS[metric_name](predictions, labels)
    print(f"Model performance: {metric_name}={score:.4f}")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--api-url")
    parser.add_argument("--dataset-path")
    parser.add_argument("--metric")
    args = parser.parse_args()

    evaluate_model(args.api_url, args.dataset_path, args.metric)
