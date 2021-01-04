import argparse
import base64
import io
from pathlib import Path
from typing import List, Iterator
import time

import numpy as np
import pandas as pd
from PIL import Image
import requests


def image_to_base64(image: Image) -> bytes:
    fp = io.BytesIO()
    image.save(fp, format="PNG")
    fp.seek(0)
    return base64.b64encode(fp.read())


def jaccard(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Jaccard (IoU) metric.

    Args:
        y_pred: 2d array of bool or int [0, 1] values
        y_true: 2d array of bool or int [0, 1] values
    """
    intersection = (y_pred * y_true).sum()
    union_plus_intersection = y_true.sum() + y_pred.sum()
    return intersection / (union_plus_intersection - intersection)


def estimate_metric(
    pred_list: Iterator[np.ndarray], label_list: Iterator[np.ndarray], class_n: int
) -> float:
    metrics_per_image = []
    for preds, labels in zip(pred_list, label_list):
        metrics_per_class = {cl: jaccard(preds == cl, labels == cl) for cl in range(class_n)}
        metric_avg = sum(metrics_per_class.values()) / len(metrics_per_class)
        metrics_per_image.append(metric_avg)
    return sum(metrics_per_image) / len(metrics_per_image)


def run_model_inference(api_url: str, image_paths: Iterator[str]) -> List[np.ndarray]:
    job_ids = []
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        input_doc = {"image": {"content": image_to_base64(image).decode()}}
        resp = requests.post(f"{api_url}/jobs", json=input_doc)
        assert resp.ok, resp.text
        job_doc = resp.json()
        job_ids.append((i, job_doc["job_id"]))
        print(f"Submitted {image_path}: {job_doc}")

    pred_list = [np.empty(0)] * len(job_ids)
    while True:
        active_job_ids = []
        for i, job_id in job_ids:
            resp = requests.get(f"{api_url}/jobs/{job_id}")
            if resp.status_code == 200:
                output_doc = resp.json()
                pred_bytes = base64.b64decode(output_doc["mask"]["content"])
                pred = np.array(Image.open(io.BytesIO(pred_bytes)))
                pred_list[i] = pred
                print(f"Fetched {job_id}")
            else:
                active_job_ids.append((i, job_id))

        if len(active_job_ids) > 0:
            print(f"Active jobs: {len(active_job_ids)}")
            job_ids = active_job_ids
            time.sleep(10)
        else:
            break

    return pred_list


def evaluate_model(api_url: str, dataset_path: str, class_n: int = 3):
    dataset_path = Path(dataset_path)
    labels_df = pd.read_csv(dataset_path / "labels.tsv", sep="\t")

    pred_list = run_model_inference(api_url,
                                    map(lambda p: dataset_path / p, labels_df.image_path.values))
    label_list = [np.asarray(Image.open(dataset_path / mask_path)) for mask_path in
                  labels_df.mask_path]

    score = estimate_metric(pred_list, label_list, class_n)
    print(f"Model performance: jaccard={score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--api-url")
    parser.add_argument("--dataset-path")
    parser.add_argument("--class-n")
    args = parser.parse_args()

    evaluate_model(args.api_url, args.dataset_path, int(args.class_n))
