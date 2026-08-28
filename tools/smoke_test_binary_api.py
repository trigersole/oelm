"""Smoke-test a running local or Hugging Face OELM binary API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LABELS = {"Boredom", "Engagement", "Confusion", "Frustration"}
LEVELS = {0: "Low", 1: "High"}
ROOT_DIR = Path(__file__).resolve().parents[1]


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach {url}: {error.reason}") from error


def validate_prediction(payload: dict) -> None:
    if set(payload) != LABELS:
        raise AssertionError(f"Expected labels {sorted(LABELS)}, found {sorted(payload)}")

    for affective_label, prediction in payload.items():
        label = prediction.get("label")
        if label not in LEVELS:
            raise AssertionError(f"{affective_label}: expected binary label, found {label}")
        if prediction.get("level") != LEVELS[label]:
            raise AssertionError(f"{affective_label}: label/level mismatch")

        probabilities = prediction.get("probabilities", {})
        low = float(probabilities.get("0", probabilities.get(0, -1)))
        high = float(probabilities.get("1", probabilities.get(1, -1)))
        if not (0 <= low <= 1 and 0 <= high <= 1):
            raise AssertionError(f"{affective_label}: probabilities are outside [0, 1]")
        if abs((low + high) - 1.0) > 1e-4:
            raise AssertionError(f"{affective_label}: probabilities do not sum to one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    health = request_json(f"{base_url}/health")
    if health.get("status") != "ok":
        raise AssertionError(f"Unhealthy service: {health}")
    if health.get("levels") != ["Low", "High"]:
        raise AssertionError(f"Unexpected levels: {health.get('levels')}")
    if health.get("feature_count") != 364:
        raise AssertionError(f"Unexpected feature count: {health.get('feature_count')}")

    with (ROOT_DIR / "binary_models" / "feature_order.json").open(
        "r", encoding="utf-8"
    ) as handle:
        feature_order = json.load(handle)
    features = {name: 0.0 for name in feature_order}
    prediction = request_json(
        f"{base_url}/predict", {"agg_features": features}
    )
    validate_prediction(prediction)

    print(f"PASS {base_url}")
    print(json.dumps({"health": health, "prediction": prediction}, indent=2))


if __name__ == "__main__":
    main()
