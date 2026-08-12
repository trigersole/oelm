"""Load-test the OELM Hugging Face API and optional Supabase data path.

The workload mirrors the browser application:

* every virtual student makes one startup prediction;
* every virtual student reads the cohort list when Supabase is enabled;
* write mode creates a session and login log for each student;
* while learning, every student predicts and inserts one result per interval.

Only Python's standard library is required. Supabase write mode is deliberately
opt-in because it creates identifiable load-test rows in the configured project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ORDER_FILE = ROOT / "src" / "data" / "featureOrder.js"
LABELS = ("Boredom", "Engagement", "Confusion", "Frustration")


@dataclass(frozen=True)
class Sample:
    operation: str
    ok: bool
    latency_ms: float
    status: int | None
    error: str | None = None


class Results:
    def __init__(self) -> None:
        self._samples: list[Sample] = []
        self._lock = threading.Lock()

    def add(self, sample: Sample) -> None:
        with self._lock:
            self._samples.append(sample)

    def snapshot(self) -> list[Sample]:
        with self._lock:
            return list(self._samples)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def load_feature_order() -> list[str]:
    source = FEATURE_ORDER_FILE.read_text(encoding="utf-8")
    match = re.search(r"export\s+const\s+FEATURE_ORDER\s*=\s*(\[.*\])\s*;", source, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not parse feature order from {FEATURE_ORDER_FILE}")
    features = json.loads(match.group(1))
    if not isinstance(features, list) or not features:
        raise RuntimeError("FEATURE_ORDER is empty or invalid")
    return [str(feature) for feature in features]


def make_aggregate(features: list[str], seed: int = 1729) -> dict[str, float]:
    """Create valid, stable feature data without using student recordings."""
    rng = random.Random(seed)
    base_by_blendshape: dict[str, float] = {}
    aggregate: dict[str, float] = {}
    for feature in features:
        blendshape, statistic_name = feature.rsplit("_", 1)
        base = base_by_blendshape.setdefault(blendshape, rng.uniform(0.03, 0.35))
        if statistic_name == "std":
            value = rng.uniform(0.005, 0.06)
        elif statistic_name == "min":
            value = max(0.0, base - rng.uniform(0.01, 0.05))
        elif statistic_name == "max":
            value = min(1.0, base + rng.uniform(0.01, 0.05))
        elif statistic_name == "skew":
            value = rng.uniform(-0.25, 0.25)
        elif statistic_name == "kurt":
            value = rng.uniform(-0.5, 0.5)
        else:
            value = base
        aggregate[feature] = round(value, 5)
    return aggregate


def request_json(
    operation: str,
    url: str,
    results: Results,
    *,
    method: str = "GET",
    payload: Any = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20,
) -> tuple[bool, Any, int | None]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(url=url, data=body, headers=request_headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
            parsed = json.loads(raw) if raw else None
        latency_ms = (time.perf_counter() - started) * 1_000
        results.add(Sample(operation, 200 <= status < 300, latency_ms, status))
        return 200 <= status < 300, parsed, status
    except HTTPError as error:
        latency_ms = (time.perf_counter() - started) * 1_000
        detail = error.read(400).decode("utf-8", errors="replace")
        message = f"HTTP {error.code}: {detail}"
        results.add(Sample(operation, False, latency_ms, error.code, message))
        return False, None, error.code
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        latency_ms = (time.perf_counter() - started) * 1_000
        message = f"{type(error).__name__}: {error}"
        results.add(Sample(operation, False, latency_ms, None, message))
        return False, None, None


def hf_predict(
    hf_url: str,
    aggregate: dict[str, float],
    results: Results,
    operation: str,
    timeout: float,
) -> dict[str, Any] | None:
    ok, response, _ = request_json(
        operation,
        f"{hf_url.rstrip('/')}/predict",
        results,
        method="POST",
        payload={"agg_features": aggregate},
        timeout=timeout,
    )
    if not ok or not isinstance(response, dict):
        return None
    if not all(label in response for label in LABELS):
        results.add(Sample(f"{operation}_invalid_response", False, 0.0, None, "Missing emotion labels"))
        return None
    return response


class SupabaseClient:
    def __init__(self, url: str, key: str, results: Results, timeout: float) -> None:
        self.url = url.rstrip("/")
        self.results = results
        self.timeout = timeout
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "return=representation",
        }

    def call(
        self,
        operation: str,
        table: str,
        *,
        method: str = "GET",
        payload: Any = None,
        params: dict[str, str] | None = None,
    ) -> tuple[bool, Any]:
        url = f"{self.url}/rest/v1/{table}"
        if params:
            url = f"{url}?{urlencode(params)}"
        ok, response, _ = request_json(
            operation,
            url,
            self.results,
            method=method,
            payload=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        return ok, response


def fallback_prediction() -> dict[str, Any]:
    return {
        label: {"label": 1, "probabilities": {"0": 0.1, "1": 0.7, "2": 0.15, "3": 0.05}}
        for label in LABELS
    }


def prediction_payload(
    session_id: str,
    participant_id: str,
    cohort_id: str,
    aggregate: dict[str, float],
    prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    prediction = prediction or fallback_prediction()
    labels_flat: dict[str, int] = {}
    probabilities: dict[str, Any] = {}
    max_confidence = 0.0
    for label in LABELS:
        item = prediction.get(label) or fallback_prediction()[label]
        predicted_label = int(item.get("label", 1))
        label_probabilities = item.get("probabilities") or {}
        confidence = label_probabilities.get(str(predicted_label), label_probabilities.get(predicted_label, 0.0))
        labels_flat[label] = predicted_label
        probabilities[label] = label_probabilities
        max_confidence = max(max_confidence, float(confidence or 0.0))
    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "cohort_id": cohort_id,
        "activity_type": "without_edit",
        "pause_and_reflect_number": 1,
        "label": json.dumps(labels_flat, separators=(",", ":")),
        "confidence": round(max_confidence, 5),
        "probabilities": probabilities,
        "model_version": "xgb-v1-load-test",
        "raw_data": aggregate,
    }


@dataclass(frozen=True)
class WorkloadConfig:
    users: int
    duration: float
    interval: float
    ramp_up: float
    hf_url: str
    timeout: float
    supabase_mode: str
    cohort_id: str
    run_id: str
    steady_only: bool


def run_student(
    user_number: int,
    config: WorkloadConfig,
    aggregate: dict[str, float],
    results: Results,
    barrier: threading.Barrier,
    supabase: SupabaseClient | None,
    existing_session: dict[str, Any] | None = None,
) -> None:
    barrier.wait()
    if config.ramp_up > 0:
        time.sleep(config.ramp_up * user_number / max(config.users - 1, 1))

    participant_id = f"loadtest-{config.run_id}-{user_number + 1:03d}"
    session_id = str(uuid.uuid4())

    if existing_session:
        session_id = str(existing_session["session_id"])
        participant_id = str(existing_session["participant_id"])

    if supabase and not config.steady_only:
        supabase.call(
            "supabase_cohort_read",
            "login_credentials",
            params={
                "select": "cohort_id,access_code,created_at,activity_type,task_description,active",
                "order": "created_at.desc",
                "limit": "500",
            },
        )

    if supabase and config.supabase_mode == "write" and not config.steady_only:
        supabase.call(
            "supabase_session_insert",
            "sessions",
            method="POST",
            payload={
                "session_id": session_id,
                "label": f"loadtest:{config.run_id}",
                "participant_id": participant_id,
                "cohort_id": config.cohort_id,
                "activity_type": "without_edit",
            },
        )
        supabase.call(
            "supabase_log_insert",
            "logs",
            method="POST",
            payload={
                "session_id": session_id,
                "cohort_id": config.cohort_id,
                "participant_id": participant_id,
                "event_name": "load_test_session_login",
                "event_data": {},
                "vlearn_url": None,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    startup_prediction = None
    if not config.steady_only:
        # The real app makes this request immediately on page load.
        startup_prediction = hf_predict(
            config.hf_url, aggregate, results, "hf_startup_predict", config.timeout
        )

    deadline = time.monotonic() + config.duration
    next_request = time.monotonic() + config.interval
    last_prediction = startup_prediction
    while next_request <= deadline:
        time.sleep(max(0.0, next_request - time.monotonic()))
        last_prediction = hf_predict(
            config.hf_url, aggregate, results, "hf_steady_predict", config.timeout
        )
        if supabase and config.supabase_mode == "write":
            supabase.call(
                "supabase_prediction_insert",
                "emotion_predictions",
                method="POST",
                payload=prediction_payload(
                    session_id,
                    participant_id,
                    config.cohort_id,
                    aggregate,
                    last_prediction,
                ),
            )
        next_request += config.interval


def summarize(samples: list[Sample], wall_seconds: float) -> dict[str, Any]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.operation].append(sample)

    summary: dict[str, Any] = {}
    for operation, operation_samples in sorted(grouped.items()):
        latencies = [sample.latency_ms for sample in operation_samples]
        successes = sum(sample.ok for sample in operation_samples)
        status_counts = Counter(str(sample.status or "network") for sample in operation_samples)
        summary[operation] = {
            "requests": len(operation_samples),
            "successes": successes,
            "errors": len(operation_samples) - successes,
            "error_rate_percent": round(100 * (len(operation_samples) - successes) / len(operation_samples), 3),
            "requests_per_second": round(len(operation_samples) / max(wall_seconds, 0.001), 3),
            "latency_ms": {
                "median": round(statistics.median(latencies), 2),
                "p95": round(percentile(latencies, 95), 2),
                "p99": round(percentile(latencies, 99), 2),
                "max": round(max(latencies), 2),
            },
            "statuses": dict(status_counts),
            "example_errors": list(dict.fromkeys(sample.error for sample in operation_samples if sample.error))[:3],
        }
    return summary


def combined_metrics(samples: list[Sample], prefix: str) -> dict[str, float]:
    selected = [sample for sample in samples if sample.operation.startswith(prefix)]
    if not selected:
        return {"requests": 0, "error_rate_percent": 0.0, "p95_ms": 0.0}
    return {
        "requests": len(selected),
        "error_rate_percent": 100 * sum(not sample.ok for sample in selected) / len(selected),
        "p95_ms": percentile([sample.latency_ms for sample in selected], 95),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=50, help="Concurrent virtual students (default: 50)")
    parser.add_argument("--duration", type=float, default=120, help="Steady phase seconds per student (default: 120)")
    parser.add_argument("--interval", type=float, default=10, help="Seconds between predictions per student (default: 10)")
    parser.add_argument("--ramp-up", type=float, default=0, help="Seconds over which students start; 0 is a worst-case burst")
    parser.add_argument("--hf-url", default=os.getenv("HF_SPACE_URL", ""))
    parser.add_argument("--timeout", type=float, default=20, help="Per-request timeout seconds")
    parser.add_argument("--supabase-mode", choices=("off", "read", "write"), default="off")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL", ""))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_ANON_KEY", ""))
    parser.add_argument("--cohort-id", default="", help="Existing test cohort ID; required for Supabase writes")
    parser.add_argument("--confirm-production-writes", action="store_true", help="Required acknowledgment for Supabase write mode")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
    parser.add_argument(
        "--steady-only",
        action="store_true",
        help="Skip startup predictions, cohort reads, session inserts, and login logs",
    )
    parser.add_argument("--max-error-rate", type=float, default=1.0, help="Allowed error percentage")
    parser.add_argument("--hf-p95-ms", type=float, default=2_000)
    parser.add_argument("--supabase-p95-ms", type=float, default=1_000)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.users < 1 or args.duration < 0 or args.interval <= 0 or args.ramp_up < 0:
        raise ValueError("users must be >= 1, duration/ramp-up >= 0, and interval > 0")
    if args.supabase_mode != "off" and (not args.supabase_url or not args.supabase_key):
        raise ValueError("Supabase mode requires SUPABASE_URL and SUPABASE_ANON_KEY")
    if args.supabase_mode == "write" and not args.confirm_production_writes:
        raise ValueError("Supabase write mode requires --confirm-production-writes")
    if args.supabase_mode == "write" and not args.cohort_id:
        raise ValueError("Supabase write mode requires --cohort-id for an existing test cohort")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        features = load_feature_order()
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    aggregate = make_aggregate(features)
    results = Results()
    supabase = None
    if args.supabase_mode != "off":
        supabase = SupabaseClient(args.supabase_url, args.supabase_key, results, args.timeout)

    config = WorkloadConfig(
        users=args.users,
        duration=args.duration,
        interval=args.interval,
        ramp_up=args.ramp_up,
        hf_url=args.hf_url,
        timeout=args.timeout,
        supabase_mode=args.supabase_mode,
        cohort_id=args.cohort_id,
        run_id=args.run_id,
        steady_only=args.steady_only,
    )

    existing_sessions: list[dict[str, Any]] | None = None
    if args.steady_only and args.supabase_mode == "write":
        assert supabase is not None
        ok, rows = supabase.call(
            "setup_existing_session_read",
            "sessions",
            params={
                "select": "session_id,participant_id",
                "cohort_id": f"eq.{args.cohort_id}",
                "order": "participant_id.asc",
                "limit": str(args.users),
            },
        )
        if not ok or not isinstance(rows, list) or len(rows) < args.users:
            print(
                f"Configuration error: steady-only write mode needs {args.users} existing "
                f"sessions in cohort {args.cohort_id!r}; found {len(rows) if isinstance(rows, list) else 0}",
                file=sys.stderr,
            )
            return 2
        existing_sessions = rows[: args.users]

    print(
        f"OELM load test run={args.run_id} users={args.users} duration={args.duration}s "
        f"interval={args.interval}s ramp_up={args.ramp_up}s supabase={args.supabase_mode}"
    )
    if args.supabase_mode == "write":
        print(f"Supabase test rows will be labeled loadtest:{args.run_id}; they are not deleted automatically.")

    barrier = threading.Barrier(args.users + 1)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.users, thread_name_prefix="oelm-student") as executor:
        futures = [
            executor.submit(
                run_student,
                user,
                config,
                aggregate,
                results,
                barrier,
                supabase,
                existing_sessions[user] if existing_sessions else None,
            )
            for user in range(args.users)
        ]
        barrier.wait()
        for future in futures:
            future.result()
    wall_seconds = time.monotonic() - started

    samples = results.snapshot()
    operation_summary = summarize(samples, wall_seconds)
    report = {
        "run_id": args.run_id,
        "configuration": {**asdict(config), "feature_count": len(features)},
        "wall_seconds": round(wall_seconds, 3),
        "operations": operation_summary,
    }

    print(f"\nCompleted in {wall_seconds:.1f}s")
    print(f"{'operation':32} {'count':>7} {'err%':>8} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'max ms':>10}")
    for operation, metrics in operation_summary.items():
        latency = metrics["latency_ms"]
        print(
            f"{operation:32} {metrics['requests']:7d} {metrics['error_rate_percent']:8.2f} "
            f"{latency['median']:10.1f} {latency['p95']:10.1f} {latency['p99']:10.1f} {latency['max']:10.1f}"
        )
        for error in metrics["example_errors"]:
            print(f"  error: {error}")

    hf = combined_metrics(samples, "hf_")
    sb = combined_metrics(samples, "supabase_")
    failures: list[str] = []
    if hf["error_rate_percent"] > args.max_error_rate:
        failures.append(f"Hugging Face error rate {hf['error_rate_percent']:.2f}% > {args.max_error_rate:.2f}%")
    if hf["p95_ms"] > args.hf_p95_ms:
        failures.append(f"Hugging Face p95 {hf['p95_ms']:.0f}ms > {args.hf_p95_ms:.0f}ms")
    if sb["requests"] and sb["error_rate_percent"] > args.max_error_rate:
        failures.append(f"Supabase error rate {sb['error_rate_percent']:.2f}% > {args.max_error_rate:.2f}%")
    if sb["requests"] and sb["p95_ms"] > args.supabase_p95_ms:
        failures.append(f"Supabase p95 {sb['p95_ms']:.0f}ms > {args.supabase_p95_ms:.0f}ms")

    report["assessment"] = {"passed": not failures, "failures": failures}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report: {args.output.resolve()}")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nPASS: configured latency and error thresholds were met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
