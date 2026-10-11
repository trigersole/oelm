from __future__ import annotations

import hashlib
import json
import unittest

from fastapi import HTTPException

import app as binary_api


class BinaryApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        binary_api.load_models()
        cls.zero_features = {name: 0.0 for name in binary_api.FEATURE_ORDER}

    def test_model_resources_and_health(self) -> None:
        health = binary_api.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["model_version"], "xgb260-static-seed1729-v1")
        self.assertEqual(health["levels"], ["Low", "High"])
        self.assertEqual(health["feature_count"], 260)
        self.assertEqual(
            health["feature_schema"],
            "52-blendshapes-x-mean-std-min-max-median",
        )
        self.assertEqual(set(health["models_loaded"]), set(binary_api.LABEL_COLS))
        for label, model in binary_api.models.items():
            with self.subTest(label=label):
                config = json.loads(model.save_config())
                self.assertEqual(config["learner"]["objective"]["name"], "binary:logistic")
                self.assertEqual(model.num_features(), 260)
                self.assertEqual(
                    model.num_boosted_rounds(),
                    binary_api.BEST_ITERATIONS[label] + 1,
                )

    def test_model_info_exposes_binary_mapping(self) -> None:
        info = binary_api.model_info()
        self.assertEqual(info["levels"], {0: "Low", 1: "High"})
        self.assertEqual(info["source_level_mapping"]["Low"], ["Very Low", "Low"])
        self.assertEqual(info["source_level_mapping"]["High"], ["High", "Very High"])
        self.assertEqual(info["training_seed"], 1729)
        self.assertEqual(info["best_iterations"], binary_api.BEST_ITERATIONS)
        self.assertEqual(
            info["decision_thresholds"],
            {
                "Boredom": 0.5299999999999999,
                "Engagement": 0.73,
                "Confusion": 0.5399999999999999,
                "Frustration": 0.22999999999999998,
            },
        )

    def test_feature_order_is_exactly_xgb260_static(self) -> None:
        self.assertEqual(len(binary_api.FEATURE_ORDER), 260)
        self.assertEqual(len(set(binary_api.FEATURE_ORDER)), 260)
        allowed_suffixes = ("_mean", "_std", "_min", "_max", "_median")
        self.assertTrue(
            all(name.endswith(allowed_suffixes) for name in binary_api.FEATURE_ORDER)
        )
        self.assertFalse(
            any(name.endswith(("_skew", "_kurt")) for name in binary_api.FEATURE_ORDER)
        )

    def test_deployment_artifacts_match_the_selected_checkpoint(self) -> None:
        for filename, expected in binary_api.MODEL_CONFIG["artifact_sha256"].items():
            with self.subTest(filename=filename):
                actual = hashlib.sha256(
                    (binary_api.MODEL_DIR / filename).read_bytes()
                ).hexdigest()
                self.assertEqual(actual, expected)

    def test_prediction_contract_is_strictly_binary(self) -> None:
        response = binary_api.predict(
            binary_api.PredictRequest(agg_features=self.zero_features)
        )
        self.assertEqual(response["model_version"], binary_api.MODEL_VERSION)
        self.assertEqual(response["feature_schema"], binary_api.FEATURE_SCHEMA)
        for label in binary_api.LABEL_COLS:
            prediction = response[label]
            with self.subTest(label=label):
                self.assertIn(prediction["label"], (0, 1))
                self.assertEqual(
                    prediction["level"], binary_api.LEVEL_NAMES[prediction["label"]]
                )
                self.assertEqual(set(prediction["probabilities"]), {0, 1})
                self.assertAlmostEqual(
                    sum(prediction["probabilities"].values()), 1.0, places=4
                )
                expected = int(
                    prediction["probabilities"][1] >= prediction["threshold"]
                )
                self.assertEqual(prediction["label"], expected)

        expected_high_probabilities = {
            "Boredom": 0.46982,
            "Engagement": 0.76372,
            "Confusion": 0.3556,
            "Frustration": 0.18983,
        }
        for label, expected_probability in expected_high_probabilities.items():
            self.assertAlmostEqual(
                response[label]["probabilities"][1], expected_probability, places=5
            )

    def test_missing_features_are_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            binary_api.predict(binary_api.PredictRequest(agg_features={}))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("Missing 260 features", raised.exception.detail)

    def test_browser_and_compatibility_routes_are_registered(self) -> None:
        post_paths = {
            route.path
            for route in binary_api.app.routes
            if "POST" in getattr(route, "methods", set())
        }
        self.assertIn("/predict-binary", post_paths)
        self.assertIn("/predict", post_paths)


if __name__ == "__main__":
    unittest.main()
