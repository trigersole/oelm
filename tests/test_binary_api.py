from __future__ import annotations

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
        self.assertEqual(health["model_version"], "xgb-binary-v1")
        self.assertEqual(health["levels"], ["Low", "High"])
        self.assertEqual(health["feature_count"], 364)
        self.assertEqual(set(health["models_loaded"]), set(binary_api.LABEL_COLS))
        for label, model in binary_api.models.items():
            with self.subTest(label=label):
                config = json.loads(model.save_config())
                self.assertEqual(config["learner"]["objective"]["name"], "binary:logistic")
                self.assertEqual(model.num_features(), 364)
                self.assertGreater(
                    model.num_boosted_rounds(), binary_api.BEST_ITERATIONS[label]
                )

    def test_model_info_exposes_binary_mapping(self) -> None:
        info = binary_api.model_info()
        self.assertEqual(info["levels"], {0: "Low", 1: "High"})
        self.assertEqual(info["source_level_mapping"]["Low"], ["Very Low", "Low"])
        self.assertEqual(info["source_level_mapping"]["High"], ["High", "Very High"])
        self.assertEqual(info["training_seed"], 1729)
        self.assertEqual(info["best_iterations"], binary_api.BEST_ITERATIONS)

    def test_prediction_contract_is_strictly_binary(self) -> None:
        response = binary_api.predict(
            binary_api.PredictRequest(agg_features=self.zero_features)
        )
        self.assertEqual(set(response), set(binary_api.LABEL_COLS))
        for label, prediction in response.items():
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

    def test_missing_features_are_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            binary_api.predict(binary_api.PredictRequest(agg_features={}))
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("Missing 364 features", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
