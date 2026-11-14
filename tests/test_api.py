import os
import tempfile
import unittest

from src.config import MODEL_PATH, VECTORIZER_PATH

MODEL_READY = MODEL_PATH.exists() and VECTORIZER_PATH.exists()
ARTICLE = ("The finance committee said on Tuesday that the new budget bill would be reviewed by lawmakers next month, "
           "according to several officials who spoke to reporters about the plan and its expected cost.")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["FND_DB_PATH"] = os.path.join(cls.tmp.name, "test_history.db")   # never touch the real history
        from app.main import create_app
        cls.client = create_app().test_client()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("FND_DB_PATH", None)
        cls.tmp.cleanup()

    def setUp(self):
        self.client.delete("/api/history")

    # ---- input validation (works with or without a trained model) ----
    def test_empty_input(self):
        r = self.client.post("/api/predict", json={"news": "   "})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"]["code"], "empty_input")

    def test_missing_field_and_wrong_type(self):
        self.assertEqual(self.client.post("/api/predict", json={}).get_json()["error"]["code"], "missing_field")
        self.assertEqual(self.client.post("/api/predict", json={"news": 123}).get_json()["error"]["code"], "invalid_type")

    def test_invalid_json_body(self):
        r = self.client.post("/api/predict", data="not json", content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"]["code"], "invalid_json")

    def test_input_too_large(self):
        r = self.client.post("/api/predict", json={"news": "word " * 20000})
        self.assertEqual(r.status_code, 413)

    def test_too_short(self):
        r = self.client.post("/api/predict", json={"news": "just a few words here"})
        self.assertEqual(r.get_json()["error"]["code"], "too_short")

    def test_unknown_route_and_wrong_method_return_json(self):
        self.assertEqual(self.client.get("/api/nope").get_json()["error"]["code"], "not_found")
        self.assertEqual(self.client.get("/api/predict").status_code, 405)

    def test_health(self):
        body = self.client.get("/api/health").get_json()
        self.assertIn(body["status"], {"ok", "model_missing"})

    def test_cors_header_for_frontend_origin(self):
        r = self.client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")
        r = self.client.get("/api/health", headers={"Origin": "http://evil.example"})
        self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

    # ---- history ----
    def test_history_starts_empty_and_database_is_created_automatically(self):
        body = self.client.get("/api/history").get_json()
        self.assertEqual((body["items"], body["total"]), ([], 0))

    def test_history_rejects_bad_query(self):
        self.assertEqual(self.client.get("/api/history?limit=abc").status_code, 400)

    # ---- analysis ----
    def test_performance_and_dataset_endpoints(self):
        self.assertEqual(self.client.get("/api/model/performance").status_code, 200)
        self.assertEqual(self.client.get("/api/dataset/summary").status_code, 200)

    def test_figure_endpoint_blocks_path_traversal(self):
        self.assertEqual(self.client.get("/api/analysis/figures/..%2f..%2fREADME.md").status_code, 404)
        self.assertEqual(self.client.get("/api/analysis/figures/missing.png").status_code, 404)

    # ---- full flow with the real saved model ----
    @unittest.skipUnless(MODEL_READY, "trained model not found (run scripts/train_models.py)")
    def test_predict_saves_history_and_clear_works(self):
        r = self.client.post("/api/predict", json={"news": ARTICLE})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn(body["prediction"], {"REAL NEWS", "FAKE NEWS"})
        self.assertTrue(50 <= body["confidence"] <= 100)
        self.assertIn("reliability", body)
        self.client.post("/api/predict", json={"news": ARTICLE + " Additional detail follows here."})
        history = self.client.get("/api/history").get_json()
        self.assertEqual(history["total"], 2)
        self.assertEqual(history["items"][0]["id"] > history["items"][1]["id"], True)   # newest first
        self.assertEqual(self.client.delete("/api/history").get_json()["deleted"], 2)
        self.assertEqual(self.client.get("/api/history").get_json()["total"], 0)

    @unittest.skipUnless(MODEL_READY, "trained model not found (run scripts/train_models.py)")
    def test_figures_are_served_as_png(self):
        r = self.client.get("/api/analysis/figures/confusion_matrix.png")
        self.assertEqual((r.status_code, r.mimetype), (200, "image/png"))
        r.close()


if __name__ == "__main__":
    unittest.main()
