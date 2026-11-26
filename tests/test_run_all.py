import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_all import assemble_result, run_all


def location_item(signer_id: str, video_id: str):
    return {
        "sample_id": f"{signer_id}_{video_id}_R_Event_1_RIGHT",
        "event_id": "R_Event_1",
        "signer_id": signer_id,
        "y_pred_top1": "FR",
        "top3": ["FR", "CR"],
        "prob_top3": [0.7, 0.3],
        "margin": 0.4,
    }


class RunAllContractTests(unittest.TestCase):
    def test_exact_context_prevents_cross_video_match(self):
        raw = {
            "location": {
                "source": "existing_prediction",
                "items": [
                    location_item("S10", "V1"),
                    location_item("S20", "V1"),
                ],
            }
        }
        result = assemble_result(
            raw,
            event_id="R_Event_1",
            signer_id="S20",
            video_id="V1",
            hand="right",
        )
        channel = result["channels"]["location"]
        self.assertEqual(channel["status"]["code"], "OK")
        self.assertEqual(channel["sample_id"], "S20_V1_R_Event_1_RIGHT")

    def test_short_event_id_is_rejected_when_ambiguous(self):
        raw = {
            "location": {
                "source": "existing_prediction",
                "items": [
                    location_item("S10", "V1"),
                    location_item("S20", "V1"),
                ],
            }
        }
        result = assemble_result(raw, event_id="R_Event_1")
        self.assertEqual(
            result["channels"]["location"]["status"]["code"],
            "AMBIGUOUS_SAMPLE",
        )

    def test_full_sample_id_supplies_cross_channel_context(self):
        raw = {
            "location": {
                "source": "existing_prediction",
                "items": [location_item("S20", "V1")],
            }
        }
        result = assemble_result(
            raw,
            sample_id="S20_V1_R_Event_1_RIGHT",
        )
        self.assertEqual(result["event_id"], "R_Event_1")
        self.assertEqual(result["signer_id"], "S20")
        self.assertEqual(result["video_id"], "V1")
        self.assertEqual(result["channels"]["location"]["status"]["code"], "OK")

    def test_missing_model_is_preserved_in_unified_json(self):
        raw = {
            "orientation": {
                "source": "unavailable",
                "status": "prediction_file_not_found",
                "error": "checkpoint missing",
            }
        }
        result = assemble_result(
            raw,
            event_id="R_Event_1",
            video_id="V1",
            hand="right",
        )
        channel = result["channels"]["orientation"]
        self.assertEqual(channel["status"]["code"], "MODEL_UNAVAILABLE")
        self.assertEqual(channel["status"]["message"], "checkpoint missing")

    def test_geometric_orientation_fallback_replaces_missing_prediction(self):
        fallback = {
            "source": "geometric_fallback",
            "items": [{
                "event_id": "R_Event_1",
                "video_id": "V1",
                "hand": "right",
                "orientation": "TU",
                "top3_labels": ["TU", "TLU", "TRU"],
                "top3_probs": [0.6, 0.2, 0.2],
            }],
        }
        result = run_all(
            event_id="R_Event_1",
            video_id="V1",
            hand="right",
            channel_results={
                "orientation": {
                    "source": "existing_prediction",
                    "status": "sample_not_found",
                    "items": [],
                }
            },
            channel_overrides={"orientation": fallback},
        )
        channel = result["channels"]["orientation"]
        self.assertEqual(channel["status"]["code"], "OK")
        self.assertEqual(channel["prediction"]["label"], "TU")
        self.assertEqual(channel["metadata"]["source"], "geometric_fallback")

    @patch("handshape_channel_data28.scripts.run_handshape._infer_unseen_handshape")
    def test_handshape_neural_inference_requires_the_unnumbered_flag(self, infer_mock):
        infer_mock.return_value = {
            "status": "ok",
            "source": "neural_inference",
            "items": [{"sample_id": "S000_V9_L_Event_1_LEFT", "topk": []}],
        }
        table = {
            "items": [{
                "sample_id": "S142_V6332_L_Event_1_LEFT",
                "event_id": "L_Event_1",
                "signer_id": "S142",
                "video_id": "V6332",
                "hand": "left",
                "topk": [{"label": "HSA", "prob": 0.9}],
            }]
        }
        from handshape_channel_data28.scripts import run_handshape as handshape

        with tempfile.TemporaryDirectory() as directory:
            table_path = Path(directory) / "test_pred.json"
            table_path.write_text(json.dumps(table), encoding="utf-8")
            with patch.object(handshape, "OUT_JSON", table_path):
                missed = handshape.run_handshape(
                    sample_id="S000_V9_L_Event_1_LEFT",
                    event_id="L_Event_1",
                    signer_id="S000",
                    video_id="V9",
                    hand="left",
                    video_path="new.mp4",
                    span={"start_frame": 0, "end_frame": 10},
                )
                self.assertEqual(missed["status"], "sample_not_found")
                infer_mock.assert_not_called()

                inferred = handshape.run_handshape(
                    sample_id="S000_V9_L_Event_1_LEFT",
                    event_id="L_Event_1",
                    signer_id="S000",
                    video_id="V9",
                    hand="left",
                    video_path="new.mp4",
                    span={"start_frame": 0, "end_frame": 10},
                    allow_unnumbered_inference=True,
                )
                self.assertEqual(inferred["source"], "neural_inference")

                kept = handshape.run_handshape(
                    sample_id="S142_V6332_L_Event_1_LEFT",
                    event_id="L_Event_1",
                    signer_id="S142",
                    video_id="V6332",
                    hand="left",
                    video_path="corpus.mp4",
                    span={"start_frame": 25, "end_frame": 44},
                    allow_unnumbered_inference=True,
                )
        self.assertEqual(kept["source"], "existing_prediction")
        self.assertEqual(kept["items"][0]["topk"][0]["label"], "HSA")
        infer_mock.assert_called_once()

    @patch("location_channel.scripts.run_location._infer_unseen_location")
    def test_location_neural_inference_requires_the_unnumbered_flag(self, infer_mock):
        infer_mock.return_value = {
            "source": "neural_inference",
            "status": "ok",
            "items": [],
        }
        from location_channel.scripts.run_location import run_location

        missed = run_location(
            sample_id="S000_V999999_L_Event_9_LEFT",
            event_id="L_Event_9",
            signer_id="S000",
            video_id="V999999",
            hand="left",
        )
        self.assertEqual(missed["status"], "sample_not_found")
        infer_mock.assert_not_called()

        inferred = run_location(
            sample_id="S000_V999999_L_Event_9_LEFT",
            event_id="L_Event_9",
            signer_id="S000",
            video_id="V999999",
            hand="left",
            allow_unnumbered_inference=True,
        )
        self.assertEqual(inferred["source"], "neural_inference")
        infer_mock.assert_called_once()

    def test_orientation_model_result_is_not_replaced_by_geometry(self):
        fallback = {
            "source": "geometric_fallback",
            "items": [{
                "event_id": "R_Event_1",
                "video_id": "V1",
                "hand": "right",
                "signer_id": "S000",
                "orientation": "TU",
                "top3_labels": ["TU", "TLU"],
                "top3_probs": [0.6, 0.4],
            }],
        }
        result = run_all(
            event_id="R_Event_1",
            video_id="V1",
            hand="right",
            signer_id="S000",
            channel_results={
                "orientation": {
                    "source": "neural_inference",
                    "status": "ok",
                    "items": [{
                        "event_id": "R_Event_1",
                        "video_id": "V1",
                        "hand": "right",
                        "signer_id": "S000",
                        "orientation": "TB",
                        "top3_labels": ["TB", "TD"],
                        "top3_probs": [0.8, 0.2],
                    }],
                }
            },
            channel_overrides={"orientation": fallback},
        )
        channel = result["channels"]["orientation"]
        self.assertEqual(channel["status"]["code"], "OK")
        self.assertEqual(channel["prediction"]["label"], "TB")
        self.assertEqual(channel["metadata"]["source"], "neural_inference")

    @patch("orientation_channel.scripts.run_orientation._infer_unseen_orientation")
    def test_orientation_neural_inference_requires_the_unnumbered_flag(self, infer_mock):
        infer_mock.return_value = {
            "source": "neural_inference",
            "status": "ok",
            "items": [{
                "event_id": "R_Event_99",
                "video_id": "V999999",
                "hand": "right",
                "signer_id": "S000",
                "orientation": "TL",
                "top3_labels": ["TL"],
                "top3_probs": [0.7],
            }],
        }
        from orientation_channel.scripts.run_orientation import run_orientation

        missed = run_orientation(
            event_id="R_Event_99",
            video_id="V999999",
            hand="right",
            signer_id="S000",
        )
        self.assertIn(missed["status"], ("sample_id_not_found", "prediction_file_not_found"))
        infer_mock.assert_not_called()

        inferred = run_orientation(
            event_id="R_Event_99",
            video_id="V999999",
            hand="right",
            signer_id="S000",
            allow_unnumbered_inference=True,
        )
        self.assertEqual(inferred["source"], "neural_inference")
        infer_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
