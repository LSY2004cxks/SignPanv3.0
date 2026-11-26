import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movement_channel.src.geometry import movement_from_wrist
from movement_channel.src.gold import lookup_gold, span_iou
from movement_channel.scripts.run_movement import run_movement
from run_all import assemble_result, run_all


def _ann_event(event_id: str, start: int, end: int, global_type: str, local_type: str = "HOLD"):
    return {
        "event_id": event_id,
        "span": {"start_frame": start, "end_frame": end},
        "movement": {
            "global": {
                "track_point": "WRIST",
                "label": [{"type": global_type, "repeat": 1}],
            },
            "local": {
                "label": [{"type": local_type, "repeat": 1}],
            },
        },
    }


class MovementChannelTests(unittest.TestCase):
    def test_span_iou(self):
        self.assertGreater(
            span_iou({"start_frame": 10, "end_frame": 20}, {"start_frame": 15, "end_frame": 25}),
            0.2,
        )

    def test_straight_path_is_st(self):
        points = np.stack([np.linspace(0, 80, 24), np.zeros(24)], axis=1)
        scores = np.ones(24)
        result = movement_from_wrist(
            points,
            scores,
            hand="right",
            event_id="R_Event_1",
            signer_id="S1",
            video_id="V1",
        )
        self.assertEqual(result["source"], "geometric_fallback")
        self.assertEqual(result["items"][0]["y_pred_top1"], "ST")

    def test_still_path_is_hold(self):
        points = np.ones((24, 2), dtype=np.float32) * 40
        points += np.random.RandomState(0).normal(0, 0.2, size=points.shape)
        result = movement_from_wrist(
            points,
            np.ones(24),
            hand="right",
            event_id="R_Event_1",
            signer_id="S1",
            video_id="V1",
        )
        self.assertEqual(result["items"][0]["y_pred_top1"], "HOLD")

    def test_gold_lookup_and_run_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann_dir = root / "S1-label"
            ann_dir.mkdir()
            payload = {
                "hands": {
                    "right": {
                        "events": [_ann_event("R_Event_1", 10, 20, "ST", "HOLD")]
                    }
                }
            }
            (ann_dir / "S1_V1.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            item = lookup_gold(
                ann_root=root,
                signer_id="S1",
                video_id="V1",
                event_id="R_Event_1",
                hand="right",
            )
            self.assertIsNotNone(item)
            self.assertEqual(item["y_pred_top1"], "ST")
            self.assertEqual(item["local_label"], "HOLD")

            raw = run_movement(
                signer_id="S1",
                video_id="V1",
                event_id="R_Event_1",
                hand="right",
                ann_root=str(root),
            )
            self.assertEqual(raw["source"], "gold_lookup")
            result = assemble_result(
                {"movement": raw},
                event_id="R_Event_1",
                signer_id="S1",
                video_id="V1",
                hand="right",
            )
            channel = result["channels"]["movement"]
            self.assertEqual(channel["status"]["code"], "OK")
            self.assertEqual(channel["prediction"]["label"], "ST")
            self.assertEqual(channel["metadata"]["source"], "gold_lookup")
            self.assertEqual(channel["metadata"]["local_label"], "HOLD")

    def test_geometric_override_when_unlabelled(self):
        fallback = movement_from_wrist(
            np.stack([np.linspace(0, 50, 16), np.zeros(16)], axis=1),
            np.ones(16),
            hand="right",
            event_id="R_Event_1",
            signer_id="S9",
            video_id="V9",
        )
        result = run_all(
            event_id="R_Event_1",
            signer_id="S9",
            video_id="V9",
            hand="right",
            channel_results={
                "movement": {
                    "source": "unavailable",
                    "status": "missing_features_and_predictions",
                    "items": [],
                }
            },
            channel_overrides={"movement": fallback},
        )
        channel = result["channels"]["movement"]
        self.assertEqual(channel["status"]["code"], "OK")
        self.assertEqual(channel["metadata"]["source"], "geometric_fallback")
        self.assertIn(channel["prediction"]["label"], {"ST", "CU", "CI", "HOLD"})

    def test_video_geometry_replaces_filename_matched_gold(self):
        fallback = movement_from_wrist(
            np.stack([np.linspace(0, 80, 16), np.zeros(16)], axis=1),
            np.ones(16),
            hand="right",
            event_id="R_Event_1",
            signer_id="S1",
            video_id="V1",
        )
        result = run_all(
            event_id="R_Event_1",
            signer_id="S1",
            video_id="V1",
            hand="right",
            channel_results={
                "movement": {
                    "source": "gold_lookup",
                    "status": "ok",
                    "items": [{
                        "event_id": "R_Event_1",
                        "signer_id": "S1",
                        "video_id": "V1",
                        "hand": "right",
                        "y_pred_top1": "NOM",
                        "gt": "NOM",
                        "top3": ["NOM"],
                        "prob_top3": [1.0],
                    }],
                }
            },
            channel_overrides={"movement": fallback},
        )
        channel = result["channels"]["movement"]
        self.assertEqual(channel["metadata"]["source"], "geometric_fallback")
        self.assertNotEqual(channel["prediction"]["label"], "NOM")
        self.assertEqual(channel["metadata"]["corpus_label"], "NOM")


if __name__ == "__main__":
    unittest.main()
