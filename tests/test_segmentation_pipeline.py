import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unnumbered_segment import (
    align_bimanual_events,
    merge_overlapping_spans,
    segment_kinematic_signing_space,
    spans_from_model_inside,
    trim_still_edges,
)
from segment_video import choose_signer_hand_landmarks, mediapipe_label_for_signer_hand
from video2json import _parse_video_uid, _is_corpus_filename, build_segment_events, use_unnumbered_path


class SegmentationPipelineTests(unittest.TestCase):
    def test_arbitrary_video_name_gets_stable_synthetic_identity(self):
        first = _parse_video_uid(Path("my new sign video.mp4"))
        second = _parse_video_uid(Path("my new sign video.mov"))

        self.assertEqual(first, second)
        self.assertEqual(first[1], "S000")
        self.assertRegex(first[2], r"^V\d+$")

    def test_front_camera_uses_the_signers_own_hand(self):
        self.assertEqual(mediapipe_label_for_signer_hand("right"), "Left")
        self.assertEqual(mediapipe_label_for_signer_hand("left"), "Right")

    def test_duplicate_mediapipe_label_keeps_the_higher_score(self):
        class _Label:
            def __init__(self, label, score):
                self.label = label
                self.score = score

        class _Handedness:
            def __init__(self, label, score):
                self.classification = [_Label(label, score)]

        raised = object()
        pocket = object()
        chosen = choose_signer_hand_landmarks(
            [pocket, raised],
            [_Handedness("Left", 0.4), _Handedness("Left", 0.9)],
            "right",
        )
        self.assertIs(chosen, raised)

    def test_plain_filename_is_not_treated_as_corpus(self):
        self.assertTrue(_is_corpus_filename(Path("slr-4-demo-142-6332.mp4")))
        self.assertFalse(_is_corpus_filename(Path("0158.mp4")))
        self.assertEqual(
            _parse_video_uid(Path("S43_slr-4-demo-43-6332.mp4")),
            ("S43_V6332", "S43", "V6332"),
        )

    @patch("segment_video.segment_from_video_file")
    def test_converts_left_and_right_spans_to_events(self, segment_mock):
        def fake_segment(*args, **kwargs):
            start = 5 if kwargs["hand"] == "left" else 20
            return {
                "n_frames": 80,
                "fps": 30.0,
                "detect_ratio": 0.9,
                "segments": [{
                    "start": start,
                    "end": start + 9,
                    "score": 0.8,
                    "inside_mean": 0.9,
                }],
            }

        segment_mock.side_effect = fake_segment
        events, video_uid = build_segment_events(
            Path("S43_slr-4-demo-43-6332.mp4"),
            checkpoint_path="best.pt",
        )

        self.assertEqual(video_uid, "S43_V6332")
        self.assertEqual([event["event_id"] for event in events], [
            "L_Event_1",
            "R_Event_1",
        ])
        self.assertEqual(events[0]["span"], {
            "start_frame": 5,
            "end_frame": 15,
        })
        self.assertEqual(events[0]["boundary"]["source"], "BoundaryTCN")

    def test_corpus_events_do_not_enter_the_unnumbered_path(self):
        self.assertFalse(use_unnumbered_path(True, "auto"))
        self.assertFalse(use_unnumbered_path(False, "model"))
        self.assertFalse(use_unnumbered_path(False, "annotation"))

    def test_unnumbered_path_runs_only_when_auto_finds_nothing(self):
        self.assertTrue(use_unnumbered_path(False, "auto"))

    def test_model_inside_probability_makes_a_span_below_the_strict_threshold(self):
        inside = np.zeros(40, dtype=np.float32)
        inside[8:24] = 0.3

        spans = spans_from_model_inside(inside)

        self.assertEqual([(item["start"], item["end"]) for item in spans], [(8, 23)])

    def test_still_hand_edges_are_removed_from_unnumbered_spans(self):
        pose = np.zeros((60, 21, 3), dtype=np.float32)
        pose[:, 0, 0] = 0.2
        pose[:25, 0, 1] = 0.80
        pose[25:41, 0, 1] = np.linspace(0.80, 0.40, 16)
        pose[41:, 0, 1] = 0.40

        spans = trim_still_edges(pose, [{
            "start": 0,
            "end": 59,
            "score": 0.4,
            "inside_mean": 0.4,
        }])

        self.assertEqual(len(spans), 1)
        self.assertGreater(spans[0]["start"], 10)
        self.assertLess(spans[0]["end"], 55)
        self.assertGreaterEqual(spans[0]["length"], 6)

    def test_completely_still_hand_span_is_dropped(self):
        pose = np.zeros((40, 21, 3), dtype=np.float32)
        pose[:, 0, 0] = 0.2
        pose[:, 0, 1] = 0.8

        spans = trim_still_edges(pose, [{"start": 0, "end": 39, "score": 0.2}])

        self.assertEqual(spans, [])

    def test_missed_frames_do_not_keep_the_still_hand(self):
        pose = np.zeros((40, 21, 3), dtype=np.float32)
        detected = np.zeros(40, dtype=bool)
        pose[:20, 0, 0] = 0.39
        pose[:20, 0, 1] = 0.85
        detected[:20] = True
        pose[23:, 0, 0] = 0.44
        pose[23:, 0, 1] = 0.50
        detected[23:] = True

        spans = trim_still_edges(
            pose,
            [{"start": 0, "end": 39, "score": 0.4}],
            detected=detected,
        )

        self.assertEqual(len(spans), 1)
        self.assertGreaterEqual(spans[0]["start"], 23)

    def test_overlapping_unnumbered_spans_become_one_span(self):
        spans = merge_overlapping_spans([
            {"start": 28, "end": 50, "score": 0.2},
            {"start": 39, "end": 82, "score": 0.4},
            {"start": 49, "end": 64, "score": 0.3},
            {"start": 72, "end": 102, "score": 0.5},
            {"start": 120, "end": 130, "score": 0.2},
        ])

        self.assertEqual(
            [(item["start"], item["end"]) for item in spans],
            [(28, 82), (72, 102), (120, 130)],
        )

    def test_later_sign_is_not_swallowed_by_the_previous_span(self):
        spans = merge_overlapping_spans([
            {"start": 25, "end": 121, "score": 0.4},
            {"start": 67, "end": 176, "score": 0.5},
            {"start": 140, "end": 241, "score": 0.4},
        ])

        self.assertEqual(
            [(item["start"], item["end"]) for item in spans],
            [(25, 121), (67, 176), (140, 241)],
        )

    def test_signs_that_only_touch_stay_separate(self):
        spans = merge_overlapping_spans([
            {"start": 4, "end": 29, "score": 0.4},
            {"start": 29, "end": 129, "score": 0.5},
        ])

        self.assertEqual(
            [(item["start"], item["end"]) for item in spans],
            [(4, 29), (29, 129)],
        )

    def test_kinematic_signing_space_preserves_chest_hold_and_splits_relocation(self):
        pose = np.zeros((100, 21, 3), dtype=np.float32)
        detected = np.ones(100, dtype=bool)
        pose[:, 0, 0] = 0.35
        pose[:, 0, 1] = 0.88
        # Preparation rise (frames 16..24)
        pose[16:25, 0, 1] = np.linspace(0.88, 0.56, 9)
        # Event 1 stroke at chest (frames 25..42)
        pose[25:43, 0, 1] = 0.56 + 0.02 * np.sin(np.linspace(0, np.pi, 18))
        pose[25:43, 0, 0] = np.linspace(0.35, 0.28, 18)
        # Transition dip & relocation (frames 43..60)
        pose[43:51, 0, 1] = np.linspace(0.57, 0.67, 8)
        pose[51:61, 0, 1] = np.linspace(0.67, 0.52, 10)
        pose[51:61, 0, 0] = np.linspace(0.28, 0.52, 10)
        # Event 2 hold at chest (frames 61..83)
        pose[61:84, 0, 1] = 0.52
        pose[61:84, 0, 0] = 0.52
        # Retraction drop to waist (frames 84..93)
        pose[84:94, 0, 1] = np.linspace(0.52, 0.88, 10)

        spans = segment_kinematic_signing_space(
            pose,
            detected=detected,
            shoulder_y=np.full(100, 0.50, dtype=np.float32),
            hip_y=np.full(100, 0.90, dtype=np.float32),
        )
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0]["start"], 23, delta=3)
        self.assertAlmostEqual(spans[0]["end"], 42, delta=3)
        self.assertAlmostEqual(spans[1]["start"], 61, delta=3)
        self.assertAlmostEqual(spans[1]["end"], 83, delta=3)

    def test_bimanual_events_are_co_aligned_when_overlapping(self):
        events = [
            {"event_id": "L_Event_1", "hand": "left", "span": {"start_frame": 25, "end_frame": 40}},
            {"event_id": "R_Event_1", "hand": "right", "span": {"start_frame": 23, "end_frame": 43}},
            {"event_id": "R_Event_2", "hand": "right", "span": {"start_frame": 62, "end_frame": 85}},
        ]
        aligned = align_bimanual_events(events)
        self.assertEqual(aligned[0]["span"], {"start_frame": 24, "end_frame": 42})
        self.assertEqual(aligned[1]["span"], {"start_frame": 24, "end_frame": 42})
        self.assertEqual(aligned[2]["span"], {"start_frame": 62, "end_frame": 85})


if __name__ == "__main__":
    unittest.main()
