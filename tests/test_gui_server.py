import io
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gui_server


SAMPLE_RESULT = [{
    "schema": "sign_phonology_v1",
    "sample_id": "S1_V1_R_Event_1_RIGHT",
    "event_id": "R_Event_1",
    "signer_id": "S1",
    "video_id": "V1",
        "hand": "right",
        "segmentation": {
            "span": {"start_frame": 10, "end_frame": 20},
            "fps": 30,
        },
        "channels": {
        "location": {
            "sample_id": "S1_V1_R_Event_1_RIGHT",
            "event_id": "R_Event_1",
            "prediction": {"label": "FR", "conf": 0.8},
            "metadata": {
                "model_label": "CR",
                "human_modified": True,
                "source": "human_annotation",
            },
            "status": {"code": "OK", "message": ""},
        }
    },
}]


class SegmentAlignmentTests(unittest.TestCase):
    def test_same_hand_overlap_is_split_and_both_hands_share_a_segment(self):
        events = [
            {
                "event_id": "L_Event_1",
                "hand": "left",
                "standard": {"start_ms": 0, "end_ms": 1720},
                "channels": {"handshape": {"prediction": {"label": "HSC"}}},
            },
            {
                "event_id": "R_Event_1",
                "hand": "right",
                "standard": {"start_ms": 0, "end_ms": 1600},
                "channels": {"handshape": {"prediction": {"label": "HSU"}}},
            },
            {
                "event_id": "L_Event_3",
                "hand": "left",
                "standard": {"start_ms": 980, "end_ms": 1420},
                "channels": {"handshape": {"prediction": {"label": "HSV"}}},
            },
        ]
        payload = gui_server.result_to_standard_json(
            events,
            video_path="0158.mp4",
            annotation_meta={"dominant_hand": "right", "record_id": "0158"},
        )
        segments = payload["segments"]
        self.assertEqual(
            [(item["segment_id"], item["start_ms"], item["end_ms"]) for item in segments],
            [
                ("S1", 0, 980),
                ("S2", 980, 1420),
                ("S3", 1420, 1600),
                ("S4", 1600, 1720),
            ],
        )
        self.assertEqual(segments[0]["hands"]["H1"]["handshape"]["value"], "HSU")
        self.assertEqual(segments[0]["hands"]["H2"]["handshape"]["value"], "HSC")
        self.assertEqual(segments[1]["hands"]["H2"]["handshape"]["value"], "HSV")
        self.assertEqual(segments[1]["hands"]["H1"]["participation"], "active")
        self.assertTrue(segments[0]["annotated"])
        self.assertTrue(segments[1]["annotated"])
        self.assertTrue(segments[2]["annotated"])
        self.assertTrue(segments[3]["annotated"])
        self.assertEqual(segments[3]["hands"]["H1"]["participation"], "inactive")
        self.assertEqual(segments[3]["hands"]["H2"]["participation"], "active")
        self.assertEqual(segments[3]["source_event_ids"], ["L_Event_1"])

    def test_segment_shorter_than_one_tenth_is_not_annotated(self):
        events = [
            {
                "event_id": "R_Event_5",
                "hand": "right",
                "standard": {"start_ms": 4833, "end_ms": 4899},
                "channels": {
                    "handshape": {"prediction": {"label": "HNU"}},
                    "location": {"prediction": {"label": "SR"}},
                    "orientation": {"prediction": {"label": "TLU"}},
                    "movement": {"prediction": {"label": "CI"}},
                },
            }
        ]
        payload = gui_server.result_to_standard_json(
            events,
            annotation_meta={"dominant_hand": "right"},
        )
        segment = payload["segments"][0]
        self.assertEqual(segment["start_ms"], 4833)
        self.assertEqual(segment["end_ms"], 4899)
        self.assertFalse(segment["annotated"])
        self.assertEqual(segment["hands"]["H1"]["handshape"]["value"], "")
        self.assertEqual(segment["hands"]["H1"]["location"]["value"], "")
        self.assertEqual(segment["hands"]["H1"]["participation"], "inactive")
        self.assertEqual(segment["source_event_ids"], ["R_Event_5"])

        xml_text = gui_server.result_to_eaf(
            events,
            annotation_meta={"dominant_hand": "right"},
        ).decode("utf-8")
        self.assertNotIn("不标注", xml_text)
        self.assertNotIn(">HNU<", xml_text)
        self.assertNotIn(">SR<", xml_text)

    def test_corpus_hands_with_the_same_span_stay_in_one_segment(self):
        events = [
            {
                "event_id": "L_Event_1",
                "hand": "left",
                "standard": {"start_ms": 833, "end_ms": 1433},
                "channels": {"handshape": {"prediction": {"label": "HSA"}}},
            },
            {
                "event_id": "R_Event_1",
                "hand": "right",
                "standard": {"start_ms": 833, "end_ms": 1433},
                "channels": {"handshape": {"prediction": {"label": "HSU"}}},
            },
            {
                "event_id": "R_Event_2",
                "hand": "right",
                "standard": {"start_ms": 2067, "end_ms": 2733},
                "channels": {"handshape": {"prediction": {"label": "HSV"}}},
            },
        ]
        payload = gui_server.result_to_standard_json(
            events,
            annotation_meta={"dominant_hand": "right"},
        )
        segments = payload["segments"]
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["hands"]["H1"]["handshape"]["value"], "HSU")
        self.assertEqual(segments[0]["hands"]["H2"]["handshape"]["value"], "HSA")
        self.assertEqual(segments[1]["hands"]["H1"]["handshape"]["value"], "HSV")
        self.assertEqual(segments[1]["hands"]["H2"]["participation"], "inactive")


class GuiAnnotationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_dir = gui_server.ANNOTATION_DIR
        self.original_corrections_file = gui_server.CORRECTIONS_FILE
        gui_server.ANNOTATION_DIR = Path(self.temp_dir.name)
        gui_server.CORRECTIONS_FILE = (
            gui_server.ANNOTATION_DIR / "_corrections.json"
        )
        gui_server.tasks.clear()
        gui_server.tasks["task-1"] = {
            "status": "done",
            "progress": 100,
            "video_path": "demo.mp4",
            "result": SAMPLE_RESULT,
            "error": None,
        }
        self.client = gui_server.app.test_client()

    def tearDown(self):
        gui_server.ANNOTATION_DIR = self.original_dir
        gui_server.CORRECTIONS_FILE = self.original_corrections_file
        gui_server.tasks.clear()
        self.temp_dir.cleanup()

    def test_save_list_load_and_export(self):
        response = self.client.post(
            "/save/task-1",
            json={
                "result": SAMPLE_RESULT,
                "annotation_meta": {
                    "record_id": "demo",
                    "dominant_hand": "right",
                    "revision_version": "2",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((gui_server.ANNOTATION_DIR / "task-1.json").exists())

        listing = self.client.get("/api/annotations").get_json()
        self.assertEqual(listing["items"][0]["event_count"], 1)

        loaded = self.client.get("/api/annotations/task-1.json")
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.get_json()["result"][0]["event_id"], "R_Event_1")
        self.assertEqual(
            loaded.get_json()["annotation_meta"]["revision_version"],
            "2",
        )

        standard_export = self.client.get(
            "/export/task-1?format=standard_json"
        )
        self.assertEqual(standard_export.status_code, 200)
        standard = json.loads(standard_export.data)
        self.assertEqual(standard["standard_version"], "3.0")
        self.assertEqual(standard["record_id"], "demo")
        self.assertEqual(
            standard["segments"][0]["hands"]["H1"]["location"]["value"],
            "FR",
        )
        self.assertEqual(standard["nmf"]["status"], "observed_absent")
        self.assertEqual(standard["nmf"]["facial_events"], [])
        self.assertEqual(standard["nmf"]["torso_events"], [])
        self.assertIn("gloss", standard["annotation_scope"]["not_implemented"])
        self.assertNotIn(
            "non_manual_features",
            standard["annotation_scope"]["not_implemented"],
        )

        csv_export = self.client.get("/export/task-1?format=csv")
        self.assertEqual(csv_export.status_code, 200)
        self.assertIn(b"human_modified", csv_export.data)

        xml_export = self.client.get("/export/task-1?format=xml")
        self.assertEqual(xml_export.status_code, 200)
        xml_text = xml_export.data.decode("utf-8")
        self.assertIn("<ANNOTATION_DOCUMENT", xml_text)
        self.assertIn('TIER_ID="Location@Right"', xml_text)
        self.assertIn("<ANNOTATION_VALUE>FR</ANNOTATION_VALUE>", xml_text)
        self.assertIn('TIME_VALUE="333"', xml_text)
        self.assertIn(".eaf", xml_export.headers.get("Content-Disposition", ""))

        eaf_export = self.client.get("/export/task-1?format=eaf")
        self.assertEqual(eaf_export.status_code, 200)
        self.assertIn(b"ANNOTATION_DOCUMENT", eaf_export.data)

        legacy_json = self.client.get("/export/task-1?format=json")
        self.assertEqual(legacy_json.status_code, 400)
        self.assertIn("standard_json", legacy_json.get_json()["error"])

        unsupported = self.client.get("/export/task-1?format=yaml")
        self.assertEqual(unsupported.status_code, 400)

        fresh_result = json.loads(json.dumps(SAMPLE_RESULT))
        fresh_channel = fresh_result[0]["channels"]["location"]
        fresh_channel["prediction"]["label"] = "CR"
        fresh_channel["metadata"] = {"source": "model_inference"}
        applied = gui_server._apply_corrections(fresh_result)
        self.assertEqual(applied, 1)
        self.assertEqual(fresh_channel["prediction"]["label"], "FR")
        self.assertEqual(fresh_channel["metadata"]["model_label"], "CR")
        self.assertTrue(fresh_channel["metadata"]["correction_applied"])

    def test_standard_export_records_spec_symbols(self):
        event = json.loads(json.dumps(SAMPLE_RESULT[0]))
        event["standard"] = {
            "participation": "active",
            "annotation_status": "observed",
            "hand_role": "H1",
        }
        event["channels"]["handshape"] = {"prediction": {"label": "HSU"}}
        event["channels"]["orientation"] = {"prediction": {"label": "TB"}}
        event["channels"]["movement"] = {"prediction": {"label": "CU"}}
        meta = {
            "record_id": "demo",
            "dominant_hand": "right",
            "nmf": {
                "status": "observed",
                "facial_events": [{
                    "event_id": "FE1",
                    "label": "ebr",
                    "start_ms": 0,
                    "end_ms": 400,
                }],
                "torso_events": [{
                    "event_id": "TM1",
                    "label": "tf",
                    "start_ms": 100,
                    "end_ms": 500,
                }],
            },
        }
        payload = gui_server.result_to_standard_json(
            [event],
            video_path="demo.mp4",
            annotation_meta=meta,
        )
        hand = payload["segments"][0]["hands"]["H1"]
        inactive = payload["segments"][0]["hands"]["H2"]
        self.assertEqual(hand["location"]["value"], "FR")
        self.assertEqual(hand["movement"]["value"], "CU")
        self.assertEqual(hand["handshape"]["value"], "HSU")
        self.assertEqual(inactive["location"]["value"], "NOL")
        self.assertEqual(inactive["movement"]["value"], "NOM")
        self.assertEqual(hand["movement"]["path"], "CU")
        self.assertEqual(hand["movement"]["local"], "HOLD")
        self.assertEqual(hand["location"]["start"], "FR")
        self.assertEqual(hand["location"]["end"], "FR")
        self.assertEqual(payload["nmf"]["facial_events"][0]["label"], "ebr")
        self.assertEqual(payload["nmf"]["torso_events"][0]["label"], "tf")

        xml_text = gui_server.result_to_eaf(
            [event],
            video_path="demo.mp4",
            annotation_meta=meta,
        ).decode("utf-8")
        self.assertIn(">FR<", xml_text)
        self.assertIn(">CU<", xml_text)
        self.assertNotIn("WFL→TMC", xml_text)
        self.assertIn('TIER_ID="Facial"', xml_text)
        self.assertIn('TIER_ID="Torso"', xml_text)
        self.assertIn(">ebr<", xml_text)
        self.assertIn(">tf<", xml_text)

    def test_annotation_pages_and_delete(self):
        for index in range(3):
            gui_server.tasks[f"page-{index}"] = {
                "status": "done",
                "progress": 100,
                "video_path": f"clip-{index}.mp4",
                "result": SAMPLE_RESULT,
                "error": None,
            }
            saved = self.client.post(
                f"/save/page-{index}",
                json={"result": SAMPLE_RESULT},
            )
            self.assertEqual(saved.status_code, 200)

        first_page = self.client.get(
            "/api/annotations?page=1&page_size=2"
        ).get_json()
        self.assertEqual(first_page["total"], 3)
        self.assertEqual(first_page["pages"], 2)
        self.assertEqual(len(first_page["items"]), 2)
        self.assertEqual(first_page["page"], 1)

        second_page = self.client.get(
            "/api/annotations?page=2&page_size=2"
        ).get_json()
        self.assertEqual(len(second_page["items"]), 1)

        name = first_page["items"][0]["name"]
        deleted = self.client.delete(f"/api/annotations/{name}")
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse((gui_server.ANNOTATION_DIR / name).exists())

        blocked = self.client.delete("/api/annotations/_corrections.json")
        self.assertEqual(blocked.status_code, 400)
        invalid = self.client.get("/api/annotations?page=abc")
        self.assertEqual(invalid.status_code, 400)

        credits = gui_server.load_credits(
            "导师：张三，作者：赵四，王五，刘六\n"
        )
        self.assertEqual(credits["advisor"], "张三")
        self.assertEqual(credits["authors"], "赵四，王五，刘六")
        self.assertEqual(
            credits["display"],
            "导师：张三，作者：赵四，王五，刘六",
        )
        english_credits = gui_server.load_credits(
            "Supervisor: Dengfeng Yao\n"
            "Authors: Yuan Zhao, Bowen Ye\n"
        )
        self.assertEqual(english_credits["advisor"], "Dengfeng Yao")
        self.assertEqual(english_credits["authors"], "Yuan Zhao, Bowen Ye")
        about = self.client.get("/about")
        self.assertEqual(about.status_code, 200)
        self.assertIn("Dengfeng Yao".encode(), about.data)
        self.assertIn("Yuan Zhao".encode(), about.data)
        self.assertNotIn("未填写".encode(), about.data)
        english_about = self.client.get("/about/en")
        self.assertEqual(english_about.status_code, 200)
        self.assertIn(b"Supervisor: Dengfeng Yao", english_about.data)

    def test_eaf_keeps_unicode_filename_unencoded(self):
        video = Path(self.temp_dir.name) / "slr-4-忘恩负义-142-6332.mp4"
        video.write_bytes(b"fake")
        xml_text = gui_server.result_to_eaf(
            SAMPLE_RESULT,
            video_path=str(video),
        ).decode("utf-8")
        self.assertIn("忘恩负义", xml_text)
        self.assertNotIn("%E5%BF%98", xml_text)
        self.assertIn('RELATIVE_MEDIA_URL="./slr-4-忘恩负义-142-6332.mp4"', xml_text)
        self.assertIn("file:///", xml_text)

    def test_eaf_builds_simple_elan_compatible_time_aligned_tiers(self):
        result = []
        for hand, prefix, start in (
            ("left", "L", 10),
            ("right", "R", 30),
        ):
            channels = {}
            for key, label in (
                ("handshape", "HNU"),
                ("movement", "CI"),
                ("location", "TTL"),
                ("orientation", "TU"),
            ):
                channels[key] = {
                    "prediction": {"label": label, "conf": 0.8},
                    "metadata": {"source": "model_inference"},
                    "status": {"code": "OK", "message": ""},
                }
            result.append({
                "schema": "sign_phonology_v1",
                "event_id": f"{prefix}_Event_1",
                "signer_id": "S1",
                "video_id": "V1",
                "hand": hand,
                "segmentation": {
                    "span": {
                        "start_frame": start,
                        "end_frame": start + 10,
                    },
                    "fps": 30,
                },
                "channels": channels,
            })

        root = ET.fromstring(gui_server.result_to_eaf(result, "demo.mp4"))
        tiers = {
            tier.attrib["TIER_ID"]: tier
            for tier in root.findall("TIER")
        }
        expected_tiers = {
            "Utterance",
            "Event@Left",
            "Event@Right",
            *{
                f"{tier_name}@{side}"
                for side in ("Left", "Right")
                for _, tier_name in gui_server.CHANNEL_TIER_NAMES
            },
        }
        self.assertEqual(set(tiers), expected_tiers)
        self.assertIn("Utterance", tiers)
        self.assertIsNotNone(
            tiers["Utterance"].find("./ANNOTATION/ALIGNABLE_ANNOTATION")
        )
        self.assertIsNone(root.find(".//REF_ANNOTATION"))

        time_slot_ids = {
            slot.attrib["TIME_SLOT_ID"]
            for slot in root.findall("./TIME_ORDER/TIME_SLOT")
        }

        for side in ("Left", "Right"):
            event_tier = tiers[f"Event@{side}"]
            self.assertNotIn("PARENT_REF", event_tier.attrib)
            self.assertEqual(event_tier.attrib["LINGUISTIC_TYPE_REF"], "event")
            event_annotation = event_tier.find(
                "./ANNOTATION/ALIGNABLE_ANNOTATION"
            )
            self.assertIsNotNone(event_annotation)
            self.assertIn(
                event_annotation.attrib["TIME_SLOT_REF1"],
                time_slot_ids,
            )
            self.assertIn(
                event_annotation.attrib["TIME_SLOT_REF2"],
                time_slot_ids,
            )

            for _, tier_name in gui_server.CHANNEL_TIER_NAMES:
                feature_tier = tiers[f"{tier_name}@{side}"]
                self.assertNotIn("PARENT_REF", feature_tier.attrib)
                self.assertEqual(
                    feature_tier.attrib["LINGUISTIC_TYPE_REF"],
                    "phonological_feature",
                )
                feature_annotation = feature_tier.find(
                    "./ANNOTATION/ALIGNABLE_ANNOTATION"
                )
                self.assertIsNotNone(feature_annotation)
                self.assertIn(
                    feature_annotation.attrib["TIME_SLOT_REF1"],
                    time_slot_ids,
                )
                self.assertIn(
                    feature_annotation.attrib["TIME_SLOT_REF2"],
                    time_slot_ids,
                )
                self.assertIsNone(
                    feature_tier.find("./ANNOTATION/REF_ANNOTATION")
                )

        linguistic_types = {
            item.attrib["LINGUISTIC_TYPE_ID"]: item
            for item in root.findall("LINGUISTIC_TYPE")
        }
        self.assertEqual(
            linguistic_types["event"].attrib["TIME_ALIGNABLE"],
            "true",
        )
        self.assertEqual(
            linguistic_types["phonological_feature"].attrib["TIME_ALIGNABLE"],
            "true",
        )
        self.assertNotIn("CONSTRAINTS", linguistic_types["event"].attrib)
        self.assertNotIn(
            "CONSTRAINTS",
            linguistic_types["phonological_feature"].attrib,
        )
        properties = {
            item.attrib["NAME"]: item.text
            for item in root.findall("./HEADER/PROPERTY")
        }
        self.assertEqual(
            json.loads(properties["SignPan.sourceJSON"]),
            result,
        )
        self.assertIn(
            "Handshape|Movement|Location|Orientation",
            properties["SignPan.tierHierarchy"],
        )
        self.assertIsNotNone(root.find("./LANGUAGE[@LANG_ID='sgn']"))

    def test_eaf_is_ordered_and_non_overlapping_for_elan(self):
        result = []
        for index, (hand, start, end) in enumerate((
            ("left", 100, 400),
            ("left", 300, 500),
            ("right", 100, 100),
        )):
            result.append({
                "event_id": f"E{index + 1}",
                "hand": hand,
                "standard": {"start_ms": start, "end_ms": end},
                "channels": {
                    "handshape": {
                        "prediction": {"label": "HSU", "conf": 0.9},
                        "status": {"code": "OK", "message": ""},
                    }
                },
            })
        xml_text = gui_server.result_to_eaf(result, "demo.mp4").decode("utf-8")
        self.assertRegex(
            xml_text,
            r'DATE="\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00"',
        )
        root = ET.fromstring(xml_text)
        slots = [
            (slot.attrib["TIME_SLOT_ID"], int(slot.attrib["TIME_VALUE"]))
            for slot in root.findall("./TIME_ORDER/TIME_SLOT")
        ]
        self.assertEqual(
            [slot_id for slot_id, _ in slots],
            [f"ts{index}" for index in range(1, len(slots) + 1)],
        )
        self.assertEqual(
            [value for _, value in slots],
            sorted(value for _, value in slots),
        )
        slot_time = dict(slots)
        for tier in root.findall("TIER"):
            spans = []
            for annotation in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION"):
                start = slot_time[annotation.attrib["TIME_SLOT_REF1"]]
                end = slot_time[annotation.attrib["TIME_SLOT_REF2"]]
                self.assertLess(start, end)
                if spans:
                    self.assertGreaterEqual(start, spans[-1][1])
                spans.append((start, end))

    def test_english_interface_route(self):
        response = self.client.get("/en")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Sign Language Phonology", response.data)
        self.assertIn(b'href="/static/gui.css"', response.data)
        self.assertNotIn(b"<style>", response.data)

        chinese = self.client.get("/")
        self.assertEqual(chinese.status_code, 200)
        self.assertIn(b'href="/static/gui.css"', chinese.data)
        self.assertNotIn(b"<style>", chinese.data)

        stylesheet = self.client.get("/static/gui.css")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn(b".timeline-progress", stylesheet.data)
        self.assertIn(b'html[lang="en"]', stylesheet.data)

        alias = self.client.get("/gui.css")
        self.assertEqual(alias.status_code, 200)
        self.assertIn(b".timeline-progress", alias.data)

    @patch("gui_server.threading.Thread")
    def test_upload_accepts_arbitrary_video_filename(self, thread_class):
        temporary_root = Path(self.temp_dir.name)
        with patch.object(gui_server, "ROOT", temporary_root):
            response = self.client.post(
                "/upload",
                data={
                    "video": (
                        io.BytesIO(b"fake-video"),
                        "my new sign video.MP4",
                    ),
                    "segment_mode": "auto",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        task = gui_server.tasks[payload["task_id"]]
        self.assertEqual(Path(task["video_path"]).name, "my new sign video.mp4")
        thread_class.return_value.start.assert_called_once()

    def test_rejects_invalid_result(self):
        response = self.client.post("/save/task-1", json={"result": {}})
        self.assertEqual(response.status_code, 400)

    def test_save_rejects_manual_label_outside_spec(self):
        result = json.loads(json.dumps(SAMPLE_RESULT))
        result[0]["channels"]["location"]["prediction"]["label"] = "NOT_A_PLACE"
        response = self.client.post(
            "/save/task-1",
            json={"result": result, "annotation_meta": {}},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "输入错误")
        self.assertFalse((gui_server.ANNOTATION_DIR / "task-1.json").exists())

    def test_save_rejects_nmf_label_outside_spec(self):
        response = self.client.post(
            "/save/task-1",
            json={
                "result": SAMPLE_RESULT,
                "annotation_meta": {
                    "nmf": {
                        "status": "observed",
                        "facial_events": [{
                            "event_id": "FE1",
                            "label": "smile",
                            "start_ms": 0,
                            "end_ms": 100,
                        }],
                        "torso_events": [],
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "输入错误")

    def test_save_keeps_unedited_model_label_outside_spec(self):
        result = json.loads(json.dumps(SAMPLE_RESULT))
        result[0]["channels"]["location"]["prediction"]["label"] = "FAL"
        result[0]["channels"]["location"]["metadata"] = {
            "source": "model_inference",
            "human_modified": False,
        }
        response = self.client.post(
            "/save/task-1",
            json={"result": result, "annotation_meta": {}},
        )
        self.assertEqual(response.status_code, 200)

    def test_custom_segmentation_add_split_delete_and_repredict(self):
        manual_events = [
            {
                "schema": "sign_phonology_v1",
                "sample_id": "S000_V1_R_Event_1_RIGHT",
                "event_id": "R_Event_1",
                "signer_id": "S000",
                "video_id": "V1",
                "hand": "right",
                "segmentation": {
                    "source": "manual_add",
                    "fps": 30,
                    "span": {"start_frame": 0, "end_frame": 15},
                },
                "standard": {
                    "segment_id": "S1",
                    "start_ms": 0,
                    "end_ms": 500,
                    "hand_role": "H1",
                    "participation": "active",
                    "annotation_status": "observed",
                    "source": "human_annotation",
                },
                "channels": {
                    "handshape": {
                        "prediction": {"label": "NOH", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": False},
                        "status": {"code": "OK", "message": ""},
                    },
                    "location": {
                        "prediction": {"label": "NOL", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": True},
                        "status": {"code": "OK", "message": ""},
                    },
                    "orientation": {
                        "prediction": {"label": "NOO", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": False},
                        "status": {"code": "OK", "message": ""},
                    },
                    "movement": {
                        "prediction": {"label": "NOM", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": False},
                        "status": {"code": "OK", "message": ""},
                    },
                },
            },
            {
                "schema": "sign_phonology_v1",
                "sample_id": "S000_V1_R_Event_2_RIGHT",
                "event_id": "R_Event_2",
                "signer_id": "S000",
                "video_id": "V1",
                "hand": "right",
                "segmentation": {
                    "source": "manual_split",
                    "fps": 30,
                    "span": {"start_frame": 15, "end_frame": 30},
                },
                "standard": {
                    "segment_id": "S2",
                    "start_ms": 500,
                    "end_ms": 1000,
                    "hand_role": "H1",
                    "participation": "active",
                    "annotation_status": "observed",
                    "source": "human_annotation",
                },
                "channels": {
                    "handshape": {
                        "prediction": {"label": "HSU", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": True},
                        "status": {"code": "OK", "message": ""},
                    },
                    "location": {
                        "prediction": {"label": "FR", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": True},
                        "status": {"code": "OK", "message": ""},
                    },
                    "orientation": {
                        "prediction": {"label": "TU", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": True},
                        "status": {"code": "OK", "message": ""},
                    },
                    "movement": {
                        "prediction": {"label": "CI", "conf": 1.0, "topk": []},
                        "metadata": {"source": "human_annotation", "human_modified": True},
                        "status": {"code": "OK", "message": ""},
                    },
                },
            },
        ]
        saved = self.client.post(
            "/save/task-manual",
            json={"result": manual_events, "annotation_meta": {"dominant_hand": "right"}},
        )
        self.assertEqual(saved.status_code, 200)

        exported = self.client.get("/export/task-manual?format=standard_json")
        self.assertEqual(exported.status_code, 200)
        std = json.loads(exported.data)
        self.assertEqual(len(std["segments"]), 2)
        self.assertEqual(std["segments"][0]["start_ms"], 0)
        self.assertEqual(std["segments"][0]["end_ms"], 500)
        self.assertEqual(std["segments"][1]["start_ms"], 500)
        self.assertEqual(std["segments"][1]["end_ms"], 1000)

        video_file = Path(self.temp_dir.name) / "demo.mp4"
        video_file.write_bytes(b"fake")
        gui_server.tasks["task-manual"]["video_path"] = str(video_file)

        fake_fresh = {
            "channels": {
                "handshape": {
                    "prediction": {"label": "HSA", "conf": 0.9, "topk": [{"label": "HSA", "prob": 0.9}]},
                    "metadata": {"source": "neural_inference", "human_modified": False},
                    "status": {"code": "OK", "message": ""},
                },
                "location": {
                    "prediction": {"label": "CL", "conf": 0.85, "topk": [{"label": "CL", "prob": 0.85}]},
                    "metadata": {"source": "neural_inference", "human_modified": False},
                    "status": {"code": "OK", "message": ""},
                },
                "orientation": {
                    "prediction": {"label": "TB", "conf": 0.8, "topk": [{"label": "TB", "prob": 0.8}]},
                    "metadata": {"source": "geometric_fallback", "human_modified": False},
                    "status": {"code": "OK", "message": ""},
                },
                "movement": {
                    "prediction": {"label": "ST", "conf": 0.75, "topk": [{"label": "ST", "prob": 0.75}]},
                    "metadata": {"source": "geometric_fallback", "human_modified": False},
                    "status": {"code": "OK", "message": ""},
                },
            }
        }
        with patch("video2json.process_event") as proc_mock, \
             patch("video2json.deploy_to_channels"), \
             patch("run_all.run_all", return_value=fake_fresh):
            proc_mock.return_value = {
                "handshape": Path(self.temp_dir.name) / "hs.json",
                "location": Path(self.temp_dir.name) / "loc.npz",
                "orientation": Path(self.temp_dir.name) / "ori.npz",
                "movement": Path(self.temp_dir.name) / "mov.json",
                "orientation_prediction": {},
                "movement_prediction": {},
            }
            repred = self.client.post(
                "/api/repredict/task-manual",
                json={
                    "result": manual_events,
                    "annotation_meta": {"dominant_hand": "right"},
                    "segment_id": "S1",
                },
            )
        self.assertEqual(repred.status_code, 200)
        repred_data = repred.get_json()
        self.assertEqual(repred_data["repredicted"], 1)
        updated_s1 = repred_data["result"][0]
        self.assertEqual(updated_s1["channels"]["handshape"]["prediction"]["label"], "HSA")
        # Channel with human_modified=True is preserved by default unless overwrite_human=True
        self.assertEqual(updated_s1["channels"]["location"]["prediction"]["label"], "NOL")


if __name__ == "__main__":
    unittest.main()
