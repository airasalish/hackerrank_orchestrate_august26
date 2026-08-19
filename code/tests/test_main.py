"""
Lightweight tests for the deterministic (non-API) logic in code/main.py — sanitize(),
is_domain_mismatch(), _keywords(), and find_evidence_candidates(). No network calls, no
API keys needed; runs against the real dataset/ CSVs already provided in this repo.

Usage:
    python -m unittest code/tests/test_main.py -v
    (or: python code/tests/test_main.py)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import main as m


class TestSanitize(unittest.TestCase):
    def test_invalid_action_falls_back_to_digest(self):
        result = m.sanitize({"action": "delete_everything", "message_type": "personal",
                              "confidence": 0.5, "evidence_message_ids": "none", "reason": "x"})
        self.assertEqual(result["action"], "digest")

    def test_invalid_message_type_falls_back_to_unknown(self):
        result = m.sanitize({"action": "mute", "message_type": "not_a_real_type",
                              "confidence": 0.5, "evidence_message_ids": "none", "reason": "x"})
        self.assertEqual(result["message_type"], "unknown")

    def test_confidence_clamped_above_one(self):
        result = m.sanitize({"action": "notify", "message_type": "urgent",
                              "confidence": 5, "evidence_message_ids": "none", "reason": "x"})
        self.assertEqual(result["confidence"], 1.0)

    def test_confidence_clamped_below_zero(self):
        result = m.sanitize({"action": "notify", "message_type": "urgent",
                              "confidence": -3, "evidence_message_ids": "none", "reason": "x"})
        self.assertEqual(result["confidence"], 0.0)

    def test_confidence_non_numeric_defaults_to_half(self):
        result = m.sanitize({"action": "notify", "message_type": "urgent",
                              "confidence": "very sure", "evidence_message_ids": "none", "reason": "x"})
        self.assertEqual(result["confidence"], 0.5)

    def test_evidence_list_converted_to_semicolon_string(self):
        result = m.sanitize({"action": "notify", "message_type": "urgent", "confidence": 0.8,
                              "evidence_message_ids": ["message_0001", "message_0002"], "reason": "x"})
        self.assertEqual(result["evidence_message_ids"], "message_0001;message_0002")

    def test_empty_evidence_list_becomes_none_string(self):
        result = m.sanitize({"action": "notify", "message_type": "urgent", "confidence": 0.8,
                              "evidence_message_ids": [], "reason": "x"})
        self.assertEqual(result["evidence_message_ids"], "none")

    def test_missing_reason_gets_placeholder_not_empty(self):
        result = m.sanitize({"action": "mute", "message_type": "spam", "confidence": 0.9,
                              "evidence_message_ids": "none", "reason": ""})
        self.assertTrue(result["reason"])


class TestDomainMismatch(unittest.TestCase):
    def test_true_when_domains_differ(self):
        self.assertTrue(m.is_domain_mismatch(
            {"official_domain": "realbank.com", "domain_used_by_sender": "realbank-support.net"}))

    def test_false_when_domains_match_case_insensitive(self):
        self.assertFalse(m.is_domain_mismatch(
            {"official_domain": "RealBank.com", "domain_used_by_sender": "realbank.com"}))

    def test_false_when_data_missing(self):
        self.assertFalse(m.is_domain_mismatch({"official_domain": "", "domain_used_by_sender": ""}))
        self.assertFalse(m.is_domain_mismatch({}))


class TestKeywords(unittest.TestCase):
    def test_filters_stopwords_and_short_words(self):
        kw = m._keywords("Hi, is this the office? Please call at 5 PM.")
        self.assertNotIn("is", kw)
        self.assertNotIn("the", kw)
        self.assertNotIn("please", kw)  # filler word, in _STOPWORDS by design
        self.assertIn("office", kw)
        self.assertIn("call", kw)

    def test_empty_text_gives_empty_set(self):
        self.assertEqual(m._keywords(""), set())
        self.assertEqual(m._keywords(None), set())


class TestFindEvidenceCandidates(unittest.TestCase):
    """Uses the real dataset/ directory already in this repo — no API calls."""

    @classmethod
    def setUpClass(cls):
        dataset_dir = Path(__file__).parent.parent.parent / "dataset"
        cls.ds = m.Dataset(dataset_dir)
        import csv
        with open(dataset_dir / "sample_messages.csv", newline="", encoding="utf-8") as f:
            cls.sample_rows = {r["message_id"]: r for r in csv.DictReader(f)}

    def test_only_returns_related_messages(self):
        row = self.sample_rows["sample_msg_001"]
        candidates = m.find_evidence_candidates(self.ds, row, current_text=row["message_text"])
        for c in candidates:
            hist_row = next(h for h in self.ds.history_by_user.get(row["user_id"], [])
                             if h["message_id"] == c["message_id"])
            related = (
                hist_row.get("sender_user_id") == row.get("sender_user_id")
                or hist_row.get("group_id") == row.get("group_id")
                or hist_row.get("business_id") == row.get("business_id")
            )
            self.assertTrue(related, f"{c['message_id']} isn't actually related to sample_msg_001")

    def test_respects_limit(self):
        row = self.sample_rows["sample_msg_001"]
        candidates = m.find_evidence_candidates(self.ds, row, current_text=row["message_text"], limit=3)
        self.assertLessEqual(len(candidates), 3)

    def test_regression_topical_similarity_beats_pure_recency(self):
        """Locks in the fix for the evidence-overlap gap found via the 30-row eval:
        message_0001 shares user/sender/group with sample_msg_001 and is topically
        relevant (same tank/valve/water topic) but is from March, while more recent
        same-group messages exist. Before the keyword-overlap re-ranking, message_0001
        was ranked out of the top-5 entirely by recency alone."""
        row = self.sample_rows["sample_msg_001"]
        candidates = m.find_evidence_candidates(self.ds, row, current_text=row["message_text"])
        ids = [c["message_id"] for c in candidates]
        self.assertIn("message_0001", ids, "topically-relevant older evidence got ranked out again")


if __name__ == "__main__":
    unittest.main(verbosity=2)
