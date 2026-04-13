from __future__ import annotations

import queue
import unittest
from unittest.mock import patch

from person_detection import pipeline as person_pipeline


class PersonDetectionLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_drop_count = person_pipeline.log_queue_drop_count
        person_pipeline.log_queue_drop_count = 0

    def tearDown(self) -> None:
        person_pipeline.log_queue_drop_count = self.previous_drop_count

    def test_enqueue_log_item_warns_when_queue_is_full(self) -> None:
        with patch.object(
            person_pipeline.log_queue,
            "put_nowait",
            side_effect=queue.Full,
        ), patch("person_detection.pipeline.print") as print_mock:
            ok = person_pipeline.enqueue_log_item("crop", "/tmp/full.jpg")

        self.assertFalse(ok)
        print_mock.assert_called_once()
        self.assertEqual(person_pipeline.log_queue_drop_count, 1)
