"""Regression checks for packet evidence, input turns, and interrupted playback."""
from __future__ import annotations

import struct
import unittest

import torch

from audio_stream import AudioWindowBuffer, PacketReceipt
from realtime_voice import RealtimeVoiceSession
from voice_endpoint import TranscriptEndpointModel


class AudioWindowTests(unittest.TestCase):
    def test_reorder_retries_and_loss(self):
        buffer = AudioWindowBuffer(start_sample=20, window_samples=8)
        self.assertEqual(buffer.push(24, torch.tensor([.3, .4])), PacketReceipt(2, 0, 0))
        self.assertEqual(buffer.push(20, torch.tensor([0., .1, float("nan"), .2])), PacketReceipt(4, 0, 0))
        self.assertEqual(buffer.push(24, torch.tensor([.3, .4])), PacketReceipt(0, 2, 0))
        with self.assertRaises(ValueError):
            buffer.push(25, torch.tensor([8., 9.]))
        buffer.seal(27)
        self.assertIsNone(buffer.pop_window())
        buffer.seal(28)
        window = buffer.pop_window()
        self.assertEqual((window.start_sample, window.end_sample), (20, 28))
        torch.testing.assert_close(window.samples, torch.tensor([0., .1, 0., .2, .3, .4, 0., 0.]))
        self.assertEqual(window.valid.tolist(), [True, True, False, True, True, True, False, False])
        self.assertEqual(window.received.tolist(), [True] * 6 + [False, False])
        self.assertEqual(buffer.push(27, torch.tensor([12., 13.])), PacketReceipt(1, 0, 1))
        with self.assertRaises(ValueError):
            buffer.seal(27)

    def test_overlapping_windows_survive_ring_wrap(self):
        buffer = AudioWindowBuffer(start_sample=17, window_samples=4, hop_samples=2, max_pending_samples=6)
        desired = torch.arange(10, dtype=torch.float32)
        buffer.push(17, desired[:6])
        with self.assertRaises(BufferError):
            buffer.push(23, desired[6:7])
        buffer.seal(21)
        first = buffer.pop_window()
        first.samples.zero_()
        buffer.push(23, desired[6:8])
        buffer.seal(25)
        torch.testing.assert_close(buffer.pop_window().samples, desired[2:6])
        torch.testing.assert_close(buffer.pop_window().samples, desired[4:8])
        self.assertIsNone(buffer.pop_window())

    def test_pcm16_absolute_sample_positions(self):
        buffer = AudioWindowBuffer(window_samples=4)
        buffer.push_pcm16le(0, struct.pack("<hhhh", -32768, -1, 0, 32767))
        buffer.seal(4)
        torch.testing.assert_close(buffer.pop_window().samples, torch.tensor([-32768, -1, 0, 32767]) / 32768)
        with self.assertRaises(ValueError):
            buffer.push_pcm16le(4, b"a")


FRAME = 320
LOUD = struct.pack("<h", 4000) * FRAME
SILENT = bytes(2 * FRAME)


class VoiceSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = RealtimeVoiceSession()
        self.offset = 0

    def feed(self, pcm, frames=1):
        events = []
        for _ in range(frames):
            if pcm is not None:
                self.session.push_pcm16le(self.offset, pcm)
            self.offset += FRAME
            events.extend(self.session.seal_input(self.offset))
        return events

    def utterance(self):
        events = self.feed(LOUD, 6)
        events += self.feed(SILENT, 15)
        return events

    def test_listen_during_playback_and_discard_unheard_text(self):
        self.feed(SILENT, 12)
        events = self.utterance()
        started = next(e for e in events if e.type == "speech_started")
        ended = events[-1]
        self.assertEqual(started.at_sample - started.data["onset_sample"], 3 * FRAME)
        self.assertEqual(ended.type, "speech_stopped")
        self.assertTrue(ended.data["response_eligible"])
        reply = self.session.start_response(ended.data["turn_id"]).data["response_id"]
        self.session.put_audio(reply, LOUD, text="heard")
        self.session.put_audio(reply, LOUD, text="unheard")
        self.session.acknowledge_playback(reply, FRAME + 20)
        self.assertEqual([e.type for e in self.session.finish_output(reply)], ["response_output_done"])
        newer = self.feed(LOUD, 3)
        cut = next(e for e in newer if e.type == "playback_clear")
        self.assertEqual(cut.data["discarded_samples"], FRAME - 20)
        self.assertEqual(cut.data["heard_text"], "heard")
        self.assertIsNone(self.session.put_audio(reply, LOUD))
        self.assertIsNone(self.session.start_response(ended.data["turn_id"]))
        new_end = self.feed(SILENT, 15)[-1]
        new_reply = self.session.start_response(new_end.data["turn_id"]).data["response_id"]
        self.assertNotEqual(new_reply, reply)
        self.session.put_audio(new_reply, LOUD, text="next")
        with self.assertRaises(ValueError):
            self.session.acknowledge_playback(new_reply, FRAME + 1)
        self.assertEqual(self.session.acknowledge_playback(new_reply, FRAME), [])
        self.assertEqual(self.session.finish_output(new_reply)[-1].data["heard_text"], "next")

    def test_no_reply_from_missing_packets_or_superseded_work(self):
        self.feed(LOUD, 6)
        stop = self.feed(None, 15)[-1]
        self.assertEqual(stop.data["reason"], "input_gap")
        self.assertIsNone(self.session.start_response(stop.data["turn_id"]))
        old_turn = self.utterance()[-1].data["turn_id"]
        newest_turn = self.utterance()[-1].data["turn_id"]
        self.assertIsNone(self.session.start_response(old_turn))
        self.assertIsNotNone(self.session.start_response(newest_turn))

    def test_transcript_policy_waits_for_a_mid_turn_pause(self):
        endpoint = TranscriptEndpointModel(0., {"tail=and": -1., "tail=tomorrow": 1.}, .5)
        self.session = RealtimeVoiceSession(endpoint_model=endpoint)
        events = self.feed(LOUD, 6)
        turn = events[0].data["turn_id"]
        self.assertTrue(self.session.report_transcript(turn, "pick today and", self.offset))
        self.assertNotIn("speech_stopped", [e.type for e in self.feed(SILENT, 18)])
        self.feed(LOUD, 6)
        self.assertTrue(self.session.report_transcript(turn, "pick today and tomorrow", self.offset))
        stop = self.feed(SILENT, 15)[-1]
        self.assertEqual(stop.type, "speech_stopped")
        self.assertEqual(stop.data["endpoint"]["rule"], "transcript_complete")
        self.assertEqual(stop.data["endpoint"]["contributions"]["tail=tomorrow"], 1.)
        self.assertFalse(self.session.report_transcript(turn, "stale turn", self.offset))

    def test_transcript_policy_falls_back_when_audio_outpaces_asr(self):
        endpoint = TranscriptEndpointModel(0., {"tail=done": 1.}, .5)
        self.session = RealtimeVoiceSession(endpoint_model=endpoint)
        turn = self.feed(LOUD, 6)[0].data["turn_id"]
        self.assertTrue(self.session.report_transcript(turn, "done", self.offset))
        self.assertFalse(self.session.report_transcript(turn, "old", self.offset - FRAME))
        self.assertFalse(self.session.report_transcript(turn, "future", self.offset + FRAME))
        self.feed(LOUD, 6)
        self.assertNotIn("speech_stopped", [e.type for e in self.feed(SILENT, 29)])
        stop = self.feed(SILENT)[-1]
        self.assertEqual(stop.data["endpoint"]["rule"], "silence_cap")
        self.assertIsNone(stop.data["endpoint"]["transcript_logit"])

    def test_transcript_policy_does_not_turn_an_input_gap_into_a_reply(self):
        self.session = RealtimeVoiceSession(endpoint_model=TranscriptEndpointModel(1., {}, 0.))
        turn = self.feed(LOUD, 6)[0].data["turn_id"]
        self.assertTrue(self.session.report_transcript(turn, "done", self.offset))
        stop = self.feed(None, 15)[-1]
        self.assertEqual(stop.data["reason"], "input_gap")
        self.assertIsNone(self.session.start_response(turn))


if __name__ == "__main__":
    unittest.main()
