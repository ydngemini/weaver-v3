"""Transport and playback-clock regressions without a physical device."""
from collections import deque
from types import SimpleNamespace
import unittest

from live_voice import BLOCK_BYTES, FRAME, SILENCE, DuplexTransport, Playback, VoiceWorker


class DeviceTransportTests(unittest.TestCase):
    def test_optional_capture_preserves_sample_offsets_during_input_loss(self):
        transport = DuplexTransport()
        pcm = b"\x10\x00" * FRAME
        transport.exchange(pcm, .98, 1.02, 1.)
        transport.exchange(None, 1., 1.04, 1.02)
        captured = []
        worker = VoiceWorker.__new__(VoiceWorker)
        worker.capture = captured.append
        worker.log = lambda row: None
        worker.pending = deque()
        worker.cancelled = None
        worker.active_id = None
        worker.session = SimpleNamespace(push_pcm16le=lambda offset, samples: None,
                                         seal_input=lambda end: [])
        for block in transport.blocks:
            worker.consume(block)
        self.assertEqual(captured, [pcm, SILENCE])

    def test_callback_mutes_on_next_block_and_never_reuses_old_audio(self):
        transport = DuplexTransport()
        stamp = SimpleNamespace(inputBufferAdcTime=.980, outputBufferDacTime=1.020, currentTime=1.)
        status = SimpleNamespace(input_overflow=False, output_underflow=False, output_overflow=False)
        output = bytearray(BLOCK_BYTES)
        transport.plan = Playback(1, b"\x10\x00" * (FRAME * 3))
        transport.callback(SILENCE, output, FRAME, stamp, status)
        self.assertNotEqual(output, SILENCE)
        self.assertEqual(transport.blocks[-1].response_id, 1)
        transport.plan = None
        transport.callback(SILENCE, output, FRAME, stamp, status)
        self.assertEqual(output, SILENCE)
        self.assertIsNone(transport.blocks[-1].response_id)

    def test_lost_capture_and_xrun_keep_distinct_evidence(self):
        transport = DuplexTransport()
        stamp = SimpleNamespace(inputBufferAdcTime=.980, outputBufferDacTime=1.020, currentTime=1.)
        status = SimpleNamespace(input_overflow=True, output_underflow=False, output_overflow=False)
        out = bytearray(BLOCK_BYTES)
        transport.callback(SILENCE, out, FRAME, stamp, status)
        self.assertIsNone(transport.blocks[-1].pcm)
        status.output_underflow = True
        transport.callback(SILENCE, out, FRAME, stamp, status)
        self.assertIn("xrun", transport.error)
        self.assertEqual(out, SILENCE)

    def test_device_acknowledges_only_samples_whose_dac_time_has_passed(self):
        transport = DuplexTransport()
        transport.plan = Playback(3, b"\x10\x00" * FRAME)
        transport.exchange(SILENCE, .98, 1.02, 1.)
        observed = []
        worker = VoiceWorker.__new__(VoiceWorker)
        worker.active_id = 3
        worker.acknowledged = 0
        worker.pending = deque(transport.blocks)
        worker.session = SimpleNamespace(acknowledge_playback=lambda ident, count:
                                         observed.append((ident, count)) or [])
        worker.acknowledge(1.)
        self.assertEqual(observed, [])
        worker.acknowledge(1.03)
        self.assertEqual(observed, [(3, FRAME // 2)])
        worker.acknowledge(1.05)
        self.assertEqual(observed[-1], (3, FRAME))
        self.assertFalse(worker.pending)
