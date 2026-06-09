"""
asr.py — Amazon Transcribe Streaming handler
Uses partial results to detect sentence boundaries fast.
"""
import asyncio
import logging
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

log = logging.getLogger(__name__)

REGION = "us-east-1"
SAMPLE_RATE = 16000
LANGUAGE_CODE = "en-US"


class _TranscribeHandler(TranscriptResultStreamHandler):
    """
    Receives transcript events from Transcribe.
    Fires on_utterance_ready only on FINAL (non-partial) results.
    """
    def __init__(self, stream, on_utterance_ready, on_partial=None):
        super().__init__(stream)
        self.on_utterance_ready = on_utterance_ready
        self.on_partial = on_partial
        self._partial_buffer = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results

        for result in results:
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()
            if not transcript:
                continue

            if result.is_partial:
                self._partial_buffer = transcript
                # Forward partials so the pipeline can treat ongoing speech as
                # activity and hold the turn open (endpointing).
                if self.on_partial:
                    await self.on_partial(transcript)
            else:
                self._partial_buffer = ""
                log.debug(f"[ASR] Final: {transcript}")
                await self.on_utterance_ready(transcript)


class ASRStream:
    def __init__(self, on_utterance_ready, on_partial=None, sample_rate=SAMPLE_RATE):
        self.on_utterance_ready = on_utterance_ready
        self.on_partial = on_partial
        self._client = TranscribeStreamingClient(region=REGION)
        self._stream = None
        self._handler_task = None
        self._sample_rate = sample_rate
        # Tracking for audio validation
        self._total_bytes_received = 0
        self._chunk_count = 0
        self._min_chunk = float("inf")
        self._max_chunk = 0

    async def start(self):
        """Open a new Transcribe streaming session."""
        # Reset audio validation stats for the new stream
        self._total_bytes_received = 0
        self._chunk_count = 0
        self._min_chunk = float("inf")
        self._max_chunk = 0

        # CRITICAL: Cancel any handler task from a PREVIOUS stream before
        # starting a new one. If the old handler task is still running (e.g.
        # waiting for trailing events from Transcribe after end_stream), it
        # will fire on_utterance_ready / on_partial into the current pipeline
        # callbacks — corrupting the new stream with stale events.
        if self._handler_task is not None and not self._handler_task.done():
            log.warning("[ASR] Cancelling stale handler task from previous stream")
            self._handler_task.cancel()
            try:
                await asyncio.wait_for(self._handler_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._handler_task = None

        log.info(f"[ASR] Starting stream — sample_rate={self._sample_rate}, encoding=pcm")
        self._stream = await self._client.start_stream_transcription(
            language_code=LANGUAGE_CODE,
            media_sample_rate_hz=self._sample_rate,
            media_encoding="pcm",
            enable_partial_results_stabilization=True,
            partial_results_stability="high",
        )

        handler = _TranscribeHandler(
            self._stream.output_stream,
            self.on_utterance_ready,
            self.on_partial,
        )

        self._handler_task = asyncio.create_task(handler.handle_events())
        log.info("[ASR] Stream started")

    def is_active(self) -> bool:
        """True when a live Transcribe stream is open and able to accept audio."""
        return self._stream is not None

    async def feed(self, audio_chunk: bytes):
        """Push a raw PCM chunk into Transcribe."""
        if self._stream:
            # ── Audio validation ─────────────────────────────────────
            self._total_bytes_received += len(audio_chunk)
            self._chunk_count += 1
            self._min_chunk = min(self._min_chunk, len(audio_chunk))
            self._max_chunk = max(self._max_chunk, len(audio_chunk))

            # PCM at sample_rate: each sample = 2 bytes (16-bit)
            # Expected chunk sizes are multiples of frame boundaries
            bytes_per_ms = self._sample_rate * 2 // 1000  # e.g. 32 at 16kHz
            if len(audio_chunk) % 2 != 0:
                log.warning(f"[ASR] Odd-sized chunk ({len(audio_chunk)} bytes) — possible corruption")
            if len(audio_chunk) < bytes_per_ms * 10:
                log.warning(f"[ASR] Very small chunk ({len(audio_chunk)} bytes, <10ms)")

            await self._stream.input_stream.send_audio_event(
                audio_chunk=audio_chunk
            )

    async def end_stream(self):
        """Signal end of audio to Transcribe."""
        if self._stream:
            await self._stream.input_stream.end_stream()
            log.info("[ASR] Stream ended")

        # Log audio validation summary
        if self._chunk_count > 0:
            avg_chunk = self._total_bytes_received / self._chunk_count
            log.info(
                f"[ASR] Audio stats: {self._chunk_count} chunks, "
                f"{self._total_bytes_received} total bytes, "
                f"min={self._min_chunk}, max={self._max_chunk}, "
                f"avg={avg_chunk:.0f}"
            )

        if self._handler_task:
            try:
                # Short timeout — don't block the pipeline for more than 500ms.
                # The handler task may be waiting for Transcribe's final events;
                # those are best-effort and not worth blocking a new stream for.
                await asyncio.wait_for(self._handler_task, timeout=0.5)
            except asyncio.TimeoutError:
                # Transcribe still hasn't closed the output stream. CANCEL the
                # handler here instead of orphaning it: once we null the
                # reference below, start()'s stale-handler guard can no longer
                # see it, so an orphaned task would keep firing stale
                # on_utterance_ready / on_partial events into the NEXT turn's
                # pipeline — producing phantom asr_results that desync the mic
                # handshake and leave subsequent audio untranscribed.
                self._handler_task.cancel()
                try:
                    await self._handler_task
                except (asyncio.CancelledError, Exception):
                    pass
            except Exception as e:
                # Transcribe sends BadRequestException if no audio arrived for 15s
                # (happens when VAD held audio back). Treat as normal stream close.
                log.debug(f"[ASR] Handler task ended: {e}")
            self._handler_task = None

        self._stream = None

    async def stop(self):
        """Hard stop on disconnect."""
        if self._stream:
            try:
                await self._stream.input_stream.end_stream()
            except Exception:
                pass
        if self._handler_task:
            self._handler_task.cancel()
        self._stream = None
        self._handler_task = None
