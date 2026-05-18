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
    def __init__(self, stream, on_utterance_ready):
        super().__init__(stream)
        self.on_utterance_ready = on_utterance_ready
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
            else:
                self._partial_buffer = ""
                log.debug(f"[ASR] Final: {transcript}")
                await self.on_utterance_ready(transcript)


class ASRStream:
    def __init__(self, on_utterance_ready):
        self.on_utterance_ready = on_utterance_ready
        self._client = TranscribeStreamingClient(region=REGION)
        self._stream = None
        self._handler_task = None

    async def start(self):
        """Open a new Transcribe streaming session."""
        self._stream = await self._client.start_stream_transcription(
            language_code=LANGUAGE_CODE,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
            enable_partial_results_stabilization=True,
            partial_results_stability="high",
        )

        handler = _TranscribeHandler(
            self._stream.output_stream,
            self.on_utterance_ready
        )

        self._handler_task = asyncio.create_task(handler.handle_events())
        log.info("[ASR] Stream started")

    async def feed(self, audio_chunk: bytes):
        """Push a raw PCM chunk into Transcribe."""
        if self._stream:
            await self._stream.input_stream.send_audio_event(
                audio_chunk=audio_chunk
            )

    async def end_stream(self):
        """Signal end of audio to Transcribe."""
        if self._stream:
            await self._stream.input_stream.end_stream()
            log.info("[ASR] Stream ended")

        if self._handler_task:
            try:
                await asyncio.wait_for(self._handler_task, timeout=5.0)
            except asyncio.TimeoutError:
                log.warning("[ASR] Handler task timeout")
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
