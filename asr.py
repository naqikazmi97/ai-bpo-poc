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
    Fires on_partial for partial results and on_utterance_ready for finals.
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
                if self.on_partial:
                    await self.on_partial(transcript)
            else:
                self._partial_buffer = ""
                log.debug(f"[ASR] Final: {transcript}")
                await self.on_utterance_ready(transcript)


class ASRStream:
    def __init__(self, on_utterance_ready, on_partial=None):
        self.on_utterance_ready = on_utterance_ready
        self.on_partial = on_partial
        self._client = TranscribeStreamingClient(region=REGION)
        self._stream = None
        self._handler_task = None
        # Bumped on every start(). end_stream() detaches from the handler task
        # without cancelling it (see end_stream), so a previous session's task
        # can keep running briefly in the background. Wrapping callbacks with
        # the generation they were created under lets us discard any late
        # events it fires after a new session has already started.
        self._generation = 0

    def is_active(self) -> bool:
        """Return True if there is a live Transcribe stream with a running handler."""
        return (
            self._stream is not None
            and self._handler_task is not None
            and not self._handler_task.done()
        )

    def _guard(self, callback, generation):
        """Wrap a callback so it's a no-op once a newer session has started."""
        async def wrapped(text):
            if generation != self._generation:
                log.debug(f"[ASR] Discarding callback from stale session (gen {generation})")
                return
            await callback(text)
        return wrapped

    async def start(self):
        """Open a new Transcribe streaming session."""
        # Create a fresh client for each session. The SDK reuses a single HTTP/2
        # connection across all start_stream_transcription calls on the same client.
        # After a few sessions the connection accumulates stale stream state and
        # new sessions stop producing transcripts. A new client = new connection.
        self._client = TranscribeStreamingClient(region=REGION)
        self._stream = await self._client.start_stream_transcription(
            language_code=LANGUAGE_CODE,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
            enable_partial_results_stabilization=True,
            partial_results_stability="medium",
        )
        self._generation += 1
        gen = self._generation
        handler = _TranscribeHandler(
            self._stream.output_stream,
            self._guard(self.on_utterance_ready, gen),
            self._guard(self.on_partial, gen) if self.on_partial else None,
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
        """Signal end of audio to Transcribe and detach from the handler.

        Transcribe doesn't close its output stream promptly after EOS, so
        waiting for the handler task to finish naturally stalls every turn.
        We don't cancel it either — cancelling while awscrt's C thread is
        mid-delivery of a chunk races the same future and raises a logged
        (but otherwise harmless) InvalidStateError on every turn. Instead we
        just detach: the task keeps running until Transcribe closes the
        stream on its own, and the generation guard in start() (plus the
        _asr_started checks in pipeline.py) discard anything it fires after
        this point.
        """
        if self._stream:
            await self._stream.input_stream.end_stream()
            log.info("[ASR] Stream ended")
        self._handler_task = None
        self._stream = None

    async def stop(self):
        """Hard stop on disconnect."""
        if self._stream:
            try:
                await self._stream.input_stream.end_stream()
            except Exception:
                pass
        self._handler_task = None
        self._stream = None