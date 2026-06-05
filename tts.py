"""
tts.py — Amazon Polly Neural TTS handler
Serial queue ensures audio chunks play in order.
Synthesizes per sentence — first audio starts before LLM finishes.
"""
import asyncio
import logging
import boto3

log = logging.getLogger(__name__)

REGION = "us-east-1"
VOICE_ID = "Danielle"
ENGINE = "neural"
OUTPUT_FORMAT = "pcm"
SAMPLE_RATE = "16000"


class TTSStream:
    def __init__(self, on_audio_ready):
        self.on_audio_ready = on_audio_ready
        self._client = boto3.client("polly", region_name=REGION)
        self._queue = asyncio.Queue()
        self._worker_task = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def synthesize(self, sentence: str):
        if not self._running:
            self.start()
        await self._queue.put(sentence)

    async def _worker(self):
        pending_audio = None  # pre-fetched audio for next sentence

        while self._running:
            try:
                sentence = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if sentence is None:
                break

            try:
                # Start synthesizing the NEXT sentence immediately in background
                # so it's ready by the time this one finishes playing
                next_task = None

                # If we already pre-fetched audio for this sentence, use it
                if pending_audio is not None:
                    audio = pending_audio
                    pending_audio = None
                else:
                    audio = await asyncio.to_thread(self._synthesize_sync, sentence)

                # Peek at the next item in the queue and pre-fetch it in parallel
                try:
                    next_sentence = self._queue.get_nowait()
                    if next_sentence is not None:
                        next_task = asyncio.create_task(
                            asyncio.to_thread(self._synthesize_sync, next_sentence)
                        )
                    else:
                        self._queue.task_done()
                except asyncio.QueueEmpty:
                    next_sentence = None

                # Play current audio while next is synthesizing in background
                if audio:
                    await self.on_audio_ready(audio)

                # Wait for pre-fetched audio and put it ready for next iteration
                if next_task is not None:
                    pending_audio = await next_task
                    # Manually mark task_done for the peeked sentence
                    self._queue.task_done()
                    # Re-queue the pre-fetched sentence so the loop picks it up normally
                    # but we already have its audio in pending_audio
                    await self._queue.put(next_sentence)

            except Exception as e:
                log.error(f"[TTS] Synthesis error: {e}")
            finally:
                self._queue.task_done()

    def _synthesize_sync(self, text: str) -> bytes:
        log.debug(f"[TTS] Synthesizing: {text[:50]}...")
        response = self._client.synthesize_speech(
            Text=text,
            OutputFormat=OUTPUT_FORMAT,
            VoiceId=VOICE_ID,
            Engine=ENGINE,
            SampleRate=SAMPLE_RATE
        )
        return response["AudioStream"].read()

    async def flush(self):
        """Drain queue on barge-in."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        log.info("[TTS] Queue flushed")

    async def stop(self):
        self._running = False
        await self._queue.put(None)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=3.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
        log.info("[TTS] Worker stopped")
