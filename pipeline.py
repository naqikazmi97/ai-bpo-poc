"""
pipeline.py — Core streaming pipeline
Chains ASR → LLM → TTS without waiting for any stage to complete.
Slot extraction runs once at session end — not per turn.
"""
import asyncio
import json
import logging
from fastapi import WebSocket
from asr import ASRStream
from llm import LLMStream
from tts import TTSStream
from session import SessionManager

log = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(self, session_id: str, websocket: WebSocket):
        
        self.session_id = session_id
        self.websocket = websocket
        self.session = SessionManager(session_id)

        # Each stage fires a callback into the next stage
        self.tts = TTSStream(on_audio_ready=self._send_audio)
        self.llm = LLMStream(session=self.session, on_sentence_ready=self._on_llm_sentence)
        self.asr = ASRStream(on_utterance_ready=self._on_user_spoke)

        self._asr_started = False
        self._current_bot_response = []
        self._extraction_done = False  # Guard: only extract once per session
        self._bot_turn_lock = asyncio.Lock()

    # ── Public interface ──────────────────────────────────────────

    async def feed_audio(self, chunk: bytes):
        """Receive raw PCM audio from client and push to ASR."""
        if not self._asr_started:
            await self._start_pipeline()
        await self.asr.feed(chunk)

    async def end_audio(self):
        """Client signalled end of speech turn."""
        log.info(f"[{self.session_id}] End of audio signal")
        await self.asr.end_stream()
        self._asr_started = False

    async def clear_session(self):
        """Reset conversation history — does not trigger extraction."""
        self.session.clear()
        self._current_bot_response = []
        self._extraction_done = False
        log.info(f"[{self.session_id}] Session cleared")

    async def end_session(self):
        """
        Explicit end_session from client (user pressed 'End Call').
        Runs extraction on full conversation, then saves.
        """
        log.info(f"[{self.session_id}] Ending session — running extraction")
        await self._extract_and_save()
        await self._send_control({
            "type": "session_ended",
            "slots": self.session.get_slots()
        })

    async def cleanup(self, already_ended: bool = False):
        """
        Called on disconnect (clean or abrupt).
        Runs extraction as safety net if end_session wasn't called.
        """
        log.info(f"[{self.session_id}] cleanup called, already_ended={already_ended}")
        await self.asr.stop()
        await self.tts.stop()

        if not already_ended:
            # Abrupt disconnect — still extract what we have
            history = self.session.get_history()
            if len(history) >= 2:
                log.info(f"[{self.session_id}] Abrupt disconnect — running extraction as fallback")
                await self._extract_and_save()

    # ── Internal callbacks ─────────────────────────────────────────

    async def _start_pipeline(self):
        await self.asr.start()
        self.tts.start()
        self._asr_started = True

    async def _on_user_spoke(self, text: str):
        """
        Transcribe produced a final utterance.
        Immediately start LLM streaming — don't wait.
        """
        async with self._bot_turn_lock:
            log.info(f"[{self.session_id}] User: {text}")

            await self._send_control({
                "type": "asr_result",
                "text": text
            })

            self._current_bot_response = []
            await self.llm.stream_response(text)
            await self._on_turn_complete()

    async def _on_llm_sentence(self, sentence: str):
        """
        LLM produced a complete sentence.
        Queue for TTS immediately — don't wait for full LLM response.
        """
        log.info(f"[{self.session_id}] LLM sentence: {sentence}")
        self._current_bot_response.append(sentence)

        await self._send_control({
            "type": "llm_sentence",
            "text": sentence
        })

        await self.tts.synthesize(sentence)

    async def _send_audio(self, audio_bytes: bytes):
        """Send PCM audio chunk to client as binary WebSocket frame."""
        await self.websocket.send_bytes(audio_bytes)

    async def _send_control(self, payload: dict):
        """Send a JSON control message to the client."""
        await self.websocket.send_text(json.dumps(payload))

    async def _on_turn_complete(self):
        full_response = " ".join(self._current_bot_response)
        self.session.add_assistant_message(full_response)
        # Wait for TTS to finish synthesizing ALL sentences
        try:
            await asyncio.wait_for(self.tts._queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            log.warning(f"[{self.session_id}] TTS queue join timed out")
        # Small buffer to ensure last audio chunk is sent to client
        await asyncio.sleep(0.3)
        await self._send_control({"type": "turn_end"})

    async def bot_start(self):
        async with self._bot_turn_lock:
            self.session.clear()
            self.tts.start()
            await self.llm.stream_response("START_CALL")
            await self._on_turn_complete()
    # ── Extraction ─────────────────────────────────────────────────

    async def _extract_and_save(self):
        """
        Run slot extraction on full conversation history.
        Called exactly once — at session end or abrupt disconnect.
        Guards against double-extraction.
        """
        if self._extraction_done:
            log.info(f"[{self.session_id}] Extraction already done — skipping")
            return

        self._extraction_done = True
        history = self.session.get_history()

        if not history:
            log.info(f"[{self.session_id}] No history — skipping extraction")
            return

        try:
            from slots import SlotExtractor
            extractor = SlotExtractor()
            slots = await extractor.extract(history)
            self.session.save_slots(slots)
            log.info(f"[{self.session_id}] Extraction complete: {slots}")
        except Exception as e:
            log.error(f"[{self.session_id}] Extraction failed: {e}", exc_info=True)

