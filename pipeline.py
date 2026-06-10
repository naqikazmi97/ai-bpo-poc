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
from llm import LLMStream, END_CALL_SENTINEL
from tts import TTSStream
from session import SessionManager

log = logging.getLogger(__name__)

# End-of-turn detection. Amazon Transcribe emits a FINAL result per speech
# SEGMENT (at natural pauses), not per conversational turn — so one spoken
# sentence arrives as several finals. We aggregate finals and commit the turn
# only after the user has been silent for UTTERANCE_SILENCE_S.
#
# This window MUST exceed a speaker's natural MID-sentence pauses (commas,
# thinking, reading an address off a bill) or long sentences get split into
# separate turns. Higher = fewer splits but more latency on every turn.
# Tune against real calls: ~1.0 feels snappy but splits deliberate speakers;
# ~1.5 is a safe default; raise toward 2.0 if long answers still split.
UTTERANCE_SILENCE_S = 1.5

# When the silence timer fires but Transcribe is still mid-segment (a partial
# arrived after the last final and hasn't finalized yet), wait this much
# longer for the final instead of committing a half-recognized tail. Repeated
# up to MAX_GRACE_ROUNDS so a stuck partial can't block the turn forever.
PARTIAL_GRACE_S = 0.4
MAX_GRACE_ROUNDS = 6


class VoicePipeline:
    def __init__(self, session_id: str, websocket: WebSocket):
        
        self.session_id = session_id
        self.websocket = websocket
        self.session = SessionManager(session_id)

        # Each stage fires a callback into the next stage
        self.tts = TTSStream(on_audio_ready=self._send_audio)
        self.llm = LLMStream(session=self.session, on_sentence_ready=self._on_llm_sentence)
        self.asr = ASRStream(
            on_utterance_ready=self._on_asr_final,
            on_partial=self._on_asr_partial,
        )

        self._asr_started = False
        self._current_bot_response = []
        self._extraction_done = False  # Guard: only extract once per session
        self._bot_turn_lock = asyncio.Lock()
        self._bot_ending_call = False  # Set when LLM emits END_CALL sentinel
        self._bot_speaking = False

        # ── End-of-turn aggregation ──
        # Finals are buffered here and only committed as one user turn after
        # UTTERANCE_SILENCE_S of silence (see _commit_utterance).
        self._utterance_parts = []
        self._commit_handle = None       # asyncio.TimerHandle for the silence timer
        self._partial_pending = False    # True while Transcribe is mid-segment
        self._last_partial_text = ""     # latest partial, used as anti-truncation fallback
        self._grace_rounds = 0           # consecutive waits for a pending partial

    # ── Public interface ──────────────────────────────────────────

    async def feed_audio(self, chunk: bytes):
        """Receive raw PCM audio from client and push to ASR."""
        # Reopen whenever there is no live Transcribe stream — not just when the
        # flag is False. The mic-restart handshake relies on the frontend's
        # end_of_audio to flip _asr_started back; if that signal is ever missed
        # (e.g. a phantom asr_result lands while the mic is mid-restart), the
        # flag can stick True while the stream is dead, and every chunk would be
        # silently dropped. Keying off the real stream state self-heals that.
        if not self._asr_started or not self.asr.is_active():
            await self._start_pipeline()
        await self.asr.feed(chunk)

    async def end_audio(self):
        """Client signalled end of speech turn."""
        log.info(f"[{self.session_id}] End of audio signal")
        self._reset_utterance_state()
        # Set False BEFORE end_stream. The handler task runs inside end_stream
        # and can fire late finals; _on_asr_final/_on_asr_partial check this
        # flag and bail out, preventing a stale commit timer from arming and
        # later sending a ghost asr_result that closes the next turn's stream.
        self._asr_started = False
        await self.asr.end_stream()
        self._reset_utterance_state()  # belt-and-suspenders: cancel any timer that slipped through

    async def clear_session(self):
        """Reset conversation history — does not trigger extraction."""
        self._reset_utterance_state()
        self.session.clear()
        self._current_bot_response = []
        self._extraction_done = False
        log.info(f"[{self.session_id}] Session cleared")

    async def end_session(self, bot_initiated: bool = False):
        """
        Explicit end_session from client (user pressed 'End Call') or
        bot-initiated (bot said goodbye).
        Runs extraction on full conversation, then saves.
        """
        log.info(f"[{self.session_id}] Ending session — bot_initiated={bot_initiated}")
        await self._extract_and_save()
        await self._send_control({
            "type": "session_ended",
            "bot_initiated": bot_initiated,
            "slots": self.session.get_slots()
        })

    async def cleanup(self, already_ended: bool = False):
        """
        Called on disconnect (clean or abrupt).
        Runs extraction as safety net if end_session wasn't called.
        """
        log.info(f"[{self.session_id}] cleanup called, already_ended={already_ended}")
        self._reset_utterance_state()
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
        # Set the flag BEFORE the network call so a second concurrent feed_audio
        # that also sees _asr_started=False cannot race in and open a second stream.
        self._asr_started = True
        await self.asr.start()
        self.tts.start()

    def _cancel_commit_timer(self):
        if self._commit_handle is not None:
            self._commit_handle.cancel()
            self._commit_handle = None

    def _reset_utterance_state(self):
        self._cancel_commit_timer()
        self._utterance_parts = []
        self._partial_pending = False
        self._last_partial_text = ""
        self._grace_rounds = 0

    def _arm_commit_timer(self, delay: float):
        """Arm the end-of-turn timer for `delay` seconds, replacing any pending one."""
        self._cancel_commit_timer()
        loop = asyncio.get_event_loop()
        self._commit_handle = loop.call_later(
            delay,
            lambda: asyncio.ensure_future(self._commit_utterance()),
        )

    async def _on_asr_partial(self, text: str):
        """
        Transcribe produced a partial (the user is actively mid-segment).
        Hold the turn open and remember the text so a slow finalization can't
        truncate the tail. Also forward to the frontend so it can show interim
        text and reset its silence timer while the user is actively speaking.
        """
        if not self._asr_started:
            return
        self._partial_pending = True
        self._last_partial_text = text
        self._grace_rounds = 0
        self._arm_commit_timer(UTTERANCE_SILENCE_S)
        await self._send_control({"type": "asr_partial", "text": text})

    async def _on_asr_final(self, text: str):
        """
        Transcribe finalized a SEGMENT (not necessarily the whole turn).
        Buffer it and wait for silence before committing the turn.
        """
        if not self._asr_started:
            log.debug(f"[{self.session_id}] Discarding late final (stream ended): {text!r}")
            return
        self._utterance_parts.append(text)
        self._partial_pending = False
        self._last_partial_text = ""
        self._grace_rounds = 0
        self._arm_commit_timer(UTTERANCE_SILENCE_S)

    async def _commit_utterance(self):
        """
        Fired after UTTERANCE_SILENCE_S of silence. Joins the buffered segments
        into one user turn and runs it — unless Transcribe is still mid-segment,
        in which case we wait a little longer for the final (anti-truncation).
        """
        self._commit_handle = None

        # Guard: timer may have been armed just before end_audio() set _asr_started=False.
        if not self._asr_started:
            log.debug(f"[{self.session_id}] Stale commit discarded (turn ended)")
            return

        # A partial arrived after the last final and hasn't finalized yet —
        # the user's last words are still being recognized. Give it more time
        # rather than committing a truncated turn.
        if self._partial_pending and self._grace_rounds < MAX_GRACE_ROUNDS:
            self._grace_rounds += 1
            self._arm_commit_timer(PARTIAL_GRACE_S)
            return

        parts = list(self._utterance_parts)
        # If a partial never finalized (grace exhausted), don't lose its text.
        if self._partial_pending and self._last_partial_text.strip():
            parts.append(self._last_partial_text)

        self._utterance_parts = []
        self._partial_pending = False
        self._last_partial_text = ""
        self._grace_rounds = 0

        full_text = " ".join(p.strip() for p in parts if p.strip()).strip()
        if not full_text:
            return

        await self._on_user_spoke(full_text)

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
        If the END_CALL_SENTINEL arrives, trigger end_session once TTS drains.
        """
        # Sentinel — not a real sentence, just a signal that the call is over.
        if sentence == END_CALL_SENTINEL:
            log.info(f"[{self.session_id}] END_CALL sentinel received — will end session after TTS drains")
            self._bot_ending_call = True
            return

        log.info(f"[{self.session_id}] LLM sentence: {sentence}")
        self._current_bot_response.append(sentence)

        await self._send_control({
            "type": "llm_sentence",
            "text": sentence
        })

        await self.tts.synthesize(sentence)

    async def _send_audio(self, audio_bytes: bytes):
        """Send PCM audio chunk to client as binary WebSocket frame."""
        log.info(f"[Pipeline] Sending audio chunk: {len(audio_bytes)} bytes")
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
        self._bot_speaking = False
        await self._send_control({"type": "turn_end"})

        # If the bot's last response contained the closing phrase, end the session now.
        if self._bot_ending_call:
            self._bot_ending_call = False
            log.info(f"[{self.session_id}] Bot-initiated end — running end_session automatically")
            await self.end_session(bot_initiated=True)

    async def handle_silence_timeout(self):
        self._reset_utterance_state()
        # Cleanly close the ASR stream before starting bot response.
        # The frontend sends end_of_audio after silence_timeout, but it can
        # race with this coroutine. Close it here explicitly so the stream
        # is always torn down before the next one opens.
        if self._asr_started:
            self._asr_started = False
            await self.asr.end_stream()
        async with self._bot_turn_lock:
            log.info(f"[{self.session_id}] Silence timeout — prompting user")
            self._current_bot_response = []
            self._bot_speaking = True
            await self.llm.stream_response("USER_SILENT")
            await self._on_turn_complete()

    async def bot_start(self):
        async with self._bot_turn_lock:
            self.session.clear()
            self.tts.start()
            self._bot_speaking = True
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