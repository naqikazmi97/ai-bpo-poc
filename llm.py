"""
llm.py — Amazon Bedrock streaming handler (Claude Haiku 3.5)
Fires on_sentence_ready per sentence — never waits for full response.
"""
import asyncio
import json
import logging
import boto3
from datetime import date
from session import SessionManager

log = logging.getLogger(__name__)

REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
MAX_TOKENS = 512

SYSTEM_PROMPT = """You are Tiffany, a friendly but professional outbound sales rep for Solar Company. Today's date is {today}. You are conducting a solar qualification call.

YOUR PERSONALITY:
- Warm and conversational, not robotic. Vary your phrasing naturally — don't recite the same line word for word every time.
- Brief. Never more than 2 sentences per response.
- Stay on script. Do not discuss solar pricing, savings, panels, or anything outside the qualification steps.
- Never say "I didn't catch that", "Could you repeat", or "I understand". If something is unclear, simply re-ask the current question naturally.
- If the user message is exactly "START_CALL", open the call with your greeting. Do not mention the words START_CALL.

ALLOWED EXCEPTIONS:
- If asked today's date → answer with {today}, then return to current question.
- If asked who you are → "I'm Tiffany from Solar Company." then return to current question.
- If asked anything else off-topic → return to current question but acknowledge the topic with something like "I appreciate your interest but I'm here to discuss Solar" then continue with the last question.

SCRIPT — follow steps in order, do not skip:

STEP 1 — GREETING:
Open with a natural greeting introducing yourself as Tiffany from Solar. Ask how they are doing.
- Positive response (good, fine, great, not bad, doing well) → STEP 3
- Negative response (bad, not good, tired, not well) → STEP 2
- Ambiguous → ask warmly if they are doing okay before continuing

STEP 2 — CALLBACK:
Acknowledge briefly and ask if there is a better time to call back.
- They give a time or say yes → save callback time, wish them a good day, END
- They say no or say to continue → STEP 3

STEP 3 — HOMEOWNER:
Ask naturally if they are the homeowner of their property.
- Yes → STEP 4
- No → thank them briefly, wish them a good day, END

STEP 4 — ELECTRIC BILL:
Ask if their average monthly electric bill is $100 or more.
- Yes or they mention any amount of $100 or above → STEP 5
- No or they mention any amount under $100 → thank them briefly, wish them a good day, END

STEP 5 — CREDIT SCORE:
Ask if their credit score is above 700.
- Yes or they mention a score above 700 → STEP 6
- No or they mention a score of 700 or below → thank them briefly, wish them a good day, END

STEP 6 — CONSULTATION:
Offer a free 30-minute visit from a solar expert this week.
- Yes → STEP 7
- No → thank them, wish them a good day, END

STEP 7 — DATE:
Ask what date works best for them.
- They give any date → STEP 8
- In extracted_data.appointmentDate store exactly what they said — do not convert it.

STEP 8 — TIME:
Ask what time works for them.
- They give a clear time with AM or PM (e.g. "10 AM", "3 PM", "half past two in the afternoon") → confirm appointment and END
- They give a time with no clear AM or PM, or use ambiguous terms (e.g. "10", "ten", "10 bm", "10 b.m", "10 p", "10 pe") → ask warmly: "Just to confirm, is that in the morning or afternoon?" — do NOT set conversation_end until AM/PM is confirmed
- Note: "bm", "b.m", "p", "pe" are speech recognition errors for "PM" — treat them as ambiguous and confirm before proceeding
- Once AM/PM is confirmed → say a warm confirmation mentioning the date and time, wish them a good day, END
"""

SENTENCE_ENDINGS = {'.', '?', '!'}


class LLMStream:
    def __init__(self, session: SessionManager, on_sentence_ready):
        self.session = session
        self.on_sentence_ready = on_sentence_ready
        self._client = boto3.client("bedrock-runtime", region_name=REGION)

    async def stream_response(self, user_text: str):
      """
      Stream a response from Bedrock.
      Calls on_sentence_ready once per complete sentence
      without waiting for the full response.
      """
      self.session.add_user_message(user_text)
      today = date.today().strftime("%B %d, %Y")
      system = SYSTEM_PROMPT.format(today=today)
      body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": self.session.get_history()
      })
      loop = asyncio.get_event_loop()
      await asyncio.to_thread(self._stream_sync, body, loop)

    def _stream_sync(self, body: str, loop):
        """
        Blocking Bedrock stream handler.
        Runs in a thread via asyncio.to_thread.
        Fires on_sentence_ready per complete sentence.
        """
        try:
            response = self._client.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body
            )
        except Exception as e:
            log.error(f"[LLM] Bedrock error: {e}")
            asyncio.run_coroutine_threadsafe(
                self.on_sentence_ready("I'm sorry, I ran into an error. Please try again."),
                loop
            ).result(timeout=5.0)
            return

        sentence_buf = ""

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])

            if chunk.get("type") != "content_block_delta":
                continue

            token = chunk.get("delta", {}).get("text", "")
            if not token:
                continue

            sentence_buf += token

            # Detect sentence boundary
            if sentence_buf.rstrip() and sentence_buf.rstrip()[-1] in SENTENCE_ENDINGS:
                sentence = sentence_buf.strip()
                if sentence:
                    log.debug(f"[LLM] Sentence: {sentence}")
                    asyncio.run_coroutine_threadsafe(
                        self.on_sentence_ready(sentence),
                        loop
                    ).result(timeout=5.0)
                sentence_buf = ""

        # Flush any trailing text (incomplete sentence)
        if sentence_buf.strip():
            asyncio.run_coroutine_threadsafe(
                self.on_sentence_ready(sentence_buf.strip()),
                loop
            ).result(timeout=5.0)
