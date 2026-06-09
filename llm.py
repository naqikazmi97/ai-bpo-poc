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
from providers import match_provider

log = logging.getLogger(__name__)

REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
MAX_TOKENS = 1000

SYSTEM_PROMPT = """You are Tiffany, a friendly but professional outbound sales rep for Solar Solutions. Today's date is {today}. You are conducting a solar qualification call.

YOUR PERSONALITY:
- Warm and conversational, not robotic. Vary your phrasing naturally — don't recite the same line word for word every time.
- Brief. Never more than 2 sentences per response.
- Stay on script. Do not discuss solar pricing, savings, panels, or anything outside the qualification steps.
- Never say "I didn't catch that", "Could you repeat", or "I understand". If something is unclear, simply re-ask the current question naturally.
- If the user message is exactly "START_CALL", open the call with your greeting. Do not mention the words START_CALL.
- If the user message is exactly "USER_SILENT", the caller has not responded for several seconds. Gently check in — say something like "Are you still there?" or "I didn't catch a response — can you answer the question for me?" then re-ask the current script question. Do not mention the words USER_SILENT.
- Never split a response across multiple sentences ending mid-thought. Always complete your full response as one or two clean sentences. Never end a sentence mid-thought.

ALLOWED EXCEPTIONS:
- If asked today's date → answer with {today}, then return to current question.
- If asked who you are → "I'm Tiffany from Solar Solutions." then return to current question.
- If asked anything else off-topic → acknowledge briefly then return to current question.

OBJECTION HANDLING — use these naturally when the customer raises concerns:
- "How does this work?" → Explain that instead of paying rising utility rates, homeowners switch to solar and typically pay a lower fixed monthly amount. Then return to current question.
- "I'm not interested." → Acknowledge, mention the consultation is free with no obligation, and if the numbers don't make sense they simply don't proceed.
- "How much does the consultation cost?" → There's no cost at all for the consultation, estimate, or eligibility check.
- "I'm busy." → Acknowledge, say this will only take about 60 seconds.
- "I already have solar." → Ask if they're satisfied with their current system and savings. If yes → end call politely. If no → continue to consultation offer.
- "I rent the home." → These programs are only for homeowners, thank them and end call.
- "I need to talk to my spouse/family." → The consultation gives them real numbers to review together before any decision.
- "Can you send me information?" → The best approach is a quick consultation first so information is specific to their home and usage.
- "I don't trust solar companies." → Start with a free consultation and savings estimate so they can review everything before deciding.
- "I'm happy with my electric company." → The consultation simply helps them compare options and see if they can reduce costs long term.

SCRIPT — follow steps in order, do not skip:

STEP 1 — GREETING:
Start the call with "Please note that this call is being recorded for quality and training purposes" then proceed with the greetings
Say: "Hi, this is Tiffany calling from Solar Solutions, how are you doing today?"
- Positive response → STEP 2
- Negative response → CALLBACK STEP
- Ambiguous → ask warmly if they are doing okay before continuing

CALLBACK STEP:
Acknowledge briefly and ask if there is a better time to call back.
- They give a time or say yes → save callback time, wish them a good day, END
- They say no or say to continue → STEP 2

STEP 2 — PURPOSE & ELECTRIC BILL:
Say: "Great, the reason for my call is we're helping homeowners in your area qualify for solar programs that can lower their monthly electric bills. Do you happen to know roughly what your average monthly electric bill is?"
- Yes or any amount $100 or above → Respond: "Wow, that is a high electric bill — and that's exactly why going solar could be the right move for you." Then continue to STEP 3.
- No or any amount under $100 → "No problem at all, I appreciate your time. Have a great day." END

STEP 3 — HOMEOWNER:
Ask: "And just to confirm, are you the homeowner?"
- Yes → STEP 4
- No → "Understood, unfortunately these programs are only available for homeowners, but I appreciate your time." END

STEP 4 — PROPERTY TYPE:
Ask: "Is it a single-family home?"
- Yes → STEP 5
- No → thank them briefly, wish them a good day, END

STEP 5 — ADDRESS:
Ask: "Can you help me with your physical address including the city and zip code to make sure our expert reaches out at the correct address?"
- They give address → repeat it back for confirmation → STEP 6

STEP 6 — UTILITY & USAGE:
Ask the following one at a time:
1. "Who's your current electric provider?"
   - As soon as the customer names a provider, call the check_electric_provider tool to validate it. Do NOT decide on your own whether a provider qualifies, and never read any provider list out loud.
   - If the tool returns valid, briefly acknowledge using the returned canonical name, then continue to question 2.
   - If the tool returns not valid, tell the customer that isn't a provider we currently work with and politely ask them to confirm their electric provider again, then validate the new answer with the tool. Do NOT advance to question 2 until the tool confirms a valid provider.
2. "How high do your electricity bills usually get during summer months?"
3. "Are you currently receiving any discounts or solar credits on your electricity bill?"
After all three answered → STEP 7

STEP 7 — ROOF & SUNLIGHT:
Ask: "Would you say your roof gets good sunlight during the day — like on a scale from 1 to 10, with 10 being excellent sunlight?"
- Score 6 or above → STEP 8. CRITICAL: this number is a sunlight rating (1–10 scale), NOT a credit score. Never use a sunlight score to evaluate credit. You MUST ask the Step 8 credit score question next.
- Score 5 or below or mentions heavy shading → thank them, wish them a good day, END

STEP 8 — CREDIT SCORE:
CRITICAL: You MUST ask this question every time. Never skip it or infer the answer from any previous response.
Ask: "One thing the program does require is a qualifying credit score, typically around 680 or above. Do you think you'd meet that requirement?"
- Customer says yes, or mentions a score above 680 → STEP 9
- Customer says no, or mentions a score of 680 or below → "Unfortunately the financing programs usually require around a 680 score or higher, so you may not qualify at the moment. I appreciate your time and have a great day." END

STEP 9 — CONSULTATION OFFER:
Say: "Perfect, based on what you shared, it sounds like you may be a good candidate, the next step would simply be a quick consultation with one of our solar experts who can give you an actual savings estimate for your home — no cost, no obligation."
- They express interest → STEP 10
- No → thank them, wish them a good day, END

STEP 10 — SCHEDULE DATE:
Ask: "Would mornings or afternoons work better for you?" then ask for a specific date.
- They give a date → store exactly what they said → STEP 11

STEP 11 — SCHEDULE TIME:
Ask what time works for them.
- Clear time with AM or PM between 8:00 AM and 7:00 PM → STEP 12
- Clear time with AM or PM but outside 8:00 AM – 7:00 PM → Say: "I'm sorry, our solar experts are only available between 8 AM and 7 PM. Could you choose a time within that window?" — do NOT advance until a valid time is given.
- Ambiguous time with no clear AM or PM (e.g. "10", "ten", "10 bm", "10 p", "10 pe") → Ask: "Just to confirm, is that in the morning or afternoon?" — do NOT advance until AM or PM is confirmed, then validate against the 8 AM–7 PM window.
- Note: "bm", "b.m", "p", "pe" are speech recognition errors for "PM" — treat as ambiguous.

STEP 12 — CONFIRMATION & END:
Say: "So we are all set — one of our solar experts will visit you on [date] at [time], before visiting, our expert will reach out to you by phone just to make sure everything is confirmed, we look forward to helping you explore your solar options. Thank you and have a great day"

END CALL.
"""

SENTENCE_ENDINGS = {'.', '?', '!'}

# Sentinel emitted after the bot's closing sentence so the pipeline
# can trigger end_session automatically.
END_CALL_SENTINEL = "__END_CALL__"

# Phrases that indicate the bot has finished the call (Step 12).
# Matched case-insensitively against the accumulated full response.
END_CALL_PHRASES = [
    "thank you for your time",
    "i hope your day gets better",
    "i hope you have a great day",
    "have a great day",
    "have a wonderful day",
    "have a good day",
    "take care",
    "appreciate your time",
    "wish you all the best",
    "enjoy the rest of your day",
]

# Module-level client — boto3 clients are thread-safe and reusable,
# so share one across all sessions instead of one per WebSocket connection.
_bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

# ── Tools ─────────────────────────────────────────────
# The supported electric-provider list lives in providers.py, NOT in the
# system prompt. The model calls this tool during STEP 6 to validate whatever
# provider the customer names; the actual matching happens in match_provider().
PROVIDER_TOOL = {
    "name": "check_electric_provider",
    "description": (
        "Validate the electric utility provider the customer named against the "
        "list of providers the solar program supports. Call this in STEP 6 as "
        "soon as the customer names their electric provider, before replying. "
        "Pass the provider name exactly as the customer said it. Returns "
        "{\"valid\": true, \"canonical_name\": \"...\"} if supported, or "
        "{\"valid\": false} if the provider is not in the supported list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "provider_name": {
                "type": "string",
                "description": "The electric provider name as stated by the customer.",
            }
        },
        "required": ["provider_name"],
    },
}

TOOLS = [PROVIDER_TOOL]


class LLMStream:
    def __init__(self, session: SessionManager, on_sentence_ready):
        self.session = session
        self.on_sentence_ready = on_sentence_ready
        self._client = _bedrock_client

    async def stream_response(self, user_text: str):
        """
        Stream a response from Bedrock.
        Calls on_sentence_ready once per complete sentence without waiting
        for the full response. Tool-use round-trips (e.g. electric-provider
        validation) are resolved inside the turn and are NOT persisted to
        session history — only the spoken text is.
        """
        self.session.add_user_message(user_text)
        log.info(f"[LLM] Full history being sent: {self.session.get_history()}")
        today = date.today().strftime("%B %d, %Y")
        system = SYSTEM_PROMPT.format(today=today)
        messages = self.session.get_history()
        loop = asyncio.get_event_loop()
        await asyncio.to_thread(self._stream_sync, system, messages, loop)

    def _run_tool(self, name: str, tool_input: dict) -> str:
        """Execute a tool call. Returns a JSON string for the tool_result."""
        if name == "check_electric_provider":
            spoken = (tool_input or {}).get("provider_name", "")
            canonical = match_provider(spoken)
            if canonical:
                log.info(f"[LLM] Provider {spoken!r} matched -> {canonical!r}")
                return json.dumps({"valid": True, "canonical_name": canonical})
            log.info(f"[LLM] Provider {spoken!r} not in supported list")
            return json.dumps({"valid": False})
        log.warning(f"[LLM] Unknown tool requested: {name}")
        return json.dumps({"error": f"unknown tool {name}"})

    def _stream_sync(self, system: str, messages: list, loop):
        """
        Blocking Bedrock stream handler.
        Runs in a thread via asyncio.to_thread.
        Fires on_sentence_ready per complete sentence and resolves any tool
        calls in a loop before finishing the turn.

        `messages` is copied and mutated locally for tool round-trips only;
        it is never written back to session history.
        """
        messages = list(messages)
        sentence_buf = ""
        full_response = ""

        while True:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": messages,
                "tools": TOOLS,
            })

            try:
                response = self._client.invoke_model_with_response_stream(
                    modelId=MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )
            except Exception as e:
                log.error(f"[LLM] Bedrock error: {e}")
                self._emit("I'm sorry, I ran into an error. Please try again.", loop)
                return

            blocks = {}          # index -> accumulated content block
            stop_reason = None

            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                etype = chunk.get("type")

                if etype == "content_block_start":
                    idx = chunk["index"]
                    cb = chunk.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        blocks[idx] = {
                            "type": "tool_use",
                            "id": cb.get("id"),
                            "name": cb.get("name"),
                            "json": "",
                        }
                    else:
                        blocks[idx] = {"type": "text", "text": ""}

                elif etype == "content_block_delta":
                    idx = chunk["index"]
                    delta = chunk.get("delta", {})
                    dtype = delta.get("type")

                    if dtype == "text_delta":
                        token = delta.get("text", "")
                        if not token:
                            continue
                        blocks.setdefault(idx, {"type": "text", "text": ""})
                        blocks[idx]["text"] += token
                        sentence_buf += token
                        full_response += token
                        if sentence_buf.rstrip() and sentence_buf.rstrip()[-1] in SENTENCE_ENDINGS:
                            sentence = sentence_buf.strip()
                            if sentence:
                                log.debug(f"[LLM] Sentence: {sentence}")
                                self._emit(sentence, loop)
                            sentence_buf = ""

                    elif dtype == "input_json_delta":
                        blk = blocks.setdefault(
                            idx, {"type": "tool_use", "id": None, "name": None, "json": ""}
                        )
                        blk["json"] += delta.get("partial_json", "")

                elif etype == "message_delta":
                    stop_reason = chunk.get("delta", {}).get("stop_reason", stop_reason)

            # ── Tool round-trip ──────────────────────────
            if stop_reason == "tool_use":
                # Speak anything the model said before the tool call (normally none).
                if sentence_buf.strip():
                    self._emit(sentence_buf.strip(), loop)
                    sentence_buf = ""

                assistant_content = []
                tool_results = []
                for idx in sorted(blocks):
                    b = blocks[idx]
                    if b["type"] == "text" and b["text"]:
                        assistant_content.append({"type": "text", "text": b["text"]})
                    elif b["type"] == "tool_use":
                        try:
                            tool_input = json.loads(b["json"]) if b["json"] else {}
                        except json.JSONDecodeError:
                            tool_input = {}
                        assistant_content.append({
                            "type": "tool_use",
                            "id": b["id"],
                            "name": b["name"],
                            "input": tool_input,
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": b["id"],
                            "content": self._run_tool(b["name"], tool_input),
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})
                continue  # re-invoke so the model continues with the tool result

            # ── Normal completion ──────────────────────
            break

        # Flush any trailing text (incomplete sentence)
        if sentence_buf.strip():
            self._emit(sentence_buf.strip(), loop)

        # After all sentences are emitted, check if this was the closing turn.
        # If so, send the sentinel so the pipeline can end the session.
        full_response_lower = full_response.lower()
        if any(phrase in full_response_lower for phrase in END_CALL_PHRASES):
            log.info("[LLM] End-of-call phrase detected — emitting END_CALL sentinel")
            self._emit(END_CALL_SENTINEL, loop)

    def _emit(self, sentence: str, loop):
        """
        Hand a sentence to on_sentence_ready on the event loop thread.
        Logged and swallowed on timeout/error so a slow downstream callback
        (TTS/websocket) can't crash the stream thread mid-call and leave
        the pipeline's turn lock held.
        """
        try:
            asyncio.run_coroutine_threadsafe(
                self.on_sentence_ready(sentence),
                loop
            ).result(timeout=5.0)
        except Exception as e:
            log.error(f"[LLM] on_sentence_ready failed for {sentence!r}: {e}")