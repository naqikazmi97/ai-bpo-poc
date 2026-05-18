"""
slots.py — End-of-session entity extraction
Runs a single Bedrock call on the full conversation history.
No per-turn calls. No partial state. One clean extraction at the end.
"""
import asyncio
import json
import logging
import boto3

log = logging.getLogger(__name__)

REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ── Slot schema ───────────────────────────────────────────────────
# Edit this to match your business requirements.
# required = must be collected, flagged as missing if absent
# optional = extract if mentioned, no alarm if absent

SLOT_SCHEMA = {
    "required": {
        "step": "Current step reached: greeting|callback|q1|q2|q3|consultation|appointment|disqualified|complete",
        "customerMood": "Customer mood: positive|negative|null",
        "isHomeowner": "Whether customer is homeowner: true/false/null",
        "highElectricBill": "Whether electric bill is $100+: true/false/null",
        "goodCreditScore": "Whether credit score is above 680: true/false/null",
        "wantsConsultation": "Whether customer wants consultation: true/false/null",
    },
    "optional": {
        "callbackTime": "Requested callback time if provided",
        "appointmentDate": "Appointment date if scheduled",
        "appointmentTime": "Appointment time if scheduled",
        "disqualifyReason": "Reason customer was disqualified if applicable",
    }
}

EXTRACTION_PROMPT = """You are an entity extraction system for a customer support call.

Extract the following fields from the conversation transcript below.
Return ONLY a valid JSON object. No explanation. No markdown. No extra text.

Fields to extract:
{schema}

Extraction rules:
- Only include a field if it was explicitly stated in the conversation.
- If a field was not mentioned at all, omit it from the JSON entirely.
- If a customer said they don't have something (e.g. "I don't have an account number"), set its value to null.
- Normalize phone numbers to digits only, no spaces or dashes (e.g. "07700900123").
- For issue_summary: write a clean one-sentence summary in third person (e.g. "Customer's internet has been down since Tuesday.").
- For resolution_status: use only one of: "resolved", "unresolved", "escalated".

Conversation transcript:
{conversation}
"""


class SlotExtractor:
    def __init__(self):
        self._client = boto3.client("bedrock-runtime", region_name=REGION)

    async def extract(self, history: list[dict]) -> dict:
        """
        Extract all entities from the full conversation history.
        Called once at session end.
        Returns dict of extracted slot values.
        """
        conversation_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        )

        schema_lines = []
        for section_name, fields in SLOT_SCHEMA.items():
            for key, description in fields.items():
                tag = "[required]" if section_name == "required" else "[optional]"
                schema_lines.append(f"- {key} {tag}: {description}")

        prompt = EXTRACTION_PROMPT.format(
            schema="\n".join(schema_lines),
            conversation=conversation_text
        )

        result = await asyncio.to_thread(self._extract_sync, prompt)

        # Log missing required slots
        missing = [
            k for k in SLOT_SCHEMA["required"]
            if k not in result or result[k] is None
        ]
        if missing:
            log.warning(f"[SlotExtractor] Missing required slots: {missing}")
        else:
            log.info("[SlotExtractor] All required slots collected")

        return result

    def _extract_sync(self, prompt: str) -> dict:
        """Blocking Bedrock call — runs in threadpool via asyncio.to_thread."""
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}]
        })

        response = self._client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        raw = json.loads(response["body"].read())
        text = raw["content"][0]["text"].strip()

        # Strip accidental markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            log.error(f"[SlotExtractor] JSON parse failed: {e} | Raw: {text}")
            return {}
