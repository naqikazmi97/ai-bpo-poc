"""
session.py — Conversation session management via DynamoDB
Persists conversation history and extracted slots.
"""
import logging
import boto3
from datetime import datetime, timezone

log = logging.getLogger(__name__)

REGION = "us-east-1"
TABLE_NAME = "ai-bpo-poc"
MAX_HISTORY_TURNS = 10   # Keep last 10 turns to control token count


class SessionManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._history = []
        self._slots = {}
        self._dynamodb = boto3.resource("dynamodb", region_name=REGION)
        self._table = self._dynamodb.Table(TABLE_NAME)
        self._load()

    # ── Load / Save ───────────────────────────────────────────────

    def _load(self):
        """Load existing session on reconnect."""
        try:
            resp = self._table.get_item(Key={"session_id": self.session_id})
            item = resp.get("Item")
            if item:
                self._history = item.get("history", [])
                self._slots   = item.get("slots", {})
                log.info(
                    f"[Session {self.session_id}] Loaded "
                    f"{len(self._history)} messages, "
                    f"{len(self._slots)} slots from DynamoDB"
                )
        except Exception as e:
            log.warning(f"[Session {self.session_id}] DynamoDB load failed: {e}")

    def _save_history(self):
        """Persist conversation history after each assistant turn."""
        try:
            self._table.put_item(Item={
                "session_id":  self.session_id,
                "history":     self._history,
                "slots":       self._slots,
                "updated_at":  datetime.now(timezone.utc).isoformat(),
                # TTL: auto-delete after 24 hours
                "ttl": int(datetime.now(timezone.utc).timestamp()) + 86400
            })
        except Exception as e:
            log.warning(f"[Session {self.session_id}] DynamoDB save failed: {e}")

    def save_slots(self, slots: dict):
        self._slots = slots
        try:
            self._table.put_item(Item={
                "sessionId": self.session_id,
                "recordType": "APPOINTMENT" if slots.get("appointmentDate") else "DISQUALIFIED" if slots.get("disqualifyReason") else "COMPLETE",
                "createdAt": datetime.now(timezone.utc).isoformat(),
               **slots
            })
            log.info(f"[Session {self.session_id}] Slots saved: {slots}")
        except Exception as e:
            log.error(f"[Session {self.session_id}] Failed to save slots: {e}")

    # ── History management ────────────────────────────────────────

    def add_user_message(self, text: str):
        self._history.append({"role": "user", "content": text})
        self._trim_history()

    def add_assistant_message(self, text: str):
        self._history.append({"role": "assistant", "content": text})
        self._trim_history()
        self._save_history()  # Persist after each complete turn

    def get_history(self) -> list:
        return self._history.copy()

    def get_slots(self) -> dict:
        return self._slots.copy()

    def clear(self):
        """Reset conversation — called when user clears the chat."""
        self._history = []
        self._slots   = {}
        try:
            self._table.delete_item(Key={"session_id": self.session_id})
        except Exception as e:
            log.warning(f"[Session {self.session_id}] DynamoDB clear failed: {e}")

    def _trim_history(self):
        """Keep last N turns to prevent context bloat."""
        max_messages = MAX_HISTORY_TURNS * 2  # user + assistant pairs
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
