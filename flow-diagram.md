# Voicebot App — Detailed Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                         APP STARTUP                                  ║
╚══════════════════════════════════════════════════════════════════════╝

  Page Load
      │
      ▼
  Show Login Overlay (Cognito)
      │
      ├─ doLogin() → Cognito authenticateUser
      │       │
      │       ├── onSuccess → hide login overlay
      │       ├── onFailure → show error
      │       └── newPasswordRequired → show new-password section
      │                   └── completeNewPasswordChallenge → hide overlay
      │
      └─ connect() called at page load (WS connects in background)
              │
              ▼
         ws.onopen → setStatus(connected), showToast
              │
              └── if pendingBotStart → send bot_start ◄─────────────────┐
                                                                          │
╔══════════════════════════════════════════════════════════════════════╗  │
║                        CALL START FLOW                               ║  │
╚══════════════════════════════════════════════════════════════════════╝  │
                                                                          │
  start-overlay shown (page load or after next-call-btn)                 │
      │                                                                   │
      ▼                                                                   │
  User clicks "Start Call"                                               │
      │                                                                   │
      ├── WS OPEN? ──YES──► send { type: bot_start }                     │
      │                                                                   │
      └── WS NOT OPEN? ──► pendingBotStart = true ─────────────────────►┘
                           (fires when ws.onopen triggers)


╔══════════════════════════════════════════════════════════════════════╗
║                     BACKEND: bot_start RECEIVED                      ║
╚══════════════════════════════════════════════════════════════════════╝

  pipeline.bot_start()
      │
      ├── session.clear()
      ├── tts.start()
      └── llm.stream_response("START_CALL")
              │
              ▼
         [STREAMING PIPELINE — see below]


╔══════════════════════════════════════════════════════════════════════╗
║               MAIN VOICE TURN PIPELINE (per turn)                    ║
╚══════════════════════════════════════════════════════════════════════╝

  FRONTEND                    BACKEND                        AWS SERVICES
  ─────────                   ───────                        ────────────
  User speaks
      │
      ▼
  ScriptProcessor captures
  PCM audio (16kHz, Int16)
      │
      ▼
  ws.send(binary frames) ───► feed_audio(chunk)
  [continuous stream]              │
                                   ▼
                              asr.feed(chunk) ──────────────► Amazon Transcribe
                                                               (streaming, PCM)
                                                                    │
                                                              partial results
                                                              (buffered, ignored)
                                                                    │
                                                              FINAL result
                                                                    │
  User clicks mic / auto      on_utterance_ready(text) ◄───────────┘
  stop (end_of_audio sent) ─►
      │                            │
      ▼                            ▼
  ws.send({end_of_audio})    _on_user_spoke(text)
                                   │
                                   ├── send { asr_result, text } ──────► Frontend
                                   │        shows user bubble           shows user msg
                                   │
                                   ├── session.add_user_message(text)
                                   │
                                   └── llm.stream_response(text)
                                              │
                                              ▼
                                        Bedrock API ──────────────────► Amazon Nova Micro
                                        (streaming)                     (system prompt:
                                              │                          Tiffany script)
                                              │
                                        token stream
                                              │
                                        accumulate into
                                        sentence buffer
                                              │
                                        sentence complete? (.  ?)
                                              │
                                              ├── send { llm_sentence } ─► Frontend
                                              │                            appends to bot bubble
                                              │
                                              └── tts.synthesize(sentence)
                                                        │
                                                        ▼
                                                  Queue sentence ───────► Amazon Polly Neural
                                                                          (Danielle, PCM 16kHz)
                                                        │
                                                  audio bytes ready
                                                        │
                                                  _send_audio(bytes) ────► ws.send(binary)
                                                                                │
                                                                           Frontend receives
                                                                           binary frame
                                                                                │
                                                                           audioQueue.push()
                                                                                │
                                                                           drainQueue()
                                                                           plays PCM via
                                                                           AudioContext
                                              │
                                        END_CALL_SENTINEL?
                                              │
                                              └── _bot_ending_call = true
                                                   (checked after TTS drains)

                                   After all sentences done:
                                   _on_turn_complete()
                                        │
                                        ├── session.add_assistant_message()
                                        ├── await tts._queue.join() (wait TTS finishes)
                                        ├── sleep 0.3s
                                        ├── send { turn_end } ─────────────► Frontend
                                        │                                     removes typing indicator
                                        │                                     if audio queue empty:
                                        │                                       startRecording()
                                        │                                     else drainQueue handles it
                                        │
                                        └── if _bot_ending_call:
                                                end_session(bot_initiated=True)


╔══════════════════════════════════════════════════════════════════════╗
║                    FRONTEND AUDIO PLAYBACK LOOP                      ║
╚══════════════════════════════════════════════════════════════════════╝

  binary WS frame arrives
        │
        ▼
  audioQueue.push(buffer)
        │
        ▼
  drainQueue() [recursive via src.onended]
        │
        ├── decode Int16 → Float32
        ├── AudioContext.createBufferSource()
        ├── src.start()
        └── src.onended = drainQueue
                │
                └── queue empty?
                        │
                        ├── YES + botTurnComplete = true
                        │       └── setTimeout(startRecording, 300ms)
                        │           [mic auto-opens after bot finishes speaking]
                        │
                        └── NO → play next chunk


╔══════════════════════════════════════════════════════════════════════╗
║                    END CALL FLOWS                                     ║
╚══════════════════════════════════════════════════════════════════════╝

  ── USER-INITIATED ──────────────────────────────────────────────────

  User clicks "End Call"
        │
        ├── stopRecording() → send end_of_audio
        ├── flush audioQueue
        └── ws.send({ end_session })
                │
                ▼
         pipeline.end_session()
                │
                ├── _extract_and_save()
                │       │
                │       └── SlotExtractor.extract(history)
                │               │
                │               └── Bedrock invoke ──────────────────► Claude Haiku
                │                   (single call,                      (structured
                │                    not streaming)                     JSON output)
                │                       │
                │               parse JSON slots
                │                       │
                │               session.save_slots(slots)
                │                       │
                │               DynamoDB put_item ──────────────────► DynamoDB
                │               (record_type =                         ai-bpo-poc
                │                APPOINTMENT |
                │                DISQUALIFIED |
                │                COMPLETE)
                │
                └── send { session_ended, slots } ──────────────────► Frontend
                                                                        renderSlots()
                                                                        close WS
                                                                        show next-call-overlay

  ── BOT-INITIATED ───────────────────────────────────────────────────

  LLM response contains "have a great day" etc.
        │
        ▼
  END_CALL_SENTINEL emitted after last sentence
        │
        ▼
  _bot_ending_call = true
        │
        ▼
  _on_turn_complete() detects flag
        │
        ▼
  end_session(bot_initiated=True)
        │
        └── [same extraction + session_ended flow as above]

  ── ABRUPT DISCONNECT ───────────────────────────────────────────────

  Client drops connection
        │
        ▼
  WebSocketDisconnect caught in main.py
        │
        ▼
  pipeline.cleanup(already_ended=False)
        │
        ├── asr.stop(), tts.stop()
        └── if history >= 2 messages:
                └── _extract_and_save() [fallback extraction]


╔══════════════════════════════════════════════════════════════════════╗
║                    START NEXT CALL FLOW                              ║
╚══════════════════════════════════════════════════════════════════════╝

  User clicks "Start Next Call"
        │
        ├── hide next-call-overlay
        ├── new sessionId (random UUID slice)
        ├── clear messages, reset metrics
        ├── reset all state flags
        │   (audioQueue, playbackActive, botTurnComplete,
        │    pendingBotStart, recording)
        ├── close old AudioContext
        ├── re-enable micBtn + endBtn
        ├── show start-overlay
        └── connect()  [new WS with new sessionId]
                │
                ▼
        [back to CALL START FLOW ↑]


╔══════════════════════════════════════════════════════════════════════╗
║                    SESSION PERSISTENCE (DynamoDB)                    ║
╚══════════════════════════════════════════════════════════════════════╝

  Every turn:
    add_user_message / add_assistant_message
      └── _trim_history() keeps last 10 turns (20 messages)

  Session init:
    _load() → get_item(session_id, "SESSION")
      └── restores history + slots if session existed

  Slot save record types:
    appointmentDate present  → record_type = "APPOINTMENT"
    disqualifyReason present → record_type = "DISQUALIFIED"
    otherwise                → record_type = "COMPLETE"
```
