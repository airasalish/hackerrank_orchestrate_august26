<h1 align="center">📲 WhatsApp Notification Router</h1>

<p align="center"><b>Ping now, save for later, or mute? Personalised triage for a noisy WhatsApp feed.</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/HackerRank_Orchestrate-August_2026-2ec866?style=for-the-badge&logo=hackerrank&logoColor=white" alt="HackerRank Orchestrate August 2026">
  <img src="https://img.shields.io/badge/Rank-%2327_of_1%2C983_%C2%B7_Top_1.4%25-f5b301?style=for-the-badge" alt="Rank #27 of 1,983, top 1.4%">
  <img src="https://img.shields.io/badge/Groq-multimodal-f55036?style=for-the-badge" alt="Groq">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

My **best finish**: **#27 of 1,983** in HackerRank's 24-hour AI agent hackathon.

Family chats, society notices, school updates, promotions, poster images, voice notes and scams all land in one stream. For each message the router chooses **`notify` / `digest` / `mute`**, names the message type, explains why, gives a confidence score, and **cites the historical messages it used as evidence**.

- 🧠 A Groq vision + text model in JSON mode reads text, image posters and voice notes
- 🔗 User, group, business and history context is joined deterministically in plain Python
- 🔁 Multi-key rotation and caching keep it inside free-tier quotas
- 🧪 Tests in [`code/tests/`](code/tests) plus an evaluation report

**▶ Solution docs and how to run it: [`code/README.md`](code/README.md)**

---

<details>
<summary><b>Original challenge README</b></summary>

# HackerRank Orchestrate

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon.

## Message Notification Router

Build an AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

The system must reason over multimodal messages, including text messages, image posters/screenshots, and voice notes.

WhatsApp is noisy. A user can receive family chats, society notices, school updates, co-worker messages, business account promotions, image posters, voice notes, and scams in the same message stream. Treating every message the same creates two bad outcomes: important messages get missed, and unwanted or risky messages interrupt the user.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, allowed values, and submission format.

**The submitted solution lives in [`code/`](./code), documented in [`code/README.md`](./code/README.md)** — setup, env vars, how to run it, how the pipeline works, and the provider decisions made along the way.

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
└── dataset/
    ├── messages.csv                  # Messages to route
    ├── output.csv                    # Blank submission template
    ├── sample_messages.csv           # Solved examples
    ├── users.csv                     # User notification behavior
    ├── groups.csv                    # Group metadata
    ├── group_members.csv             # User-group relationships
    ├── business_accounts.csv         # Business sender metadata
    ├── user_business_history.csv     # User-business history
    ├── message_history.csv           # Historical messages
    ├── message_events.csv            # User reactions to historical messages
    ├── images.csv                    # Image IDs and media file paths
    ├── voice_notes.csv               # Voice note IDs and media file paths
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## What You Need to Build

For every row in `dataset/messages.csv`, produce one row in `output.csv` with:

| Column | Meaning |
|---|---|
| `message_id` | Incoming message ID |
| `action` | One of `notify`, `digest`, or `mute` |
| `message_type` | Best-fit message category |
| `reason` | Short human-readable explanation |
| `confidence` | Number from `0` to `1` |
| `evidence_message_ids` | Historical message IDs used as evidence; write `none` if there is no useful evidence |

Your system should make personalized decisions using the provided message, user, group, business, media, and historical interaction data.
For image and voice-note messages, `images.csv` and `voice_notes.csv` only provide file paths; your system should inspect the media files themselves.

---

## Suggested Workflow

1. Inspect `dataset/sample_messages.csv` to understand the expected output format.
2. Load `dataset/messages.csv` and all relevant context files.
3. Build your routing system using any approach: LLMs, retrieval, rules, classifiers, agents, or hybrids.
4. Write predictions to `output.csv`.
5. Evaluate your approach on the solved sample rows before submitting.

You may use any language or runtime. Python, JavaScript, and TypeScript are all reasonable choices.

---

## Requirements

Your solution must:

- be runnable from the terminal
- read the provided files from `dataset/`
- produce a valid `output.csv`
- include one prediction for every `message_id` in `dataset/messages.csv`
- not use organizer-only files or hardcoded labels

If you use API keys or secrets, read them from environment variables. Never hardcode secrets in the repo.

---

## Evaluation

Your `output.csv` will be compared against hidden ground-truth labels.

The scoring will consider:

- correctness of `action`
- correctness of `message_type`
- usefulness and consistency of `reason`
- whether `evidence_message_ids` point to relevant historical messages
- reasonable confidence calibration

Strong systems will combine retrieval, structured metadata, behavioral history, safety checks, OCR/ASR handling, and contextual reasoning.

---

## Chat Transcript Logging

This repo includes an [`AGENTS.md`](./AGENTS.md) file for AI coding tools. It asks compatible tools to append conversation summaries to:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate_august26/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` |

Upload this log as your chat transcript at submission time. Do not paste secrets into the chat.

---

## Submission

Submit the following files as instructed by HackerRank:

1. **Code zip**: full runnable solution, prompts/configs, README, and any evaluation files.
2. **Predictions CSV**: final `output.csv` for all rows in `dataset/messages.csv`.
3. **Chat transcript**: the `log.txt` described above.

Before submitting, confirm:

- `output.csv` has one row per row in `dataset/messages.csv`.
- `output.csv` has the exact required columns in the exact required order.
- Your runnable code and setup instructions are included in `code.zip`.

</details>

---

<sub>Built by **Aira K. Salish** · [GitHub](https://github.com/airasalish) · [Portfolio](https://aira-salish.vercel.app) · [LinkedIn](https://linkedin.com/in/airasalish)</sub>
