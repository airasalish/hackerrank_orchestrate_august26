# Operational Analysis — Message Notification Router

*Note: this file isn't explicitly required by this round's problem statement (unlike the
prior Orchestrate round's), but is included for the same reasons cost/latency awareness
matters generally — and to have a defensible answer ready if asked in the interview.*

## Model & cost

**Model:** `qwen/qwen3.6-27b` via Groq API — free tier, $0 cost, no credit card.
Voice notes and image posters are handled by the same model (vision + native
multimodal input), decoupled from the routing decision itself into separate lightweight
calls.

## Call volume

| Step | When it runs | Approx. calls for 110 messages |
|---|---|---|
| Image extraction | Once per image message (~20 of 110) | ~20 |
| Voice transcription | Once per voice message (~13 of 110) | ~13 |
| Routing decision | Once per message (110), plus retries on failure | 110–160+ |

## Rate limits — the actual operational story

Groq's free/on-demand tier caps `qwen/qwen3.6-27b` at **200,000 tokens/day per account**,
evaluated on a rolling window (not a fixed midnight reset). This repo's development alone
(smoke tests, a 30-row evaluation pass, prompt iteration) consumed most of one account's
daily budget before the real 110-row run even started — meaning a single free account
cannot complete this dataset in one sitting.

Mitigations implemented in `code/main.py`:
- **Multi-key rotation** (`KeyPool`): every `GROQ_API_KEY`, `GROQ_API_KEY_2`, ... found in
  the environment is round-robined across calls, including across retries, so a
  rate-limited key doesn't block the very next attempt from trying a different account.
- **Dead-key cooldown**: once a key returns a "tokens per day" error, it's marked dead and
  skipped in rotation until its cooldown passes, instead of wasting a retry+wait cycle on
  a key already known to be capped.
- **Crash-safe fallback, distinguishable from real results**: a message that exhausts all
  retries gets a safe `digest`/`unknown` placeholder rather than crashing the run — and
  that placeholder is tagged so a later resume specifically retries it instead of
  mistaking it for a completed row.
- **Small, cheap requests**: images resized to 300px, evidence shortlist capped at 5
  historical messages, `reasoning_format: "hidden"` so the model's internal reasoning
  doesn't inflate the visible completion or the token bill.

## Accuracy (sample_messages.csv, 30 labeled rows)

| Field | Correct | Accuracy |
|---|---|---|
| `action` | 27/30 | 90% |
| `message_type` | 25/30 | 83% |
| `evidence_message_ids` (overlap with gold) | 19/30 | 63% |
| **Overall** | 71/90 | **79%** |

**Root cause of the evidence-overlap gap, found by inspecting a specific mismatch, not
guessed:** for `sample_msg_001`, gold cites `message_0001` as evidence. That message
genuinely shares the same `user_id`, `sender_user_id`, and `group_id` as the current
message — it's real, relevant evidence. But it's from March, and this user/group pair has
several more recent related messages. `find_evidence_candidates()` in `code/main.py` sorts
by recency and keeps only the top 5, so `message_0001` was correctly identified as related
but ranked out of the shortlist by more recent alternatives from the same conversation.

This means the evidence-selection heuristic optimizes for **recency**, while some gold
answers were chosen for **topical similarity** regardless of age. The fix — given more
runway — would be re-ranking candidates by text similarity to the current message (e.g. a
lightweight keyword/embedding overlap score) instead of pure recency, or including more
than 5 candidates when the shortlist is topically thin. Not applied this round: it would
mean re-running the full 110-row set again, spending more of an already-tight free-tier
token budget on a metric that's secondary to `action`/`message_type` correctness, which
are already strong.

## Latency

Individual calls typically return in a few seconds; wall-clock time for the full run is
dominated by rate-limit waits, not model latency, given the free tier's daily cap.

## What was not optimized

- **Paid tier billing** would remove the daily cap entirely — not pursued to keep this a
  $0 submission, per the free-tier constraint the problem statement is written around.
- **Parallel/async calls** would reduce wall-clock time further but risks bursting
  per-minute limits on top of the daily cap already in play.
- **Caching** repeated evidence lookups across messages from the same sender/group/business
  would cut a small amount of redundant computation (all in Python, not API calls, so no
  cost impact — just a minor efficiency item).
