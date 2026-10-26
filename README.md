# WhatsApp notification router

My entry for HackerRank Orchestrate, August 2026 (24 hours; finished #27 of 1,983).

For each incoming message (text, image poster or voice note), it decides whether to notify the user now, save the message for a digest, or mute it. It also gives the message type, a short reason, a confidence score, and the past messages it used as evidence. A Groq-hosted multimodal model reads the content. User, group and sender history are joined in plain Python.

The solution and how to run it are in [`code/README.md`](code/README.md).
