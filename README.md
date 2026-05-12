# pi-bridge

Telegram ↔ Pi coding agent bridge. Lets you chat with [pi](https://github.com/earendil-works/pi-coding-agent) through Telegram — including model switching, voice messages, and multi-user support.

## How it works

```
Telegram user → bot API → bridge.py → pi (RPC mode) → LLM → response back
```

The bridge runs `pi --mode rpc` as a subprocess. User messages are forwarded as prompts. Pi's streaming text deltas are collected and sent back as a single Telegram message.

## Features

- **Telegram chat** — text MarkdownV2 formatting
- **Voice messages** — transcribed with faster-whisper, sent to pi
- **Model switching** — `/model` command in Telegram lists and switches providers/models
- **Thinking toggle** — `/thinking on|off|auto`
- **New session** — `/new` resets conversation
- **Abort** — stop a running generation
- **Multi-user** — `ALLOWED_USERS` env var restricts access

## Setup

### 1. Create a Telegram bot

Message [@BotFather](https://t.me/BotFather), create a bot, get the token.

### 2. Install dependencies

```bash
pip install python-telegram-bot faster-whisper
```

### 3. Configure

```bash
mkdir -p /root/.pi/agent

# Telegram bot token
echo "YOUR_BOT_TOKEN" > /root/.pi/agent/telegram-token

# Pi settings (optional, bridge creates it automatically)
echo '{"provider": "local", "model": "gemma-4-26b"}' > /root/.pi/agent/settings.json
```

### 4. Run

```bash
# Single user (your Telegram ID)
ALLOWED_USERS="123456789" python3 bridge.py

# Multi-user
ALLOWED_USERS="123456789,987654321" python3 bridge.py

# No restriction (anyone can use your bot)
python3 bridge.py
```

## Telegram commands

| Command | Action |
|---------|--------|
| Any text | Sent to pi as prompt |
| `/model` | List and switch between available models |
| `/thinking on\|off\|auto` | Set reasoning mode |
| `/new` | Start a new conversation session |
| `/abort` | Stop current generation |
| Voice message | Transcribed and sent to pi |

## Files

- `bridge.py` — the whole thing (~500 lines)
- `~/.pi/agent/telegram-token` — bot token
- `~/.pi/agent/settings.json` — persisted model/provider selection

## Requirements

- Python 3.10+
- [pi](https://github.com/earendil-works/pi-coding-agent) installed and in PATH
- Telegram bot token
- `python-telegram-bot`, `faster-whisper`
