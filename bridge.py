#!/usr/bin/env python3
"""Telegram ↔ Pi RPC bridge with /model switching."""

import asyncio
import json
import os
import signal
import sys
import textwrap
from pathlib import Path

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from faster_whisper import WhisperModel
WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")

# ── Config ────────────────────────────────────────────────────────────────────
TOKEN_FILE = Path("/root/.pi/agent/telegram-token")
SETTINGS_FILE = Path("/root/.pi/agent/settings.json")
ALLOWED_USERS = set(
    int(uid) for uid in os.environ.get("ALLOWED_USERS", "").split(",") if uid
)

def read_token() -> str:
    return TOKEN_FILE.read_text().strip()

def read_settings() -> dict:
    return json.loads(SETTINGS_FILE.read_text())

def write_settings(data: dict):
    SETTINGS_FILE.write_text(json.dumps(data, indent=2) + "\n")

# ── Pi subprocess manager ─────────────────────────────────────────────────────

class PiProcess:
    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future] = {}  # request_id → Future
        self._prompt_futures: list[asyncio.Future] = []
        self._current_prompt_id: str | None = None
        self._response_parts: list[str] = []
        self._lock = asyncio.Lock()

    async def start(self):
        if self.proc and self.proc.returncode is None:
            return
        self.proc = await asyncio.create_subprocess_exec(
            "pi", "--mode", "rpc", "--session-dir", "/root/.pi/agent/sessions",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        asyncio.create_task(self._read_stdout())
        asyncio.create_task(self._read_stderr())

    async def stop(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except TimeoutError:
                self.proc.kill()

    async def _read_stdout(self):
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            await self._handle_event(event)

    async def _read_stderr(self):
        assert self.proc and self.proc.stderr
        while True:
            line = await self.proc.stderr.readline()
            if not line:
                break
            print(f"[pi stderr] {line.rstrip()}", file=sys.stderr, flush=True)

    async def _handle_event(self, event: dict):
        etype = event.get("type")

        # Track text deltas for current prompt
        if etype == "message_update":
            delta = event.get("assistantMessageEvent", {})
            if delta.get("type") == "text_delta":
                self._response_parts.append(delta["delta"])

        elif etype == "agent_end":
            # Resolve all waiting prompt futures
            text = "".join(self._response_parts)
            self._response_parts = []
            while self._prompt_futures:
                fut = self._prompt_futures.pop(0)
                if not fut.done():
                    fut.set_result(text)
                text = ""  # only first gets the text, rest get empty

        # Handle response to commands (get_available_models, set_model, etc.)
        elif etype == "response":
            req_id = event.get("id")
            if req_id and req_id in self._pending:
                fut = self._pending.pop(req_id)
                if not fut.done():
                    fut.set_result(event)

    async def _send(self, cmd: dict) -> dict | None:
        """Send a command and wait for its response."""
        async with self._lock:
            if not self.proc or self.proc.returncode is not None:
                await self.start()
            assert self.proc and self.proc.stdin
            req_id = f"req-{id(cmd)}"
            cmd["id"] = req_id
            fut = asyncio.Future()
            self._pending[req_id] = fut
            self.proc.stdin.write((json.dumps(cmd) + "\n").encode())
            await self.proc.stdin.drain()
        try:
            return await asyncio.wait_for(fut, timeout=10)
        except TimeoutError:
            self._pending.pop(req_id, None)
            return {"success": False, "error": "timeout"}

    async def prompt(self, message: str) -> str:
        """Send a prompt and wait for the full text response."""
        async with self._lock:
            if not self.proc or self.proc.returncode is not None:
                print(f"[bridge] Starting pi subprocess...", flush=True)
                await self.start()
                print(f"[bridge] Pi started (pid={self.proc.pid})", flush=True)
            if not self.proc:
                raise RuntimeError("self.proc is None")
            if self.proc.stdin is None:
                raise RuntimeError(
                    f"self.proc.stdin is None! proc.pid={self.proc.pid}, "
                    f"proc.returncode={self.proc.returncode}, "
                    f"proc.stdout={self.proc.stdout}, "
                    f"proc.stderr={self.proc.stderr}"
                )
            fut = asyncio.Future()
            self._prompt_futures.append(fut)
            cmd = json.dumps({"type": "prompt", "message": message})
            print(f"[bridge] Sending prompt: {cmd}", flush=True)
            self.proc.stdin.write((cmd + "\n").encode())
            await self.proc.stdin.drain()
            print(f"[bridge] Prompt sent, waiting for response...", flush=True)
        try:
            result = await asyncio.wait_for(fut, timeout=300)
            print(f"[bridge] Got response ({len(result)} chars)", flush=True)
            return result
        except TimeoutError:
            print(f"[bridge] Timed out waiting for response", flush=True)
            return "⌛ Timed out waiting for response."

    async def abort(self):
        if self.proc and self.proc.returncode is None:
            async with self._lock:
                self.proc.stdin.write((json.dumps({"type": "abort"}) + "\n").encode())
                await self.proc.stdin.drain()

    async def new_session(self) -> bool:
        resp = await self._send({"type": "new_session"})
        return resp is not None and resp.get("success", False)

    async def get_models(self) -> list[dict]:
        resp = await self._send({"type": "get_available_models"})
        if resp and resp.get("success") and resp.get("data"):
            return resp["data"].get("models", [])
        return []

    async def set_model(self, provider: str, model_id: str) -> bool:
        resp = await self._send({"type": "set_model", "provider": provider, "modelId": model_id})
        return resp is not None and resp.get("success", False)

    async def set_thinking(self, level: str) -> bool:
        resp = await self._send({"type": "set_thinking_level", "level": level})
        return resp is not None and resp.get("success", False)

    async def get_state(self) -> dict | None:
        resp = await self._send({"type": "get_state"})
        if resp and resp.get("success"):
            return resp.get("data")
        return None


pi = PiProcess()

# ── Auth check ────────────────────────────────────────────────────────────────

def authorized(update: Update) -> bool:
    if not ALLOWED_USERS:
        return True
    uid = update.effective_user.id if update.effective_user else 0
    return uid in ALLOWED_USERS

# ── Command: /start ───────────────────────────────────────────────────────────

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Pi Agent** ready\\.\n\n"
        "Send me a prompt and I'll work on it\\.\n\n"
        "/model — list or switch models\n"
        "/status — check agent health\n"
        "/abort — cancel current task",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

# ── Command: /model ───────────────────────────────────────────────────────────

async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return

    args = context.args  # list of words after /model
    settings = read_settings()

    if args:
        # Switch model
        query = " ".join(args).lower()
        models = await pi.get_models()
        match = None
        for m in models:
            if query in m["id"].lower() or query in m["name"].lower():
                match = m
                break
        if match:
            ok = await pi.set_model(match["provider"], match["id"])
            if ok:
                settings["model"] = {"provider": match["provider"], "modelId": match["id"]}
                write_settings(settings)
                await update.message.reply_text(
                    f"✅ Switched to *{match['name']}* \\(`{match['id']}`\\)",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            else:
                await update.message.reply_text("❌ Failed to switch model\\.", parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await update.message.reply_text(
                f"❌ No model matching \"{query}\"\\. Use /model to list\\.", parse_mode=ParseMode.MARKDOWN_V2
            )
    else:
        # List models
        models = await pi.get_models()
        if not models:
            await update.message.reply_text("❌ No models available\\.")
            return

        current = settings.get("model", {})
        current_id = current.get("modelId", "")
        lines = []
        for m in models:
            marker = " →" if m["id"] == current_id else "  "
            lines.append(
                f"{marker} *{m['name']}*\n"
                f"     `{m['id']}` \\(${m.get('cost',{}).get('input',0):.2f}/M in\\)"
            )
        await update.message.reply_text(
            "📋 **Available models:**\n\n" + "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN_V2,
        )

# ── Command: /status ──────────────────────────────────────────────────────────

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return

    state = await pi.get_state()
    settings = read_settings()
    current = settings.get("model", {})

    if state:
        streaming = "🔵 streaming" if state.get("isStreaming") else "⚪ idle"
        model = f"{current.get('provider','?')}/{current.get('modelId','?')}"
        await update.message.reply_text(
            f"💻 *Agent running*\n"
            f"• Model: `{model}`\n"
            f"• Status: {streaming}\n"
            f"• Messages: {state.get('messageCount', 0)}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    else:
        await update.message.reply_text("⚠️ Agent not responding.")

# ── Command: /abort ───────────────────────────────────────────────────────────

async def new_session_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return
    ok = await pi.new_session()
    if ok:
        await update.message.reply_text("🆕 *New session started\\.*", parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await update.message.reply_text("⚠️ Failed to start new session\\.", parse_mode=ParseMode.MARKDOWN_V2)

async def abort_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return
    await pi.abort()
    await update.message.reply_text("✋ Abort sent.")

# ── Command: /thinking ────────────────────────────────────────────────────────

async def thinking_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        return

    args = context.args
    settings = read_settings()
    levels = ["off", "minimal", "low", "medium", "high", "xhigh"]

    if not args:
        current = settings.get("thinking", "medium")
        await update.message.reply_text(
            f"🧠 *Current thinking:* `{current}`\n\n"
            f"Levels: `off` | `minimal` | `low` | `medium` | `high` | `xhigh`\n\n"
            f"Use `/thinking off` to disable\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    level = args[0].lower()
    if level not in levels:
        await update.message.reply_text(
            f"❌ Unknown level \"{level}\"\\. Use: `off` | `minimal` | `low` | `medium` | `high` | `xhigh`",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    ok = await pi.set_thinking(level)
    if ok:
        settings["thinking"] = level
        write_settings(settings)
        await update.message.reply_text(
            f"✅ Thinking set to *{level}*\\.", parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await update.message.reply_text("❌ Failed to set thinking level\\.", parse_mode=ParseMode.MARKDOWN_V2)

# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        await update.message.reply_text("🔒 Access denied\\.")
        return

    text = update.message.text.strip()
    if not text:
        return

    status_msg = await update.message.reply_text("🤔 Thinking\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)

    try:
        response = await pi.prompt(text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        await status_msg.edit_text(f"❌ Error: {e}")
        return

    if not response:
        await status_msg.edit_text("⚠️ No response\\.")
        return
    if response.startswith("⌛"):
        await status_msg.edit_text(response)
        return

    # Delete "Thinking..." message
    await status_msg.delete()

    # Split long messages for Telegram (4096 char limit)
    for chunk in textwrap.wrap(
        response, width=3800, replace_whitespace=False,
        drop_whitespace=False, break_long_words=False
    ):
        if chunk.strip():
            # Escape for MarkdownV2
            esc = chunk.replace("\\", "\\\\").replace("_", "\\_").replace("*", "\\*")\
                       .replace("[", "\\[").replace("]", "\\]").replace("(", "\\(")\
                       .replace(")", "\\)").replace("~", "\\~").replace("`", "\\`")\
                       .replace(">", "\\>").replace("#", "\\#").replace("+", "\\+")\
                       .replace("-", "\\-").replace("=", "\\=").replace("|", "\\|")\
                       .replace("{", "\\{").replace("}", "\\}").replace(".", "\\.")\
                       .replace("!", "\\!")
            await update.message.reply_text(esc, parse_mode=ParseMode.MARKDOWN_V2)

# ── Voice handler ─────────────────────────────────────────────────────────────

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not authorized(update):
        await update.message.reply_text("🔒 Access denied\.")
        return

    voice = update.message.voice
    status_msg = await update.message.reply_text("🎤 Transcribing\.\.\.", parse_mode=ParseMode.MARKDOWN_V2)

    try:
        file = await voice.get_file()
        voice_path = "/tmp/voice_msg.ogg"
        await file.download_to_drive(voice_path)

        segments, info = WHISPER_MODEL.transcribe(voice_path, language=None)
        text = " ".join([seg.text.strip() for seg in segments]).strip()

        os.remove(voice_path)

        if not text:
            await status_msg.edit_text("⚠️ Could not understand audio\\.")
            return

        await status_msg.edit_text(f"🗣️ *Heard:* `{text}`", parse_mode=ParseMode.MARKDOWN_V2)

        # Now send to pi
        thinking_msg = await update.message.reply_text("🤔 Thinking\.\.\.", parse_mode=ParseMode.MARKDOWN_V2)
        response = await pi.prompt(text)

        if response.startswith("⌛"):
            await thinking_msg.edit_text(response)
            return
        if not response:
            await thinking_msg.edit_text("⚠️ No response\.")
            return

        await thinking_msg.delete()

        for chunk in textwrap.wrap(
            response, width=3800, replace_whitespace=False,
            drop_whitespace=False, break_long_words=False
        ):
            if chunk.strip():
                esc = chunk.replace("\\", "\\\\").replace("_", "\\_").replace("*", "\\*")\
                           .replace("[", "\\[").replace("]", "\\]").replace("(", "\\(")\
                           .replace(")", "\\)").replace("~", "\\~").replace("`", "\\`")\
                           .replace(">", "\\>").replace("#", "\\#").replace("+", "\\+")\
                           .replace("-", "\\-").replace("=", "\\=").replace("|", "\\|")\
                           .replace("{", "\\{").replace("}", "\\}").replace(".", "\\.")\
                           .replace("!", "\\!")
                await update.message.reply_text(esc, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        await status_msg.edit_text(f"❌ Error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    token = read_token()
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("thinking", thinking_cmd))
    app.add_handler(CommandHandler("new", new_session_cmd))
    app.add_handler(CommandHandler("abort", abort_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Start pi subprocess
    async def on_startup(app):
        await pi.start()
        print("Bridge started.", flush=True)

    async def on_shutdown(app):
        await pi.stop()
        print("Bridge stopped.", flush=True)

    app.post_init = on_startup
    app.post_shutdown = on_shutdown

    print("Bot starting...", flush=True)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
