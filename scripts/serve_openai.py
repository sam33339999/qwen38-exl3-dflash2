#!/usr/bin/env python3
"""
Minimal OpenAI-compatible server for the EXL3 Qwen3.8-27B + DFlash2 stack.

Implements exactly what the r0b0bench / Q200v2 harnesses need:
  GET  /health
  GET  /v1/models                       (max_model_len = 262144)
  POST /v1/chat/completions             (JSON, or SSE when stream=true)
  POST /v1/chat/completions/render      (exact chat-template token ids)
  POST /v1/completions                  (raw prompt str or token-id list; SSE when stream=true)

Cache is quantized (cq3), full advertised 262k context, one sequence slot
(max_history = draft window for the DFlash2 verify pass). Requests are served
FIFO through a single worker thread; the generator batches nothing larger than
one job at this context on a 24 GB card.

Usage:
  python3 scripts/serve_openai.py --target models/qwen38-27b-exl3 \
      --draft models/dflash2-exl3 --port 8889
"""
import argparse
import json
import os
import queue
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3"))

import torch  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402
from exllamav3.generator.sampler.custom import (  # noqa: E402
    SS, SS_Argmax, SS_Base, SS_DRY, SS_PresFreqP, SS_RepP, SS_Sample, SS_Temperature,
    CustomSampler,
)

MAX_BODY = 32 * 1024 * 1024
_THINK_CLOSE = "</think>"


class ThinkStream:
    """Split incremental generator text on a single </think> tag.

    Thinking prompts leave the model inside an open think block, so text before
    the close tag is reasoning and text after it is the answer. A close tag can
    arrive split across chunks; a suffix that is still a prefix of </think> is
    held until the next chunk or finish().
    """

    def __init__(self, in_think):
        self.in_think = bool(in_think)
        self.pending = ""
        # Match the non-stream split, which strips whitespace after </think>
        # even when that whitespace arrives in a later chunk.
        self.skip_ws = False

    def _content(self, text):
        if self.skip_ws:
            text = text.lstrip("\n ")
            if not text:
                return []
            self.skip_ws = False
        return [("content", text)]

    def feed(self, text):
        if not text:
            return []
        if not self.in_think:
            return self._content(text)
        self.pending += text
        idx = self.pending.find(_THINK_CLOSE)
        if idx >= 0:
            reasoning = self.pending[:idx]
            rest = self.pending[idx + len(_THINK_CLOSE):]
            self.pending = ""
            self.in_think = False
            self.skip_ws = True
            out = []
            if reasoning:
                out.append(("reasoning", reasoning))
            out.extend(self._content(rest))
            return out
        hold = 0
        for n in range(1, len(_THINK_CLOSE)):
            if self.pending.endswith(_THINK_CLOSE[:n]):
                hold = n
        emit = self.pending[:-hold] if hold else self.pending
        self.pending = self.pending[-hold:] if hold else ""
        return [("reasoning", emit)] if emit else []

    def finish(self):
        tail = self.pending
        self.pending = ""
        if not tail:
            return []
        if self.in_think:
            return [("reasoning", tail)]
        return self._content(tail)


def draft_acceptance(new_tokens, accepted):
    """Mean tokens per verification round. Same definition as acceptance_check.py."""
    if not new_tokens:
        return 0.0
    rounds = max(1, new_tokens - accepted)
    return new_tokens / rounds


def stop_label(eos_reason, in_think, think_closed):
    """Say whether generation finished, hit the cap, and whether that was inside the think block."""
    still_thinking = bool(in_think) and not think_closed
    if eos_reason == "cancelled":
        return "客戶端中止"
    if eos_reason == "max_new_tokens" and still_thinking:
        return "思考中碰到 max_tokens，被截斷"
    if eos_reason == "max_new_tokens":
        return "回答中碰到 max_tokens，被截斷"
    if eos_reason == "stop_token" and still_thinking:
        return "模型在思考中自行停止，沒有寫出回答"
    if eos_reason == "stop_token":
        return "正常結束"
    if eos_reason == "stop_string":
        return "碰到 stop 字串結束"
    if eos_reason == "loop_detected" and still_thinking:
        return "思考內容重複打轉，已停止"
    if eos_reason == "loop_detected":
        return "偵測到重複迴圈而停止"
    if eos_reason == "end_filter":
        return "filter 結束"
    return f"結束 ({eos_reason or 'unknown'})"


class ClientGone(Exception):
    pass


class SS_ThinkExit(SS_Base):
    """Nudge a long think trace toward </think> without ending the request.

    Exact repeats are already handled by DRY. The failure here is a checklist that
    keeps changing the quoted words, so the same sentence never repeats. After
    `budget` generated tokens with the think block still open, the close-tag logit
    rises linearly and the model can start the answer.
    """

    def __init__(self, close_id, prompt_len, budget, ramp, max_bias):
        self.close_id = int(close_id)
        self.prompt_len = int(prompt_len)
        self.budget = int(budget)
        self.ramp = max(1, int(ramp))
        self.max_bias = float(max_bias)

    def reqs_past_ids(self):
        return True

    def _bias(self, past_ids):
        if past_ids is None or past_ids.numel() == 0:
            return 0.0
        row = past_ids[0] if past_ids.dim() == 2 else past_ids
        prompt_len = min(self.prompt_len, row.shape[0])
        if (row[:prompt_len] == self.close_id).any():
            return 0.0
        generated = row[prompt_len:]
        if generated.numel() == 0 or (generated == self.close_id).any():
            return 0.0
        over = int(generated.shape[0]) - self.budget
        if over <= 0:
            return 0.0
        return min(self.max_bias, over / self.ramp * self.max_bias)

    def run(self, state):
        if state.state == SS.INIT:
            state.logits = state.in_logits.to(torch.float, copy = True)
            state.state = SS.LOGITS
        if state.state != SS.LOGITS or self.close_id >= state.dim:
            return
        bias = self._bias(state.past_ids)
        if bias:
            state.logits[..., self.close_id] += bias


class Server:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.max_model_len = args.max_model_len
        self.default_max_tokens = args.default_max_tokens
        window = args.loop_window
        reps = args.loop_min_reps
        self.loop_stop = (window, reps) if window > 1 and 1 < reps < window else None
        self.rep_p = args.rep_penalty
        self.freq_p = args.freq_penalty
        self.freq_range = args.freq_range
        self.dry_multiplier = args.dry_multiplier
        self.dry_base = args.dry_base
        self.dry_allowed_length = args.dry_allowed_length
        self.dry_range = args.dry_range
        self.think_budget = args.think_budget
        self.think_ramp = args.think_ramp
        self.think_bias = args.think_bias

        tcfg = Config.from_directory(args.target)
        dcfg = Config.from_directory(args.draft)
        draft_model = Model.from_config(dcfg)
        max_history = draft_model.caps.get("default_draft_size", 4)

        model = Model.from_config(tcfg)
        print(f"[serve] loading target {args.target} (cache {args.cache_tokens} tokens, cq{args.cq}) ...", flush=True)
        t0 = time.time()
        cache = Cache(model, max_num_tokens = args.cache_tokens, layer_type = CacheLayer_quant,
                      k_bits = args.cq, v_bits = args.cq,
                      max_history = max_history, max_batch_size = 1)
        model.load(progressbar = False)
        tokenizer = Tokenizer.from_config(tcfg)

        draft_cache = Cache(draft_model, max_num_tokens = args.cache_tokens, layer_type = CacheLayer_quant,
                            k_bits = args.cq, v_bits = args.cq, max_batch_size = 1)
        draft_model.load(progressbar = False)
        self.gen = Generator(model, cache, tokenizer, draft_model = draft_model, draft_cache = draft_cache)
        self.tokenizer = tokenizer
        close = tokenizer.encode("</think>", encode_special_tokens = True)[0].tolist()
        self.think_close_id = close[0] if len(close) == 1 else None
        if self.think_close_id is None:
            print(f"[serve] </think> is {close} tokens; think-exit bias disabled", flush = True)
        free, total = torch.cuda.mem_get_info()
        print(f"[serve] loaded in {time.time()-t0:.0f}s; VRAM in use {(total-free)/1e9:.2f} GB", flush=True)

        eos_ids = list(getattr(model.config, "eos_token_id_list", None) or [])
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(tokenizer.eos_token_id)
        self.stop_ids = eos_ids

        self.queue = queue.Queue()
        self.worker = threading.Thread(target = self._worker, daemon = True)
        self.worker.start()
        print(
            f"[serve] READY  context={self.max_model_len}  default_max_tokens={self.default_max_tokens}  "
            f"loop={('%dx%d' % self.loop_stop) if self.loop_stop else 'off'}  "
            f"dry={self.dry_multiplier} rep={self.rep_p} freq={self.freq_p}/{self.freq_range}  "
            f"think_budget={self.think_budget}+{self.think_ramp}",
            flush=True,
        )

    # ---------------------------------------------------------------- worker

    def submit(self, input_ids, max_new_tokens, temperature, skip_special_tokens, want_ids,
               stream = False, in_think = False, poll = None, penalties = None):
        """Queue one job.

        Non-stream calls block until the job finishes and return the result dict.
        Stream calls return a holder immediately. The worker pushes each text
        delta on holder["q"] and then None; the result dict is on holder["result"].
        holder["cancel"] stops the job between generation steps. poll() should
        return True when the HTTP client has gone away.
        """
        req_id = uuid.uuid4().hex
        holder = {
            "event": threading.Event(),
            "result": None,
            "error": None,
            "cancel": threading.Event(),
            "q": queue.Queue() if stream else None,
        }
        self.queue.put({
            "id": req_id,
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "decode_special_tokens": not skip_special_tokens,
            "holder": holder,
            "want_ids": want_ids,
            "stream_q": holder["q"],
            "in_think": bool(in_think),
            "penalties": penalties or {},
        })
        if stream:
            return holder
        deadline = time.time() + 7200
        while not holder["event"].wait(timeout = 0.5):
            if time.time() >= deadline:
                holder["cancel"].set()
                break
            if poll and poll():
                holder["cancel"].set()
        holder["event"].wait(timeout = 30)
        if holder["error"]:
            raise RuntimeError(holder["error"])
        return holder["result"]

    def _worker(self):
        while True:
            spec = self.queue.get()
            try:
                pen = spec.get("penalties") or {}
                ids = spec["input_ids"]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                sampler = self._sampler(spec["temperature"], pen, ids, spec.get("in_think"))
                job = Job(
                    input_ids = ids,
                    max_new_tokens = spec["max_new_tokens"],
                    sampler = sampler,
                    stop_conditions = self.stop_ids,
                    decode_special_tokens = spec["decode_special_tokens"],
                    identifier = spec["id"],
                    stop_on_loop = self.loop_stop,
                )
                t0 = time.time()
                tag = spec["id"][:8]
                in_think = bool(spec.get("in_think"))
                print(
                    f"[gen] {tag} 開始 max_tokens={spec['max_new_tokens']} thinking={'on' if in_think else 'off'} "
                    f"dry={pen.get('dry_multiplier', self.dry_multiplier)} "
                    f"rep={pen.get('rep_p', self.rep_p)}",
                    flush = True,
                )
                self.gen.enqueue(job)
                final = None
                out_ids = []
                stream_q = spec.get("stream_q")
                splitter = ThinkStream(in_think)
                t_first = None
                last_log = 0.0
                live_tokens = 0
                accepted = 0
                rejected = 0
                stopped = None
                cancel = spec["holder"]["cancel"]
                while self.gen.num_remaining_jobs():
                    if cancel.is_set():
                        self.gen.cancel(job)
                        stopped = "cancelled"
                        break
                    for r in self.gen.iterate():
                        if r.get("identifier") != spec["id"]:
                            continue
                        if r.get("stage") == "streaming":
                            piece = r.get("text") or ""
                            if piece and stream_q is not None:
                                stream_q.put(piece)
                            if piece:
                                splitter.feed(piece)
                            job_obj = r.get("job")
                            if job_obj is not None:
                                live_tokens = int(getattr(job_obj, "new_tokens", 0) or 0)
                                accepted = int(getattr(job_obj, "accepted_draft_tokens", 0) or 0)
                                rejected = int(getattr(job_obj, "rejected_draft_tokens", 0) or 0)
                            now = time.time()
                            if t_first is None and live_tokens:
                                t_first = now
                                last_log = now
                            elif live_tokens and now - last_log >= 1.0:
                                dt = now - t_first
                                phase = "思考中" if splitter.in_think else "回答中"
                                al = draft_acceptance(live_tokens, accepted)
                                print(
                                    f"[gen] {tag} {live_tokens} tok  {live_tokens / dt:.1f} tok/s  "
                                    f"dflash acc={al:.2f} accept={accepted} rej={rejected}  {phase}",
                                    flush = True,
                                )
                                last_log = now
                            tid = r.get("token_ids")
                            if tid is not None and spec["want_ids"]:
                                out_ids.extend(tid.torch().flatten().tolist() if hasattr(tid, "torch") else list(tid))
                        if r.get("eos"):
                            final = r
                    if stopped:
                        break
                    if cancel.is_set():
                        if not (final and final.get("eos")):
                            self.gen.cancel(job)
                            stopped = "cancelled"
                        break
                r = final or {}
                if stopped:
                    r = dict(r)
                    r["eos_reason"] = stopped
                    if not r.get("full_completion"):
                        r["full_completion"] = getattr(job, "full_completion", "") or ""
                text = r.get("full_completion")
                if text is None:
                    text = r.get("text") or ""
                new_tokens = int(r.get("new_tokens") or live_tokens or 0)
                accepted = int(r.get("accepted_draft_tokens") if r.get("accepted_draft_tokens") is not None else accepted)
                rejected = int(r.get("rejected_draft_tokens") if r.get("rejected_draft_tokens") is not None else rejected)
                gen_time = float(r.get("time_generate") or 0.0)
                if gen_time <= 0 and t_first is not None:
                    gen_time = time.time() - t_first
                tok_s = (new_tokens / gen_time) if gen_time > 0 and new_tokens else 0.0
                al = draft_acceptance(new_tokens, accepted)
                think_closed = not splitter.in_think
                label = stop_label(r.get("eos_reason"), in_think, think_closed)
                phase = "thinking" if in_think and splitter.in_think else "answer"
                phase_zh = "思考中" if phase == "thinking" else "回答中"
                prefill = r.get("time_prefill")
                prefill_s = f"  prefill {float(prefill):.2f}s" if isinstance(prefill, (int, float)) else ""
                print(
                    f"[gen] {tag} 結束：{label}  {new_tokens} tok  {tok_s:.1f} tok/s  "
                    f"dflash acc={al:.2f} accept={accepted} rej={rejected}  {phase_zh}{prefill_s}",
                    flush = True,
                )
                stats = {
                    "eos_reason": r.get("eos_reason"),
                    "stop": label,
                    "phase": phase,
                    "tok_per_s": round(tok_s, 2),
                    "acceptance_length": round(al, 3),
                    "accepted_draft_tokens": accepted,
                    "rejected_draft_tokens": rejected,
                    "max_tokens": spec["max_new_tokens"],
                    "prefill_s": round(float(prefill), 3) if isinstance(prefill, (int, float)) else None,
                }
                spec["holder"]["result"] = {
                    "text": text,
                    "prompt_tokens": int(r.get("prompt_tokens") or ids.shape[-1]),
                    "completion_tokens": new_tokens,
                    "eos_reason": r.get("eos_reason"),
                    "token_ids": out_ids if spec["want_ids"] else None,
                    "elapsed": time.time() - t0,
                    "accepted_draft_tokens": accepted,
                    "rejected_draft_tokens": rejected,
                    "stats": stats,
                }
                if stream_q is not None:
                    stream_q.put(None)
            except Exception as exc:  # noqa: BLE001
                spec["holder"]["error"] = f"{type(exc).__name__}: {exc}"
                if spec.get("stream_q") is not None:
                    spec["stream_q"].put(exc)
            finally:
                spec["holder"]["event"].set()

    # ---------------------------------------------------------------- helpers

    def _sampler(self, temperature, pen, ids, in_think):
        freq_range = int(pen.get("freq_range", self.freq_range))
        steps = [
            SS_RepP(pen.get("rep_p", self.rep_p), freq_range, 0),
            SS_PresFreqP(pen.get("pres_p", 0.0), pen.get("freq_p", self.freq_p), freq_range, 0),
            SS_DRY(
                pen.get("dry_multiplier", self.dry_multiplier),
                pen.get("dry_base", self.dry_base),
                pen.get("dry_allowed_length", self.dry_allowed_length),
                pen.get("dry_range", self.dry_range),
            ),
        ]
        if in_think and self.think_close_id is not None and self.think_budget > 0:
            steps.append(SS_ThinkExit(
                self.think_close_id,
                int(ids.shape[-1]),
                pen.get("think_budget", self.think_budget),
                pen.get("think_ramp", self.think_ramp),
                pen.get("think_bias", self.think_bias),
            ))
        if not temperature:
            steps.append(SS_Argmax())
        else:
            steps.append(SS_Temperature(temperature))
            steps.append(SS_Sample())
        return CustomSampler(steps)

    def render_chat(self, messages, template_kwargs):
        ids = self.tokenizer.hf_chat_template(
            messages, add_generation_prompt = True, **(template_kwargs or {})
        )
        return ids

    def finish_reason(self, r):
        if r.get("eos_reason") == "max_new_tokens":
            return "length"
        return "stop"


SERVE: Server = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _begin_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._sse_started = True

    def _sse(self, payload):
        if payload is None:
            block = b"data: [DONE]\n\n"
        else:
            block = b"data: " + json.dumps(payload, ensure_ascii = False).encode() + b"\n\n"
        self.wfile.write(f"{len(block):X}\r\n".encode())
        self.wfile.write(block)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _usage(self, result):
        prompt = int(result.get("prompt_tokens") or 0)
        completion = int(result.get("completion_tokens") or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):  # noqa: N802
        if self.path in ("/health", "/v1/health"):
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {
                "object": "list",
                "data": [{
                    "id": SERVE.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "max_model_len": SERVE.max_model_len,
                }],
            })
        else:
            self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):  # noqa: N802
        self._sse_started = False
        try:
            body = self._read_body()
        except Exception as exc:  # noqa: BLE001
            return self._send(400, {"error": {"message": f"invalid JSON: {exc}"}})
        try:
            if self.path == "/v1/chat/completions/render":
                return self._render(body)
            if self.path == "/v1/chat/completions":
                if body.get("stream"):
                    return self._chat_stream(body)
                return self._chat(body)
            if self.path == "/v1/completions":
                if body.get("stream"):
                    return self._completions_stream(body)
                return self._completions(body)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001
            if self._sse_started:
                return
            return self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
        return self._send(404, {"error": {"message": "not found"}})

    def _render(self, body):
        messages = body.get("messages") or []
        kwargs = body.get("chat_template_kwargs") or {}
        ids = SERVE.render_chat(messages, kwargs)
        self._send(200, {"token_ids": ids[0].tolist() if ids.dim() == 2 else ids.tolist()})

    def _chat(self, body):
        model_name = body.get("model") or SERVE.model_name
        messages = body.get("messages") or []
        template_kwargs = body.get("chat_template_kwargs") or {}
        ids = SERVE.render_chat(messages, template_kwargs)
        max_tokens = int(body.get("max_tokens") or SERVE.default_max_tokens)
        temperature = float(body.get("temperature") or 0)
        in_think = template_kwargs.get("enable_thinking", True) is not False
        r = SERVE.submit(ids, max_tokens, temperature,
                         skip_special_tokens = body.get("skip_special_tokens", True),
                         want_ids = False, in_think = in_think, poll = self._client_closed,
                         penalties = self._penalties(body))
        # Thinking models: keep the reasoning trace out of `content` so quality graders
        # judge the answer, not the reasoning (vLLM reasoning-parser behavior).
        text = r["text"]
        reasoning_content = None
        content = text
        if "</think>" in text:
            reasoning_content, content = text.split("</think>", 1)
            content = content.lstrip("\n ")
            if not content.strip():
                content = text
        self._send(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning_content": reasoning_content},
                "finish_reason": SERVE.finish_reason(r),
            }],
            "usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            },
            "stats": r.get("stats"),
        })

    def _completions(self, body):
        model_name = body.get("model") or SERVE.model_name
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = torch.tensor([prompt], dtype = torch.long)
        elif isinstance(prompt, str):
            ids = SERVE.tokenizer.encode(prompt)
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            ids = torch.tensor([prompt[0]], dtype = torch.long)
        else:
            return self._send(400, {"error": {"message": "prompt must be a string or a token-id list"}})
        max_tokens = int(body.get("max_tokens") or 256)
        temperature = float(body.get("temperature") or 0)
        r = SERVE.submit(ids, max_tokens, temperature,
                         skip_special_tokens = body.get("skip_special_tokens", True),
                         want_ids = False, poll = self._client_closed,
                         penalties = self._penalties(body))
        self._send(200, {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": r["text"],
                "finish_reason": SERVE.finish_reason(r),
            }],
            "usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            },
            "stats": r.get("stats"),
        })

    def _opt_float(self, body, key, default):
        value = body.get(key)
        if value is None or value == "":
            return default
        return float(value)

    def _penalties(self, body):
        # Omitted fields use the server defaults. An explicit 0 turns that penalty off.
        return {
            "rep_p": self._opt_float(body, "repetition_penalty", SERVE.rep_p),
            "freq_p": self._opt_float(body, "frequency_penalty", SERVE.freq_p),
            "freq_range": int(self._opt_float(body, "frequency_range", SERVE.freq_range)),
            "pres_p": self._opt_float(body, "presence_penalty", 0.0),
            "dry_multiplier": self._opt_float(body, "dry_multiplier", SERVE.dry_multiplier),
            "dry_base": self._opt_float(body, "dry_base", SERVE.dry_base),
            "dry_allowed_length": int(self._opt_float(body, "dry_allowed_length", SERVE.dry_allowed_length)),
            "dry_range": int(self._opt_float(body, "dry_range", SERVE.dry_range)),
        }

    def _client_closed(self):
        sock = self.connection
        try:
            readable, _, _ = select.select([sock], [], [], 0)
        except (OSError, ValueError):
            return True
        if not readable:
            return False
        try:
            peek = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        except BlockingIOError:
            return False
        except OSError:
            return True
        return peek == b""

    def _iter_stream(self, holder):
        while True:
            try:
                item = holder["q"].get(timeout = 0.5)
            except queue.Empty:
                if self._client_closed():
                    holder["cancel"].set()
                    raise ClientGone()
                continue
            if item is None:
                if holder["error"]:
                    raise RuntimeError(holder["error"])
                return holder["result"] or {}
            if isinstance(item, BaseException):
                raise RuntimeError(holder["error"] or str(item))
            yield item

    def _chat_stream(self, body):
        model_name = body.get("model") or SERVE.model_name
        messages = body.get("messages") or []
        template_kwargs = body.get("chat_template_kwargs") or {}
        ids = SERVE.render_chat(messages, template_kwargs)
        max_tokens = int(body.get("max_tokens") or SERVE.default_max_tokens)
        temperature = float(body.get("temperature") or 0)
        in_think = template_kwargs.get("enable_thinking", True) is not False
        holder = SERVE.submit(ids, max_tokens, temperature,
                              skip_special_tokens = body.get("skip_special_tokens", True),
                              want_ids = False, stream = True, in_think = in_think,
                              penalties = self._penalties(body))
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        try:
            self._begin_sse()
            self._sse(self._chat_chunk(cid, created, model_name, {"role": "assistant"}, None))
            splitter = ThinkStream(in_think)
            for piece in self._iter_stream(holder):
                for kind, text in splitter.feed(piece):
                    delta = {"reasoning_content": text} if kind == "reasoning" else {"content": text}
                    self._sse(self._chat_chunk(cid, created, model_name, delta, None))
            result = holder["result"] or {}
            for kind, text in splitter.finish():
                delta = {"reasoning_content": text} if kind == "reasoning" else {"content": text}
                self._sse(self._chat_chunk(cid, created, model_name, delta, None))
            self._sse(self._chat_chunk(
                cid, created, model_name, {}, SERVE.finish_reason(result), self._usage(result),
                result.get("stats"),
            ))
            self._sse(None)
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError, ClientGone):
            holder["cancel"].set()
            return
        except Exception as exc:  # noqa: BLE001
            self._sse_error(exc)

    def _completions_stream(self, body):
        model_name = body.get("model") or SERVE.model_name
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = torch.tensor([prompt], dtype = torch.long)
        elif isinstance(prompt, str):
            ids = SERVE.tokenizer.encode(prompt)
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], list):
            ids = torch.tensor([prompt[0]], dtype = torch.long)
        else:
            return self._send(400, {"error": {"message": "prompt must be a string or a token-id list"}})
        max_tokens = int(body.get("max_tokens") or 256)
        temperature = float(body.get("temperature") or 0)
        holder = SERVE.submit(ids, max_tokens, temperature,
                              skip_special_tokens = body.get("skip_special_tokens", True),
                              want_ids = False, stream = True,
                              penalties = self._penalties(body))
        cid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        try:
            self._begin_sse()
            for piece in self._iter_stream(holder):
                self._sse(self._completion_chunk(cid, created, model_name, piece, None))
            result = holder["result"] or {}
            self._sse(self._completion_chunk(
                cid, created, model_name, "", SERVE.finish_reason(result), self._usage(result),
                result.get("stats"),
            ))
            self._sse(None)
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError, ClientGone):
            holder["cancel"].set()
            return
        except Exception as exc:  # noqa: BLE001
            self._sse_error(exc)

    def _sse_error(self, exc):
        if not self._sse_started:
            raise exc
        try:
            self._sse({"error": {"message": str(exc), "type": "server_error"}})
            self._sse(None)
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _chat_chunk(self, cid, created, model, delta, finish, usage = None, stats = None):
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
        }
        if usage is not None:
            payload["usage"] = usage
        if stats is not None:
            payload["stats"] = stats
        return payload

    def _completion_chunk(self, cid, created, model, text, finish, usage = None, stats = None):
        payload = {
            "id": cid,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": finish,
            }],
        }
        if usage is not None:
            payload["usage"] = usage
        if stats is not None:
            payload["stats"] = stats
        return payload


def main():
    global SERVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required = True)
    ap.add_argument("--draft", required = True)
    ap.add_argument("--host", default = "127.0.0.1")
    ap.add_argument("--port", type = int, default = 8889)
    ap.add_argument("--model-name", default = "qwen38-27b-exl3-dflash2")
    ap.add_argument("--max-model-len", type = int, default = 262144)
    ap.add_argument("--cache-tokens", type = int, default = 270336)
    ap.add_argument("--cq", type = int, default = 3)
    ap.add_argument("--default-max-tokens", type = int, default = 32768,
                    help = "chat max_tokens when the client omits it")
    ap.add_argument("--loop-window", type = int, default = 0,
                    help = "hard-stop token loop window; 0 leaves repetition to the sampler penalty")
    ap.add_argument("--loop-min-reps", type = int, default = 3)
    ap.add_argument("--rep-penalty", type = float, default = 1.0,
                    help = "transformers repetition penalty; 1.0 disables it")
    ap.add_argument("--freq-penalty", type = float, default = 0.3,
                    help = "penalize tokens already used in the recent window")
    ap.add_argument("--freq-range", type = int, default = 512)
    ap.add_argument("--think-budget", type = int, default = 4096,
                    help = "generated tokens before </think> starts getting a logit bonus; 0 disables")
    ap.add_argument("--think-ramp", type = int, default = 1536)
    ap.add_argument("--think-bias", type = float, default = 16.0)
    ap.add_argument("--dry-multiplier", type = float, default = 0.8,
                    help = "DRY sequence penalty; 0 disables it")
    ap.add_argument("--dry-base", type = float, default = 1.75)
    ap.add_argument("--dry-allowed-length", type = int, default = 2)
    ap.add_argument("--dry-range", type = int, default = 4096,
                    help = "how many recent tokens DRY looks at; 0 means the whole context")
    args = ap.parse_args()

    SERVE = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(
        f"[serve] listening on http://{args.host}:{args.port}  default_max_tokens={args.default_max_tokens}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
