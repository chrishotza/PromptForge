#!/usr/bin/env python3
"""Offline-source, loopback-only probe for MLX-Serve Gemma 4 post-tool continuations.

Ten fixed, synthetic cases (five histories x streaming/non-streaming).
Does not read Xcode projects, run generated code or contact the internet.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DEFAULT_MODEL = "mlx-community/gemma-4-12b-it-4bit"

def fixtures():
    lookup = {"name": "lookup_value", "description": "Return a synthetic value",
              "input_schema": {"type": "object", "properties": {"key": {"type": "string"}},
                               "required": ["key"]}}
    read = {"name": "read_source", "description": "Read a synthetic Swift snippet",
            "input_schema": {"type": "object", "properties": {"file": {"type": "string"}},
                             "required": ["file"]}}
    simple = [{"role": "user", "content": "Reply with exactly 42."}]
    def after_tools(count):
        messages = [{"role": "user", "content": "Use the lookup result, then answer with only 42."}]
        for i in range(count):
            tid = "toolu_synthetic_" + str(i)
            messages.extend([
                {"role": "assistant", "content": [{"type": "tool_use", "id": tid,
                    "name": "lookup_value", "input": {"key": "alpha"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
                    "content": "alpha = 42"}]}
            ])
        return messages
    swift = [
        {"role": "user", "content": "Read the Swift file; reply with PingManager only."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_swift",
            "name": "read_source", "input": {"file": "Synthetic.swift"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_swift",
            "content": "import Foundation\nstruct PingManager { }\n"}]}
    ]
    return [
        ("basic", simple, [], "42"),
        ("tools_declared", simple, [lookup], "42"),
        ("after_one_tool", after_tools(1), [lookup], "42"),
        ("after_two_tools", after_tools(2), [lookup], "42"),
        ("after_swift_read", swift, [read], "PingManager"),
    ]

def parse_response(data, stream):
    texts, thoughts, tools = [], [], []
    stop, anomalies, complete = None, 0, False
    if not stream:
        obj = json.loads(data)
        for block in obj.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                thoughts.append(block.get("thinking", ""))
            elif block.get("type") == "tool_use":
                tools.append(block.get("name", ""))
        stop = obj.get("stop_reason")
        complete = isinstance(obj.get("content"), list)
    else:
        opened = set()
        for line in data.decode("utf-8", "replace").splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                ev = json.loads(payload)
            except ValueError:
                anomalies += 1
                continue
            typ, idx = ev.get("type"), ev.get("index")
            if typ == "content_block_start":
                if idx in opened:
                    anomalies += 1
                opened.add(idx)
                block = ev.get("content_block") or {}
                if block.get("type") == "tool_use":
                    tools.append(block.get("name", ""))
            elif typ == "content_block_delta":
                if idx not in opened:
                    anomalies += 1
                delta = ev.get("delta") or {}
                if delta.get("type") == "text_delta":
                    texts.append(delta.get("text", ""))
                elif delta.get("type") == "thinking_delta":
                    thoughts.append(delta.get("thinking", ""))
            elif typ == "content_block_stop":
                opened.discard(idx)
            elif typ == "message_delta":
                stop = (ev.get("delta") or {}).get("stop_reason") or stop
            elif typ == "message_stop":
                complete = True
        anomalies += len(opened)
    text, thought = "".join(texts), "".join(thoughts)
    marker = text.count("<|channel>thought") + thought.count("<|channel>thought")
    return dict(text=text, thinking_len=len(thought), tools=tools, stop=stop,
                marker_count=marker, anomalies=anomalies, complete=complete)

def classify(out, expected):
    if not out["complete"] or out["anomalies"]:
        return "PROTOCOL_FAILURE"
    if out["marker_count"]:
        return "CHANNEL_MARKER_ANOMALY"
    if out["tools"]:
        return "TOOL_CALL_INSTEAD_OF_ANSWER"
    if not out["text"].strip():
        return "EMPTY_OR_THINKING_ONLY"
    if out["text"].strip(" .\n") == expected:
        return "CONTENT_OK"
    return "WRONG_CONTENT"

def call(base, route, body, timeout):
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    key = os.environ.get("MLX_SERVE_API_KEY", "")
    if key:
        headers["x-api-key"] = key
    req = Request(base.rstrip("/") + route, headers=headers,
                  data=json.dumps(body).encode() if body is not None else None)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read(3_000_000)
    except HTTPError as exc:
        status = exc.code
        exc.close()
        raise RuntimeError("HTTP_" + str(status)) from None
    except URLError as exc:
        raise RuntimeError("LOCAL_CONNECT_FAILED_" + type(exc.reason).__name__) from None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11234")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compare-model")
    parser.add_argument("--output", default="gemma4_xcode_probe_report.json")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    host = urlsplit(args.base_url)
    if host.scheme != "http" or host.hostname not in {"127.0.0.1", "localhost", "::1"} or host.username or host.password:
        parser.error("A plain HTTP loopback server is required.")
    if host.query or host.fragment:
        parser.error("No URL query or fragment allowed.")
    if args.timeout < 1 or not 32 <= args.max_tokens <= 1024:
        parser.error("Invalid timeout or max-tokens.")
    try:
        listing = json.loads(call(args.base_url, "/v1/models", None, args.timeout))
        available = [x.get("id") for x in listing.get("data", [])]
    except (RuntimeError, ValueError) as exc:
        parser.error("Local preflight failed: " + type(exc).__name__ + ": " + str(exc))
    models = [args.model] + ([args.compare_model] if args.compare_model else [])
    for m in models:
        if m not in available:
            parser.error("Model not advertised by local /v1/models: " + str(m))
    report = {"schema": "mlx-xcode-synthetic-probe-v1", "model_ids": models,
              "python": platform.python_version(), "os": platform.system(),
              "os_release": platform.release(), "server": args.base_url,
              "privacy": "Synthetic only; no generated text, project source, API keys, or raw server logs",
              "cases": []}
    runs = [(model, name, msgs, tools, expected, streaming)
            for model in models for name, msgs, tools, expected in fixtures()
            for streaming in (False, True)]
    print("Running", len(runs), "synthetic loopback-only requests", flush=True)
    for index, (model, name, msgs, tools, expected, streaming) in enumerate(runs, 1):
        body = {"model": model, "messages": msgs, "stream": streaming,
                "max_tokens": args.max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
        row = {"case": name, "model": model, "stream": streaming,
               "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
        start = time.monotonic()
        try:
            out = parse_response(call(args.base_url, "/v1/messages", body, args.timeout), streaming)
            row.update(status=classify(out, expected), marker_count=out["marker_count"],
                       visible_chars=len(out["text"]), reasoning_chars=out["thinking_len"],
                       tool_names=out["tools"], stop_reason=out["stop"], anomalies=out["anomalies"])
        except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
            row.update(status="REQUEST_ERROR", error_type=type(exc).__name__,
                       error_code=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__)
        row["elapsed_seconds"] = round(time.monotonic() - start, 2)
        report["cases"].append(row)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{index}/{len(runs)}: {name} stream={streaming}: {row['status']}", flush=True)
    counts = {}
    for case in report["cases"]:
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    report["summary"] = counts
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Complete:", counts, "Review before sharing", args.output)

if __name__ == "__main__":
    main()
