#!/usr/bin/env python3
"""Ephemeral B-0244 target-side command client."""

import base64
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request


OUTPUT_CAP = 4_194_304


def exchange(path, body):
    request = urllib.request.Request(
        os.environ["DYN_ENDPOINT"].rstrip("/") + path,
        data=json.dumps(body, separators=(",", ":")).encode(),
        method="POST",
        headers={
            "X-Dyn-Token": os.environ["DYN_TARGET_TOKEN"],
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except Exception:
            return error.code, {"error": "http_error"}


def execute(command):
    started = time.monotonic()
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command["shell"]],
        cwd=command["cwd"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=int(command["timeout_seconds"]))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    return {
        "exit_code": process.returncode if process.returncode >= 0 else None,
        "signal": -process.returncode if process.returncode < 0 else None,
        "timed_out": timed_out,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "stdout_b64": base64.b64encode(stdout[:OUTPUT_CAP]).decode(),
        "stderr_b64": base64.b64encode(stderr[:OUTPUT_CAP]).decode(),
        "stdout_truncated": len(stdout) > OUTPUT_CAP,
        "stderr_truncated": len(stderr) > OUTPUT_CAP,
        "runner": "python-direct-macos",
    }


def main():
    session_id = os.environ["DYN_SESSION_ID"]
    build_id = os.environ["DYN_BUILD_ID"]
    after_seq = 0
    while True:
        try:
            status, response = exchange(
                "/v1/target/poll",
                {"session_id": session_id, "after_seq": after_seq, "build_id": build_id},
            )
        except Exception:
            time.sleep(2)
            continue
        if status != 200:
            if status in (401, 404):
                return
            time.sleep(2)
            continue
        kind = response.get("kind")
        if kind == "wait":
            time.sleep(min(5, max(1, int(response.get("retry_after_ms", 1500)) / 1000)))
            continue
        if kind == "complete":
            after_seq = int(response["seq"])
            continue
        if kind != "command":
            return
        command = response["command"]
        if command.get("kind") == "stop":
            return
        seq = int(command["seq"])
        result = {
            "session_id": session_id,
            "build_id": build_id,
            "seq": seq,
            **execute(command),
        }
        while True:
            try:
                result_status, _ = exchange("/v1/target/results", result)
            except Exception:
                result_status = 0
            if result_status in (200, 201):
                after_seq = seq
                break
            if result_status in (401, 404):
                return
            time.sleep(2)


if __name__ == "__main__":
    main()
