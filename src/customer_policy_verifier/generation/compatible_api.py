"""OpenRouter Chat Completions transport for resumable batch generation.

Loading configuration is offline. Only ``call`` performs inference; callers load
the key with ``pilot.load_key`` after the user has authorized batch execution.
"""

import email.utils
import json
import math
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from customer_policy_verifier.common.paths import resolve_path
from customer_policy_verifier.generation.pilot import schema_issues

API_BASE = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.7-flash"


def load_config() -> dict:
    """Return fresh, serializable settings without reading a key or the network."""
    return {"api_base": API_BASE, "model": MODEL,
            "client": "curl-default-user-agent", "temperature": 0.2,
            "max_tokens": 1800, "timeout_seconds": 180, "max_attempts": 4,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},
            "provider": {"require_parameters": True}}


class ApiFailure(RuntimeError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


class RequestGate:
    """Bound active requests; halve concurrency and share a cooldown after a 429."""

    def __init__(self, limit: int):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Concurrency must be a positive integer")
        self.limit = limit
        self.active = 0
        self.cooldown_until = 0.0
        self.rate_limit_events = 0
        self.stop = threading.Event()
        self.condition = threading.Condition()

    def acquire(self) -> None:
        with self.condition:
            while True:
                if self.stop.is_set():
                    raise InterruptedError("Run stopped; saved stages can be resumed")
                delay = self.cooldown_until - time.monotonic()
                if self.active < self.limit and delay <= 0:
                    self.active += 1
                    return
                self.condition.wait(timeout=min(max(delay, 0.1), 1.0))

    def release(self) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def throttle(self, seconds: float) -> None:
        with self.condition:
            self.limit = max(1, self.limit // 2)
            self.rate_limit_events += 1
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
            self.condition.notify_all()


def curl_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n") + '"'


def retry_after(headers: str) -> float | None:
    for line in headers.splitlines():
        name, _, value = line.partition(":")
        if name.lower() != "retry-after":
            continue
        try:
            delay = float(value.strip())
            if math.isfinite(delay):
                return max(0.0, delay)
        except ValueError:
            try:
                return max(0.0, email.utils.parsedate_to_datetime(value.strip()).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return None


def call(config: dict, key: str, body: dict, schema: dict, request_path: Path,
         gate: RequestGate) -> tuple[dict, dict]:
    if config.get("api_base") != API_BASE:
        raise ValueError("OpenRouter credentials may only be sent to " + API_BASE)
    if not isinstance(key, str) or not key.strip() or any(char in key for char in "\r\n"):
        raise ValueError("A nonempty OpenRouter key without line breaks is required")
    if not 1 <= config["max_attempts"] <= 4 or config["timeout_seconds"] <= 0:
        raise ValueError("Expected 1–4 attempts and a positive request timeout")
    for field in ("model", "temperature", "max_tokens", "response_format", "reasoning", "provider"):
        if body.get(field) != config[field]:
            raise ValueError(f"Request {field} does not match the frozen configuration")
    if body.get("stream", False) is not False:
        raise ValueError("Batch transport requires non-streaming responses")
    if key in json.dumps(body, ensure_ascii=False):
        raise ValueError("API keys must never appear in the saved request body")
    request_path = resolve_path(request_path).resolve()
    if json.loads(request_path.read_text(encoding="utf-8")) != body:
        raise ValueError("Saved request file does not match the request body")
    # curl receives credentials on stdin, never in process arguments or saved requests.
    curl_config = "\n".join([
        "url = " + curl_quote(config["api_base"] + "/chat/completions"),
        'request = "POST"',
        "header = " + curl_quote("Authorization: Bearer " + key),
        'header = "Content-Type: application/json"',
        "data-binary = " + curl_quote("@" + str(request_path)),
    ]) + "\n"
    attempts = []
    for attempt in range(config["max_attempts"]):
        gate.acquire()
        started = time.monotonic()
        try:
            # Private temporary headers, deleted even when curl fails or times out.
            with tempfile.NamedTemporaryFile(prefix="openrouter-", suffix=".headers") as headers_file:
                command = ["curl", "--disable", "--silent", "--show-error",
                           "--proto", "=https", "--connect-timeout", "15",
                           "--max-time", str(config["timeout_seconds"]), "--config", "-",
                           "--dump-header", headers_file.name, "--write-out", "\n%{http_code}"]
                try:
                    proc = subprocess.run(command, input=curl_config, text=True,
                                          capture_output=True,
                                          timeout=config["timeout_seconds"] + 10)
                except subprocess.TimeoutExpired:
                    proc = subprocess.CompletedProcess(command, 28, "", "curl process timed out")
                except OSError as exc:
                    raise ApiFailure(f"Could not start curl: {type(exc).__name__}", attempts) from exc
                headers = Path(headers_file.name).read_text(errors="replace")
        finally:
            gate.release()
        raw, _, code = proc.stdout.rpartition("\n")
        raw = raw.replace(key, "[REDACTED]")
        http_status = int(code) if code.isdigit() else 0
        item = {"attempt": attempt + 1, "http_status": http_status, "curl_exit": proc.returncode,
                "elapsed_seconds": round(time.monotonic() - started, 3)}
        attempts.append(item)
        error_status = http_status
        if proc.returncode == 0 and http_status == 200:
            try:
                response = json.loads(raw)
                if not isinstance(response, dict):
                    raise ValueError("Expected a response object")
            except (ValueError, TypeError) as exc:
                raise ApiFailure("HTTP 200 response is not a Chat Completions result", attempts) from exc
            if response.get("error"):
                error = response["error"]
                code = error.get("code") if isinstance(error, dict) else None
                error_status = int(code) if str(code).isdigit() else 0
                item["api_error_code"] = error_status
            else:
                return _parse_result(response, config, schema, attempts)
        item["error"] = (raw[:2000] or proc.stderr[:1000]).replace(key, "[REDACTED]")
        transient = error_status in {408, 429, 500, 502, 503, 504} or proc.returncode in {6, 7, 18, 28, 35, 52, 56}
        delay = retry_after(headers)
        if delay is None:
            delay = min(60, 5 * 2 ** attempt)
        if error_status == 429:
            gate.throttle(delay)
        if not transient or attempt + 1 == config["max_attempts"] or delay > 300:
            raise ApiFailure(f"API request failed: HTTP {http_status}, API {error_status}, curl {proc.returncode}", attempts)
        if gate.stop.wait(delay):
            raise InterruptedError("Run stopped during retry cooldown")
    raise AssertionError("Unreachable")


def _parse_result(response: dict, config: dict, schema: dict, attempts: list[dict]) -> tuple[dict, dict]:
    try:
        choice = response["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ApiFailure("HTTP 200 response is not a Chat Completions result", attempts) from exc
    problems = []
    if choice.get("finish_reason") != "stop":
        problems.append(f"finish_reason={choice.get('finish_reason')}")
    try:
        output = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        output = None
    if output is None:
        output = {"unparsed_content": content}
        problems.append("model output is not valid JSON")
    else:
        problems += schema_issues(output, schema["json_schema"]["schema"])
        if not isinstance(output, dict):
            output = {"unparsed_content": content}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    cost = usage.get("cost")
    numeric_cost = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0
    return output, {
        "id": response.get("id"), "model": response.get("model"),
        "requested_model": config["model"], "api_base": config["api_base"],
        "usage": usage, "reported_cost_usd": cost if numeric_cost else None,
        "finish_reason": choice.get("finish_reason"), "schema_issues": problems,
        "attempts": attempts,
    }
