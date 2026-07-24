from __future__ import annotations

import json
import mimetypes
import re
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .business import apply_web_preset, request_mapping_from_business
from .executor import run_request
from .schema import SKILL_ROOT, load_request, request_from_mapping
from .utils import utc_now


@dataclass
class Job:
    id: str
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    run_dir: str | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    current_step: str = "等待调度"
    progress_current: int = 0
    progress_total: int = 0
    current_candidate: dict[str, Any] | None = None
    live_generation: dict[str, Any] = field(default_factory=dict)
    candidate_results: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    task_kind: str = "pending"
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "phase":
            self.current_step = str(event.get("message") or event.get("phase"))
        elif kind == "plan_ready":
            self.run_dir = str(event["run_dir"])
            candidates = event.get("candidates") or []
            self.progress_total = len(candidates)
            self.task_kind = (
                "automatic_compression"
                if any(item.get("method") != "dense" for item in candidates)
                else "dense_baseline_validation"
            )
            self.current_step = "能力规划完成，准备执行候选"
        elif kind == "candidate_start":
            candidate = event["candidate"]
            self.current_candidate = candidate
            self.progress_current = max(int(event["index"]) - 1, 0)
            self.progress_total = int(event["total"])
            self.current_step = (
                f"{candidate['algorithm']} / {candidate['structure']} → "
                f"{candidate['backend']}"
            )
            self.live_generation = {}
        elif kind == "generation_start":
            self.live_generation = {
                "status": "streaming",
                "backend": event.get("backend"),
                "prompt": event.get("prompt"),
                "text": "",
                "tokens": 0,
                "tokens_per_second": 0,
            }
            self.current_step = f"{event.get('backend')} 正在生成文本"
        elif kind == "generation_token":
            self.live_generation.update(
                {
                    "status": "streaming",
                    "text": event.get("text", ""),
                    "tokens": event.get("tokens", 0),
                    "tokens_per_second": event.get("tokens_per_second", 0),
                    "elapsed_seconds": event.get("elapsed_seconds", 0),
                }
            )
        elif kind == "generation_complete":
            self.live_generation.update(
                {
                    "status": "completed",
                    "backend": event.get("backend"),
                    "prompt": event.get("prompt"),
                    "text": event.get("text", ""),
                    "tokens": event.get("tokens", 0),
                    "tokens_per_second": event.get("tokens_per_second", 0),
                    "elapsed_seconds": event.get("elapsed_seconds", 0),
                }
            )
        elif kind == "candidate_complete":
            self.progress_current = int(event["index"])
            self.progress_total = int(event["total"])
            self.candidate_results.append(
                {
                    key: event.get(key)
                    for key in (
                        "candidate",
                        "status",
                        "metrics",
                        "quality",
                        "generation_sample",
                        "artifact",
                        "error",
                        "reasons",
                    )
                }
            )
        elif kind == "report_ready":
            self.run_dir = str(event["run_dir"])
            self.evaluation = event.get("evaluation")
            self.current_step = "报告已生成"

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "run_dir": self.run_dir,
            "error": self.error,
            "logs": self.logs[-500:],
            "current_step": self.current_step,
            "progress": {
                "current": self.progress_current,
                "total": self.progress_total,
            },
            "current_candidate": self.current_candidate,
            "live_generation": self.live_generation,
            "candidate_results": self.candidate_results,
            "evaluation": self.evaluation,
            "task_kind": self.task_kind,
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        # Conversion and inference jobs share CUDA state and local artifact
        # repositories. Run one job at a time unless a future scheduler assigns
        # isolated devices explicitly.
        self.execution_lock = threading.Lock()

    def create(self, request, mode: str, yes: bool) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        with self.lock:
            self.jobs[job.id] = job
        job.logs.append(f"[{utc_now()}] 任务已创建，等待 GPU 执行队列")

        def worker() -> None:
            with self.execution_lock:
                if job.cancel.is_set():
                    job.status = "cancelled"
                    job.current_step = "已取消"
                    job.finished_at = utc_now()
                    return
                job.status = "running"
                job.current_step = "检查环境、模型与后端"

                def log(message: str) -> None:
                    job.logs.append(message)
                    if "run directory: " in message:
                        job.run_dir = message.split("run directory: ", 1)[1].strip()
                    match = re.search(r"candidate \[(\d+)/(\d+)\] ([^:]+)", message)
                    if match:
                        job.progress_current = max(int(match.group(1)) - 1, 0)
                        job.progress_total = int(match.group(2))
                        job.current_step = f"执行候选 {match.group(3)}"
                    if "generating evaluation and report" in message:
                        job.progress_current = job.progress_total
                        job.current_step = "生成评测与报告"

                try:
                    target = run_request(
                        request,
                        mode=mode,
                        yes=yes,
                        cancel_event=job.cancel,
                        logger=log,
                        event_handler=job.handle_event,
                    )
                    job.run_dir = str(target)
                    job.progress_current = job.progress_total
                    job.status = "cancelled" if job.cancel.is_set() else "completed"
                    job.current_step = "已取消" if job.cancel.is_set() else "已完成"
                except Exception as exc:
                    job.status = "failed"
                    job.current_step = "执行失败"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.logs.append(traceback.format_exc())
                job.finished_at = utc_now()

        threading.Thread(target=worker, daemon=True, name=f"llm-job-{job.id}").start()
        return job


MANAGER = JobManager()
WEB_ROOT = SKILL_ROOT / "assets" / "web"


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMAutoCompress/0.1"

    def _json(self, value: Any, status: int = 200) -> None:
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size > 1_000_000:
            raise ValueError("request body exceeds 1 MB")
        return json.loads(self.rfile.read(size) or b"{}")

    def _job(self, job_id: str) -> Job | None:
        with MANAGER.lock:
            return MANAGER.jobs.get(job_id)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._json({"status": "ok", "time": utc_now()})
            return
        if path == "/api/jobs":
            with MANAGER.lock:
                values = [job.public() for job in MANAGER.jobs.values()]
            self._json({"jobs": values})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.split("/")[3]
            job = self._job(job_id)
            if not job:
                self._json({"error": "job not found"}, 404)
            else:
                self._json(job.public())
            return
        if path.startswith("/api/artifacts/"):
            parts = path.split("/", 4)
            if len(parts) != 5:
                self._json({"error": "artifact path required"}, 400)
                return
            job = self._job(parts[3])
            if not job or not job.run_dir:
                self._json({"error": "job artifacts unavailable"}, 404)
                return
            root = Path(job.run_dir).resolve()
            target = (root / unquote(parts[4])).resolve()
            if root not in target.parents and target != root:
                self._json({"error": "unsafe artifact path"}, 400)
                return
            if not target.is_file():
                self._json({"error": "artifact not found"}, 404)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(target.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        target = WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
        target = target.resolve()
        if WEB_ROOT.resolve() not in target.parents or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            mimetypes.guess_type(target.name)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            body = self._body()
            if self.path == "/api/bootstrap":
                mapping = request_mapping_from_business(
                    model=str(body.get("model", "opt-125m")),
                    prompt=str(body.get("prompt", "")),
                    profile=str(body.get("profile", "interactive")),
                )
                mapping = apply_web_preset(
                    mapping,
                    str(body.get("preset", "auto-full")),
                    pruning_granularity=str(
                        body.get("pruning_granularity", "2:4")
                    ),
                    profile=str(body.get("profile", "interactive")),
                )
                mapping["calibration"].update(samples=8, sequence_length=128)
                mapping["evaluation"]["max_tokens_smoke"] = 512
                mapping["search"].update(
                    target_checkpoint_ratio=float(
                        body.get("target_checkpoint_ratio", 2.0)
                    ),
                    pruning_granularity=str(
                        body.get("pruning_granularity", "2:4")
                    ),
                )
                mapping["constraints"].update(
                    max_relative_ppl_increase=float(
                        body.get("max_relative_ppl_increase", 0.05)
                    ),
                    min_same_backend_speedup=float(
                        body.get("min_same_backend_speedup", 1.0)
                    ),
                )
                request_from_mapping(mapping)
                self._json({"request": mapping})
                return
            if self.path == "/api/jobs":
                if body.get("request_path"):
                    request = load_request(body["request_path"])
                elif isinstance(body.get("request"), dict):
                    request = request_from_mapping(body["request"])
                else:
                    mapping = request_mapping_from_business(
                        model=str(body.get("model", "opt-125m")),
                        prompt=str(body.get("prompt", "")),
                        profile=str(body.get("profile", "interactive")),
                    )
                    mapping = apply_web_preset(
                        mapping,
                        str(body.get("preset", "auto-full")),
                        pruning_granularity=str(
                            body.get("pruning_granularity", "2:4")
                        ),
                        profile=str(body.get("profile", "interactive")),
                    )
                    if str(body.get("mode", "smoke")) == "smoke":
                        mapping["calibration"].update(
                            samples=8,
                            sequence_length=128,
                        )
                        mapping["evaluation"]["max_tokens_smoke"] = 512
                    mapping["search"].update(
                        target_checkpoint_ratio=float(
                            body.get("target_checkpoint_ratio", 2.0)
                        ),
                        pruning_granularity=str(
                            body.get("pruning_granularity", "2:4")
                        ),
                    )
                    mapping["constraints"].update(
                        max_relative_ppl_increase=float(
                            body.get("max_relative_ppl_increase", 0.05)
                        ),
                        min_same_backend_speedup=float(
                            body.get("min_same_backend_speedup", 1.0)
                        ),
                    )
                    request = request_from_mapping(mapping)
                job = MANAGER.create(
                    request,
                    mode=str(body.get("mode", "smoke")),
                    yes=bool(body.get("yes", False)),
                )
                self._json(job.public(), 202)
                return
            if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                job_id = self.path.split("/")[3]
                job = self._job(job_id)
                if not job:
                    self._json({"error": "job not found"}, 404)
                else:
                    job.cancel.set()
                    job.logs.append(f"[{utc_now()}] cancellation requested")
                    self._json(job.public())
                return
            self._json({"error": "unknown endpoint"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} {format % args}")


def serve(host: str = "127.0.0.1", port: int = 7860) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "Warning: non-localhost binding exposes model paths and benchmark logs; "
            "use a trusted network only."
        )
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"LLM AutoCompress workbench: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
