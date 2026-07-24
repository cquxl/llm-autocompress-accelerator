import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

from llm_autocompress.webapp import Handler, Job


def test_web_health_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/api/health", timeout=5
        ) as response:
            value = json.loads(response.read())
        assert value["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()


def test_web_job_exposes_progress_fields():
    value = Job(id="demo").public()
    assert value["current_step"] == "等待调度"
    assert value["progress"] == {"current": 0, "total": 0}
    assert value["task_kind"] == "pending"


def test_web_job_tracks_generation_and_candidate_metrics():
    job = Job(id="demo")
    candidate = {
        "id": "dense__transformers",
        "algorithm": "dense",
        "structure": "dense",
        "backend": "transformers",
    }
    job.handle_event(
        {
            "type": "candidate_start",
            "index": 1,
            "total": 2,
            "candidate": candidate,
        }
    )
    job.handle_event(
        {
            "type": "generation_token",
            "text": "hello",
            "tokens": 2,
            "tokens_per_second": 12.5,
        }
    )
    job.handle_event(
        {
            "type": "candidate_complete",
            "index": 1,
            "total": 2,
            "candidate": candidate,
            "status": "completed",
            "metrics": {"decode_tokens_per_second": 100},
            "artifact": {"compression_ratio": 2.0},
        }
    )
    value = job.public()
    assert value["progress"] == {"current": 1, "total": 2}
    assert value["live_generation"]["text"] == "hello"
    assert value["candidate_results"][0]["metrics"]["decode_tokens_per_second"] == 100
    assert value["candidate_results"][0]["artifact"]["compression_ratio"] == 2.0
