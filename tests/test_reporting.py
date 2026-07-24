from pathlib import Path

from llm_autocompress.reporting import evaluate_run, render_report
from llm_autocompress.schema import SKILL_ROOT, load_request
from llm_autocompress.utils import write_json


def test_microbenchmark_is_never_recommended(tmp_path: Path):
    request = load_request(SKILL_ROOT / "assets" / "requests" / "opt-125m.yaml")
    plan = {
        "model": {
            "resolved_path": request.model.path,
            "family": "opt",
            "kind": "dense",
        },
        "selection_policy": {},
        "candidates": [
            {
                "id": "wanda_2_4__cusparselt",
                "method": "wanda_2_4",
                "algorithm": "wanda",
                "structure": "2:4",
                "backend": "cusparselt",
                "scope": "linear_and_checkpoint",
                "status": "planned",
                "runnable": True,
                "reasons": [],
                "artifact_dir": str(tmp_path / "artifacts" / "candidate"),
                "real_artifact": True,
                "notes": "",
            }
        ],
    }
    write_json(
        tmp_path / "benchmarks" / "dense__transformers.json",
        {
            "status": "completed",
            "quality": {"perplexity": 20.0},
            "records": [{"tpot_p50_ms": 10.0}],
        },
    )
    write_json(
        tmp_path / "benchmarks" / "wanda_2_4__cusparselt.json",
        {
            "status": "completed",
            "implementation": "samoyeds_cusparselt24_kernel",
            "kernel_source": "native/samoyeds_cusparselt/cusparselt24_mod.cu",
            "uses_torch_private_cslt": False,
            "records": [{"same_backend_speedup": 2.0}],
        },
    )
    evaluation = evaluate_run(tmp_path, request, plan)
    assert evaluation["recommended"] is None
    assert (
        evaluation["evaluated"][0]["kernel_implementation"]
        == "samoyeds_cusparselt24_kernel"
    )
    assert evaluation["evaluated"][0]["uses_torch_private_cslt"] is False
    report = render_report(request, plan, evaluation, {"gpus": []})
    assert "microbenchmarks" in report
    assert "samoyeds_cusparselt24_kernel" in report
