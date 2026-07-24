from pathlib import Path

from llm_autocompress.adapters.compression import prepare_spinfer_phase2_script


def test_spinfer_copy_disables_fake_sparsity(tmp_path: Path):
    output = tmp_path / "phase2.py"
    metadata = prepare_spinfer_phase2_script(output)
    text = output.read_text(encoding="utf-8")
    assert metadata["fake_sparsity"] is False
    assert "FAKE_SPARSITY = False" in text
    assert "FAKE_SPARSITY = True" not in text
