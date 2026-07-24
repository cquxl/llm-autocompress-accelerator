from pathlib import Path

import yaml

from llm_autocompress.environment import _gpu_profile
from llm_autocompress.site import REPO_ROOT, load_site_config
from llm_autocompress.utils import source_tree_fingerprint


def test_unconfigured_host_uses_bundled_d2prune(tmp_path: Path):
    config = tmp_path / "site.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_root": str(tmp_path / "models"),
                "data_root": str(tmp_path / "data"),
            }
        ),
        encoding="utf-8",
    )
    site = load_site_config(config)
    assert site.d2prune_root == REPO_ROOT / "third_party" / "d2prune_core"
    assert (site.d2prune_root / "prune" / "wanda" / "wanda.py").is_file()
    assert (
        site.d2prune_root / "prune" / "sparsegpt" / "sparsegpt.py"
    ).is_file()


def test_site_config_expands_relative_paths_from_config_directory(tmp_path: Path):
    config = tmp_path / "config" / "site.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump(
            {
                "model_root": "../models",
                "data_root": "../data",
                "run_root": "../runs",
            }
        ),
        encoding="utf-8",
    )
    site = load_site_config(config)
    assert site.model_root == (tmp_path / "models").resolve()
    assert site.data_root == (tmp_path / "data").resolve()
    assert site.run_root == (tmp_path / "runs").resolve()


def test_a40_profile_selects_sm86_and_structured_sparsity():
    profile = _gpu_profile(
        [
            {
                "name": "NVIDIA A40",
                "compute_capability": "8.6",
                "memory_total_mb": 46068,
            }
        ]
    )
    assert profile["supports_2_4"] is True
    assert profile["supports_samoyeds_source_build"] is True
    assert "sm_86" in profile["note"]


def test_bundled_d2prune_has_stable_nonempty_source_fingerprint():
    source = REPO_ROOT / "third_party" / "d2prune_core"
    first = source_tree_fingerprint(source)
    second = source_tree_fingerprint(source)
    assert first == second
    assert first != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
