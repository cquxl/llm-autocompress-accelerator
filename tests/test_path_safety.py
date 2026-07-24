from pathlib import Path

import pytest

from llm_autocompress.executor import create_run_dir
from llm_autocompress.schema import SKILL_ROOT, load_request


def test_external_output_requires_explicit_opt_in(tmp_path: Path):
    request = load_request(SKILL_ROOT / "assets" / "requests" / "opt-125m.yaml")
    request.output_dir = str(tmp_path)
    with pytest.raises(ValueError, match="allow_external_output"):
        create_run_dir(request)
