"""The identical-image diagnostic uses production request pins, without a GPU."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = runpy.run_path(str(ROOT / "ops/sglang-baseline-control.py"))


def test_control_preserves_campaign_settings_and_uses_identical_images(tmp_path):
    fields = json.loads(CONTROL["FIELDS"].read_text())
    trace = tmp_path / "trace.json"
    trace.write_bytes((ROOT / "fixtures/bench/sample_trace.json").read_bytes())
    request = CONTROL["prepare_request"](fields, trace)
    assert request["engines"]["candidates"] == [request["engines"]["baseline"]]
    assert (
        request["engines"]["baseline"]["image"]
        == fields["bench"]["baseline_engine_image_digest"]
    )
    assert request["model"] == fields["bench"]["model"]
    assert request["hardware"] == {"gpu_count": 4, "gpu_sku_expected": "RTX5090"}
    assert request["sla_bench"]["repetitions"] == 3
    assert request["correctness"] == fields["bench"]["correctness"]
    assert request["scoring_rule"] == fields["scoring_rule"]
