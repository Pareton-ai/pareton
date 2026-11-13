"""Launch settings a candidate must leave as the baseline has them.

A campaign pins the engine's launch flags, but a patch can still change what
those flags resolve to from inside the engine: a new default, or a value
filled in at startup. The Open Kernel Challenge judge reads the running
server's settings and fails a run whose denied settings differ from stock's.
The harness does the same against the round's own baseline: both engines are
started with identical flags, so any difference in a fixed setting comes from
the patch.

Only the settings behind the denied flags are compared. A value the server
resolves from other settings (``cuda_graph_config``, memory figures) may
differ, as it may at OKC.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Mapping

# OKC's ``launch_flags.denied`` (openkernelchallenge.dev/api/v1/serving, read
# 2026-11-13), by the reason OKC gives for each group.
OKC_SERVING_DENIED_FLAGS: dict[str, tuple[str, ...]] = {
    "changes the target model's precision": (
        "--dtype",
        "--quantization*",
        "--kv-cache-dtype",
        "--prefill-kv-cache-dequant-dtype",
        "--modelopt-*",
        "--quantize-and-serve",
        "--enable-tf32-matmul",
        "--enable-quant-communications",
        "--deepep-dispatcher-output-dtype",
        "--flashinfer-mxfp4-moe-precision",
        "--quantization-param-path",
        "--enable-w4a4-mxfp4-megamoe",
        "--speculative-enable-w4a4-mxfp4-megamoe",
        "--rl-quant-profile",
    ),
    "changes the model or makes speculation inexact": (
        "--model-path",
        "--revision",
        "--json-model-override-args",
        "--*lora*",
        "--speculative-accept-threshold-*",
        "--speculative-use-rejection-sampling",
        "--watermark-*",
        "--decrypted-config-file",
        "--custom-weight-loader",
        "--model-loader-extra-config",
        "--preferred-sampling-params",
        "--sampling-defaults",
        "--model-impl",
        "--tokenizer-path",
        "--checkpoint-engine-*",
        "--modelexpress-config",
        "--kt-*",
        "--speculative-draft-model-quantization",
        "--speculative-draft-kv-cache-dtype",
        "--speculative-draft-load-format",
        "--speculative-draft-model-revision",
        "--decrypted-draft-config-file",
        "--speculative-token-map",
        "--speculative-ngram-external-corpus-path",
        "--speculative-dspark-*-path",
    ),
    "truncates prompts": ("--context-length", "--allow-auto-truncate"),
    "keeps state outside the server or sends data out of the pod": (
        "--hicache-storage-*",
        "--*lmcache*",
        "--*flexkv*",
        "--*external-linker*",
        "--weight-cache-*",
        "--remote-instance-*",
        "--otlp-*",
        "--kv-events*",
        "--load-publish-endpoint",
        "--export-metrics-*",
        "--debug-tensor-dump-*",
        "--crash-dump-folder",
        "--log-requests*",
        "--forward-hooks",
        "--encoder-*urls",
        "--decoupled-spec-*endpoint*",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--nccl-init-addr",
        "--disaggregation-*",
        "--spec-trace-dir",
        "--file-storage-path",
    ),
    "is set by the judge": (
        "--config",
        "--host",
        "--port",
        "--max-running-requests",
        "--cuda-graph-max-bs-decode",
        "--cuda-graph-config",
        "--cuda-graph-bs-decode",
        "--speculative-draft-model-path",
        "--load-format",
        "--skip-tokenizer-init",
        "--enable-cache-report",
        "--incremental-streaming-output",
    ),
}

# Named like the d2 review policy that reviews the same rules: a campaign
# reviewed under ``okc_serving_v1`` has its rounds enforce this list too.
POLICIES: dict[str, dict[str, tuple[str, ...]]] = {
    "okc_serving_v1": OKC_SERVING_DENIED_FLAGS,
}

# Settings behind a denied flag whose reported value is not the flag's own.
# SGLang folds the CUDA-graph convenience flags and its own defaults into
# ``cuda_graph_config``, so what /server_info shows there is derived: OKC
# passes a run whose ``cuda_graph_config.decode.bs`` moved as a new default,
# and fails one whose ``cuda_graph_bs_decode`` did.
RESOLVED_SETTINGS: dict[str, tuple[str, ...]] = {
    "okc_serving_v1": ("cuda_graph_config",),
}

_MISSING = object()
_UNSTABLE = object()
_MAX_SHOWN = 120
_MAX_LISTED = 8
# The name a candidate's unreadable settings are reported under.
UNREADABLE = "/server_info"


def validate_policy(name: Any) -> str | None:
    """A known policy name, or None when the campaign fixes no settings."""
    if name is None:
        return None
    if not isinstance(name, str) or name not in POLICIES:
        raise ValueError(
            f"fixed_launch_settings must be one of {sorted(POLICIES)}, got {name!r}"
        )
    return name


def _setting_pattern(flag: str) -> str:
    return flag.removeprefix("--").replace("-", "_")


def fixed_settings(server_info: Mapping[str, Any], policy: str) -> dict[str, Any]:
    """The settings of one running server that the policy fixes.

    ``server_info`` is SGLang's ``/server_info``: the resolved server
    arguments by flag name. A setting is kept whole, so one that is itself a
    mapping (``preferred_sampling_params``) differs when any key under it
    does. A mapping no pattern matches is searched by dotted name. A setting
    in ``RESOLVED_SETTINGS`` is left out with everything under it.
    """
    patterns = [
        _setting_pattern(flag) for flags in POLICIES[policy].values() for flag in flags
    ]
    resolved = RESOLVED_SETTINGS.get(policy, ())
    kept: dict[str, Any] = {}

    def walk(value: Mapping[str, Any], prefix: str) -> None:
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if name in resolved:
                continue
            if any(fnmatchcase(name, pattern) for pattern in patterns):
                kept[name] = item
            elif isinstance(item, Mapping):
                walk(item, name)

    walk(server_info, "")
    return kept


def stable_settings(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict:
    """The reference two baseline runs give: the fixed settings they agree on.

    A setting that differs between two starts of the same stock engine says
    nothing about a patch, so it is marked and never compared.
    """
    return {
        name: first[name]
        if name in first and name in second and first[name] == second[name]
        else _UNSTABLE
        for name in set(first) | set(second)
    }


def _shown(value: Any) -> str:
    if value is _MISSING:
        return "<absent>"
    text = repr(value)
    return text if len(text) <= _MAX_SHOWN else text[: _MAX_SHOWN - 3] + "..."


def changed_settings(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Fixed settings whose value on the candidate differs from the baseline's.

    A setting only the candidate reports counts as changed: the patch added a
    setting of a fixed kind. A setting the baseline runs disagreed on
    (``stable_settings``) is skipped.
    """
    changed: dict[str, dict[str, str]] = {}
    for name in sorted(set(baseline) | set(candidate)):
        before = baseline.get(name, _MISSING)
        after = candidate.get(name, _MISSING)
        if before is _UNSTABLE:
            continue
        if before != after:
            changed[name] = {"baseline": _shown(before), "candidate": _shown(after)}
    return changed


def unreadable_settings(error: str) -> dict[str, dict[str, str]]:
    """What ``changed_settings`` would say of a server that reports nothing.

    The baseline answered ``/server_info`` in the same round, so a candidate
    that serves but cannot be asked its settings is the patch's doing, and no
    setting it runs with can be checked.
    """
    return {UNREADABLE: {"baseline": "readable", "candidate": error}}


def denied_reason(changed: Mapping[str, Mapping[str, str]]) -> str:
    """The disqualification reason for a candidate with changed settings."""
    parts = [
        f"{name}: {values['baseline']} -> {values['candidate']}"
        for name, values in list(changed.items())[:_MAX_LISTED]
    ]
    if len(changed) > _MAX_LISTED:
        parts.append(f"and {len(changed) - _MAX_LISTED} more")
    return (
        "launch_setting_denied: the server runs with launch settings the "
        "campaign fixes changed from the baseline's (" + "; ".join(parts) + ")"
    )
