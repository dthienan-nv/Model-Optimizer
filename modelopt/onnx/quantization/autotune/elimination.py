# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backward-elimination Q/DQ search (coordinate descent from a fully quantized start).

The forward per-region search in :mod:`workflows` profiles each region mostly in
isolation: while one region's schemes are measured, other regions carry either their
committed best (often empty) or nothing. Q/DQ gains in CNNs are dominated by
*cross-region* boundary cancellation — adjacent int8 regions eliminate each other's
Q/DQ conversions — so single-region forward moves each look latency-neutral and the
search collapses to a near-FP16 local optimum, even when a fully quantized network
is far faster (observed on vovnet: full int8 30 ms vs forward-search result 51 ms
vs FP16 baseline 52.5 ms).

This module searches in the opposite direction with a *stationary* objective:

1. Start from a fully quantized configuration — either the Q/DQ placement of a
   pre-quantized baseline model (the heuristic quantizer's output) or the union of
   all regions' full insertion schemes.
2. Measure the whole model end-to-end. Every subsequent measurement is also a whole
   model with the *current global configuration*, so cross-region cancellation is
   always reflected.
3. Coordinate descent: for each region-group of insertion points, try removing the
   whole group; keep the removal only if it improves latency by more than the noise
   floor. Repeat passes until a fixpoint.

The starting configuration is itself a measured candidate, so the result is
guaranteed to be no slower than the imported baseline placement (within noise).
"""

import argparse
import json
import shlex
from pathlib import Path

import onnx

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.autotuner import QDQAutotuner
from modelopt.onnx.quantization.autotune.common import Config, Region
from modelopt.onnx.quantization.autotune.export_utils import export_qdq_onnx
from modelopt.onnx.quantization.autotune.insertion_points import ResolvedInsertionPoint
from modelopt.onnx.quantization.autotune.region_pattern import RegionPattern
from modelopt.onnx.quantization.autotune.workflows import (
    benchmark_onnx_model,
    init_benchmark_instance,
)
from modelopt.onnx.quantization.qdq_utils import get_quantized_tensors

__all__ = ["run_backward_elimination"]


def _region_label(region: Region) -> str:
    return f"region_{region.id}_L{region.level}_{region.type.value}"


def _normalize_seed_tensors(autotuner: QDQAutotuner, qdq_model: onnx.ModelProto) -> set[str]:
    """Map a Q/DQ baseline's quantized-tensor names onto the FP16 graph's tensor names.

    ``get_quantized_tensors`` returns each DequantizeLinear input, which in QDQ models
    is usually the Q node output named ``<orig>_QuantizeLinear_Output``; the FP16 graph
    only knows ``<orig>``. Try the raw name first, then the stripped one.
    """
    graph_tensors: set[str] = set(autotuner.graph.tensor_users_map)
    for node in autotuner.graph.nodes:
        graph_tensors.update(t.name for t in node.inputs if t.name)
        graph_tensors.update(t.name for t in node.outputs if t.name)

    raw = get_quantized_tensors(qdq_model)
    mapped: set[str] = set()
    unmatched: list[str] = []
    for name in raw:
        if name in graph_tensors:
            mapped.add(name)
        elif name.removesuffix("_QuantizeLinear_Output") in graph_tensors:
            mapped.add(name.removesuffix("_QuantizeLinear_Output"))
        else:
            unmatched.append(name)
    logger.info(
        f"Seed tensor mapping: {len(mapped)}/{len(raw)} matched the FP16 graph"
        + (f"; unmatched e.g. {unmatched[:3]}" if unmatched else "")
    )
    return mapped


def _group_seed_tensors_by_region(
    autotuner: QDQAutotuner, seed_tensors: set[str]
) -> dict[str, set[ResolvedInsertionPoint]]:
    """Group tensor-level insertion points from a Q/DQ baseline by owning region.

    A tensor belongs to the most specific (lowest level) region that contains any
    of its consumer nodes. Tensors without a consumer in any region fall into the
    ``unassigned`` group.
    """
    node_to_region: dict[int, Region] = {}
    for region in sorted(autotuner.regions, key=lambda r: r.level):
        for node_idx in region.nodes:
            node_to_region.setdefault(node_idx, region)

    tensor_users_map = autotuner.graph.tensor_users_map
    groups: dict[str, set[ResolvedInsertionPoint]] = {}
    matched = 0
    for tensor in sorted(seed_tensors):
        consumers = tensor_users_map.get(tensor, [])
        region = next(
            (node_to_region[c] for c in consumers if c in node_to_region),
            None,
        )
        label = _region_label(region) if region is not None else "unassigned"
        groups.setdefault(label, set()).add(ResolvedInsertionPoint(tensor_name=tensor))
        if region is not None:
            matched += 1
    logger.info(
        f"Grouped {len(seed_tensors)} seed tensors into {len(groups)} region groups "
        f"({matched} tensors matched a region)"
    )
    return groups


def _full_quantization_groups(
    autotuner: QDQAutotuner,
) -> dict[str, set[ResolvedInsertionPoint]]:
    """Build the fully quantized starting point from every region's full insertion scheme."""
    groups: dict[str, set[ResolvedInsertionPoint]] = {}
    for region in autotuner.regions:
        pattern = RegionPattern.from_region(region, autotuner.graph)
        full_scheme = pattern.get_full_insertion_scheme(region, autotuner.graph)
        points = pattern.matches(region, autotuner.graph, full_scheme)
        if points:
            groups[_region_label(region)] = set(points)
    total = sum(len(p) for p in groups.values())
    logger.info(
        f"Built full-quantization seed: {total} insertion points across {len(groups)} regions"
    )
    return groups


def run_backward_elimination(
    model_or_path: str | onnx.ModelProto,
    qdq_baseline_path: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    epsilon_ms: float = 0.1,
    max_passes: int = 3,
    quant_type: str = "int8",
    default_dq_dtype: str = "float16",
    benchmark_fn=None,
    verbose: bool = False,
) -> dict:
    """Run backward-elimination Q/DQ search and return a result summary dict.

    Args:
        model_or_path: FP16/FP32 ONNX model (no Q/DQ) to optimize.
        qdq_baseline_path: Optional pre-quantized model whose Q/DQ placement seeds the
            search. If None, starts from full quantization of every region.
        output_path: Where to save the best model (default: <output_dir>/eliminated.onnx).
        output_dir: Directory for logs/report (default: alongside output_path or cwd).
        epsilon_ms: Noise floor; a removal is accepted only if it improves the current
            best latency by more than this.
        max_passes: Maximum coordinate-descent passes over all groups.
        quant_type: "int8" (default) or "fp8".
        default_dq_dtype: DequantizeLinear output dtype ("float16" default).
        benchmark_fn: Callable(model_bytes) -> latency_ms. Defaults to the global
            benchmark instance (init_benchmark_instance must have been called).
        verbose: Verbose autotuner config logging.

    Returns:
        Dict with baseline/seed/final latencies, kept/removed groups, and step log.
    """
    model = onnx.load(model_or_path) if isinstance(model_or_path, str) else model_or_path

    out_dir = Path(output_dir) if output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = str(out_dir / "eliminated.onnx")

    config = Config(
        default_quant_type=quant_type, default_dq_dtype=default_dq_dtype, verbose=verbose
    )
    autotuner = QDQAutotuner(model)
    autotuner.initialize(config)
    base_model = autotuner.onnx_model

    if benchmark_fn is None:
        benchmark_fn = benchmark_onnx_model

    def measure(points: set[ResolvedInsertionPoint], tag: str) -> float:
        exported = export_qdq_onnx(
            base_model, points, config, insert_qdq=bool(points)
        )
        latency = benchmark_fn(exported.SerializeToString())
        logger.info(f"[eliminate] {tag}: {len(points)} Q/DQ points → {latency:.3f} ms")
        return latency

    # 1. Starting configuration. Recommended: seed with the heuristic quantizer's
    #    placement (a plain PTQ model via qdq_baseline_path); full quantization of
    #    every region is the fallback when no baseline exists.
    if qdq_baseline_path:
        seed_tensors = _normalize_seed_tensors(autotuner, onnx.load(qdq_baseline_path))
        groups = _group_seed_tensors_by_region(autotuner, seed_tensors)
    else:
        groups = _full_quantization_groups(autotuner)
    if not groups:
        raise ValueError("No insertion points found to start elimination from")

    active = set(groups)

    def union(labels: set[str]) -> set[ResolvedInsertionPoint]:
        pts: set[ResolvedInsertionPoint] = set()
        for lb in labels:
            pts |= groups[lb]
        return pts

    # 2. Reference measurements (both are real candidates)
    steps: list[dict] = []
    baseline_latency = measure(set(), "fp16 baseline (no Q/DQ)")
    best_latency = measure(union(active), "seed (fully quantized start)")
    best_active = set(active)
    steps.append({"config": "baseline", "latency_ms": baseline_latency})
    steps.append({"config": "seed", "groups": len(active), "latency_ms": best_latency})

    # 3. Coordinate descent: remove one group at a time, keep real improvements.
    for pass_idx in range(max_passes):
        improved = False
        for label in sorted(active, key=lambda lb: -len(groups[lb])):
            candidate = active - {label}
            latency = measure(union(candidate), f"pass{pass_idx + 1} -{label}")
            accepted = latency < best_latency - epsilon_ms
            steps.append(
                {
                    "pass": pass_idx + 1,
                    "removed": label,
                    "points": len(groups[label]),
                    "latency_ms": latency,
                    "accepted": accepted,
                }
            )
            if accepted:
                active = candidate
                best_latency = latency
                best_active = set(active)
                improved = True
                logger.info(
                    f"[eliminate] accepted removal of {label} "
                    f"(new best {best_latency:.3f} ms, {len(active)} groups left)"
                )
        if not improved:
            logger.info(f"[eliminate] converged after pass {pass_idx + 1}")
            break

    # 4. Export best configuration (fall back to baseline if quantization never helped)
    final_points = union(best_active)
    if baseline_latency < best_latency - epsilon_ms:
        logger.warning(
            f"[eliminate] no quantized configuration beat the FP16 baseline "
            f"({baseline_latency:.3f} ms); exporting the unquantized model"
        )
        final_points = set()
        best_latency = baseline_latency
        best_active = set()

    final_model = export_qdq_onnx(base_model, final_points, config, insert_qdq=bool(final_points))
    onnx.save(final_model, output_path)

    result = {
        "baseline_latency_ms": baseline_latency,
        "seed_groups": len(groups),
        "final_latency_ms": best_latency,
        "final_groups": sorted(best_active),
        "removed_groups": sorted(set(groups) - best_active),
        "final_points": len(final_points),
        "output_path": output_path,
        "steps": steps,
    }
    report_path = out_dir / "elimination_report.json"
    report_path.write_text(json.dumps(result, indent=2))
    logger.info(
        f"[eliminate] done: baseline {baseline_latency:.3f} ms → final {best_latency:.3f} ms "
        f"({len(final_points)} Q/DQ points, {len(best_active)}/{len(groups)} groups kept); "
        f"report → {report_path}"
    )
    return result


def _make_dry_run_benchmark():
    """Deterministic offline stand-in for a device benchmark (mechanics testing only).

    Latency decreases monotonically with the number of Q/DQ nodes in the exported
    model, so elimination should keep every group. Counts Q nodes by parsing the
    exported bytes — exercising the real export path.
    """

    def fake_benchmark(model_bytes: bytes) -> float:
        m = onnx.load_from_string(model_bytes)
        n_q = sum(1 for n in m.graph.node if n.op_type == "QuantizeLinear")
        return max(1.0, 50.0 - float(n_q))

    return fake_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Backward-elimination Q/DQ search (whole-model, stationary objective)"
    )
    parser.add_argument("--onnx_path", required=True, help="FP16/FP32 model without Q/DQ")
    parser.add_argument(
        "--qdq_baseline",
        default=None,
        help="Pre-quantized model whose Q/DQ placement seeds the search "
        "(default: full quantization of every region)",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--epsilon_ms", type=float, default=0.1)
    parser.add_argument("--max_passes", type=int, default=3)
    parser.add_argument("--quant_type", default="int8", choices=["int8", "fp8"])
    parser.add_argument("--default_dq_dtype", default="float16")
    parser.add_argument("--timing_cache", default=None)
    parser.add_argument("--warmup_runs", type=int, default=5)
    parser.add_argument("--timing_runs", type=int, default=20)
    parser.add_argument(
        "--trtexec_args",
        default=None,
        help="Extra trtexec args as one string, e.g. "
        "'--remoteAutoTuningConfig=ssh://... --safe --skipInference --cpuOnly'",
    )
    parser.add_argument("--remote_model_path", default="/tmp/trtexec_benchmark_model.trt")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a deterministic fake benchmark (no device); tests export mechanics",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    benchmark_fn = None
    if args.dry_run:
        benchmark_fn = _make_dry_run_benchmark()
    else:
        init_benchmark_instance(
            use_trtexec=True,
            timing_cache_file=args.timing_cache,
            warmup_runs=args.warmup_runs,
            timing_runs=args.timing_runs,
            trtexec_args=shlex.split(args.trtexec_args) if args.trtexec_args else None,
            remote_model_path=args.remote_model_path,
        )

    run_backward_elimination(
        args.onnx_path,
        qdq_baseline_path=args.qdq_baseline,
        output_path=args.output_path,
        output_dir=args.output_dir,
        epsilon_ms=args.epsilon_ms,
        max_passes=args.max_passes,
        quant_type=args.quant_type,
        default_dq_dtype=args.default_dq_dtype,
        benchmark_fn=benchmark_fn,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
