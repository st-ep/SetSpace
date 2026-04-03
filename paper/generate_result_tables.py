#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_PATHS = {
    "regression": REPO_ROOT / "results" / "point_cloud_mean_regression_run" / "metrics.json",
    "ahmed": REPO_ROOT / "results" / "ahmedml_surface_forces_run" / "metrics.json",
    "airfrans": REPO_ROOT / "results" / "airfrans_field_prediction_run" / "metrics.json",
    "regression_pointnext": REPO_ROOT / "results" / "point_cloud_mean_regression_pointnext_run" / "metrics.json",
    "ahmed_pointnext": REPO_ROOT / "results" / "ahmedml_surface_forces_pointnext_run" / "metrics.json",
    "airfrans_pointnext": REPO_ROOT / "results" / "airfrans_field_prediction_pointnext_run" / "metrics.json",
}
MODEL_ORDER = [
    "uniform",
    "geometry_aware",
]
MODEL_LABELS = {
    "uniform": "Uniform",
    "geometry_aware": "kNN",
}
POINTNEXT_ROW = "pointnext_uniform"
POINTNEXT_LABEL = "PointNeXt"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    kind: str
    default_path: Path
    header: str


TASK_SPECS = [
    TaskSpec(
        "regression",
        "Synthetic Regr.",
        "regression",
        DEFAULT_METRICS_PATHS["regression"],
        r"\shortstack{Synthetic\\Regr.}",
    ),
    TaskSpec(
        "ahmed",
        "AhmedML",
        "regression",
        DEFAULT_METRICS_PATHS["ahmed"],
        "AhmedML",
    ),
    TaskSpec(
        "airfrans",
        "AirfRANS",
        "regression",
        DEFAULT_METRICS_PATHS["airfrans"],
        "AirfRANS",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX result tables from benchmark metrics.")
    parser.add_argument("--regression_metrics", default=str(DEFAULT_METRICS_PATHS["regression"]))
    parser.add_argument("--ahmed_metrics", default=str(DEFAULT_METRICS_PATHS["ahmed"]))
    parser.add_argument("--airfrans_metrics", default=str(DEFAULT_METRICS_PATHS["airfrans"]))
    parser.add_argument("--regression_pointnext_metrics", default=str(DEFAULT_METRICS_PATHS["regression_pointnext"]))
    parser.add_argument("--ahmed_pointnext_metrics", default=str(DEFAULT_METRICS_PATHS["ahmed_pointnext"]))
    parser.add_argument("--airfrans_pointnext_metrics", default=str(DEFAULT_METRICS_PATHS["airfrans_pointnext"]))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "paper" / "tables"))
    parser.add_argument(
        "--breakdown_source",
        choices=["regression", "ahmed"],
        default="ahmed",
    )
    parser.add_argument(
        "--resolution_source",
        choices=["regression", "ahmed"],
        default="ahmed",
    )
    parser.add_argument("--breakdown_point", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def get_model_metrics(payload: dict[str, Any] | None, model_name: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    return payload.get("models", {}).get(model_name, {}).get("metrics")


def infer_uniform_key(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "uniform"
    modes = payload.get("sampling_modes", [])
    if "uniform" in modes:
        return "uniform"
    if "uniform_object" in modes:
        return "uniform_object"
    return modes[0] if modes else "uniform"


def point_counts(payload: dict[str, Any] | None) -> list[int]:
    if payload is None:
        return []
    return [int(v) for v in payload.get("point_counts", [])]


def closest_point_key(payload: dict[str, Any] | None, preferred: int | None) -> str | None:
    counts = point_counts(payload)
    if not counts:
        return None
    if preferred is None:
        preferred = payload.get("training", {}).get("train_points")
    if preferred is None:
        preferred = counts[0]
    target = min(counts, key=lambda c: (abs(c - int(preferred)), c))
    return str(target)


def avg_nonuniform_from_per_setting(
    payload: dict[str, Any] | None,
    model_name: str,
    metric_name: str,
    point_key: str | None,
) -> float | None:
    if payload is None or point_key is None:
        return None
    metrics = get_model_metrics(payload, model_name)
    if metrics is None:
        return None
    per_setting = metrics.get("per_setting", {})
    uniform_key = infer_uniform_key(payload)
    modes = [mode for mode in payload.get("sampling_modes", []) if mode != uniform_key]
    if not modes:
        modes = [uniform_key]
    values = []
    for mode in modes:
        try:
            values.append(float(per_setting[mode][point_key][metric_name]))
        except KeyError:
            return None
    return sum(values) / len(values) if values else None


def first_available_model(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    for model_name in MODEL_ORDER:
        if model_name in payload.get("models", {}):
            return model_name
    models = list(payload.get("models", {}).keys())
    return models[0] if models else None


def metric_direction(metric_id: str) -> bool:
    return metric_id in {"id_acc", "worst_acc", "pred_cons", "mode_metric_cls"}


def format_metric(value: float | None, metric_id: str) -> str:
    if value is None:
        return "--"
    if metric_id in {"id_acc", "worst_acc", "pred_cons", "mode_metric_cls"}:
        return f"{100.0 * value:.1f}"
    return f"{value:.3f}"


def wrap_best(text: str, is_best: bool) -> str:
    if text == "--" or not is_best:
        return text
    return rf"\textbf{{{text}}}"


def compute_best_mask(values: list[float | None], higher_is_better: bool) -> list[bool]:
    valid = [v for v in values if v is not None]
    if not valid:
        return [False] * len(values)
    best = max(valid) if higher_is_better else min(valid)
    return [v is not None and abs(v - best) <= 1e-12 for v in values]


def summarize_main_task(payload: dict[str, Any] | None, task_kind: str, model_name: str) -> dict[str, float | None]:
    point_key = closest_point_key(payload, preferred=None)
    if payload is None or point_key is None:
        return {"id": None, "worst": None, "consistency": None}
    metrics = get_model_metrics(payload, model_name)
    if metrics is None:
        return {"id": None, "worst": None, "consistency": None}

    aggregate = metrics["aggregate"]
    uniform_key = infer_uniform_key(payload)
    if task_kind == "classification":
        return {
            "id": float(aggregate["accuracy_by_count"][point_key][uniform_key]),
            "worst": float(aggregate["worst_case_accuracy"][point_key]),
            "consistency": avg_nonuniform_from_per_setting(payload, model_name, "prediction_consistency", point_key),
        }
    return {
        "id": float(aggregate["rmse_by_count"][point_key][uniform_key]),
        "worst": float(aggregate["worst_case_rmse"][point_key]),
        "consistency": float(aggregate["avg_nonuniform_prediction_drift"][point_key]),
    }

def regression_table_rows(pointnext_payload: dict[str, Any] | None = None) -> list[tuple[str, str, dict[str, Any] | None]]:
    rows = [(model_name, MODEL_LABELS[model_name], None) for model_name in MODEL_ORDER]
    if pointnext_payload is not None:
        rows.append((POINTNEXT_ROW, POINTNEXT_LABEL, pointnext_payload))
    return rows


def render_main_results_table(
    task_payloads: dict[str, dict[str, Any] | None],
    pointnext_payloads: dict[str, dict[str, Any] | None],
) -> str:
    row_specs = regression_table_rows(
        pointnext_payload=pointnext_payloads.get("regression") or pointnext_payloads.get("ahmed")
    )
    rows: dict[str, list[tuple[str, float | None]]] = {row_name: [] for row_name, _label, _payload in row_specs}
    header_cells = []
    metric_ids: list[str] = []

    for spec in TASK_SPECS:
        header_cells.append(rf"\multicolumn{{3}}{{c}}{{{spec.label}}}")
        if spec.kind == "classification":
            metric_triplet = [
                ("ID Acc. $\\uparrow$", "id_acc"),
                ("Worst Acc. $\\uparrow$", "worst_acc"),
                ("Pred. Cons. $\\uparrow$", "pred_cons"),
            ]
        else:
            metric_triplet = [
                ("ID RMSE $\\downarrow$", "id_rmse"),
                ("Worst RMSE $\\downarrow$", "worst_rmse"),
                ("Pred. Drift $\\downarrow$", "pred_drift"),
            ]
        for row_name, _row_label, maybe_payload in row_specs:
            payload = task_payloads.get(spec.key) if row_name != POINTNEXT_ROW else pointnext_payloads.get(spec.key)
            model_name = row_name if row_name != POINTNEXT_ROW else "pointnext"
            summary = summarize_main_task(payload, spec.kind, model_name)
            rows[row_name].extend(
                [
                    (metric_triplet[0][1], summary["id"]),
                    (metric_triplet[1][1], summary["worst"]),
                    (metric_triplet[2][1], summary["consistency"]),
                ]
            )
        metric_ids.extend(metric_id for _label, metric_id in metric_triplet)

    best_masks_by_col: list[list[bool]] = []
    for col_idx, metric_id in enumerate(metric_ids):
        values = [rows[row_name][col_idx][1] for row_name, _row_label, _payload in row_specs]
        best_masks_by_col.append(compute_best_mask(values, higher_is_better=metric_direction(metric_id)))

    body_lines = []
    for model_idx, (row_name, row_label, _payload) in enumerate(row_specs):
        cells = [row_label]
        for col_idx, (metric_id, value) in enumerate(rows[row_name]):
            formatted = format_metric(value, metric_id)
            cells.append(wrap_best(formatted, best_masks_by_col[col_idx][model_idx]))
        body_lines.append(" & ".join(cells) + r" \\")

    second_header = ["Model"]
    for spec in TASK_SPECS:
        if spec.kind == "classification":
            second_header.extend([r"\shortstack{ID\\Acc.}", "Worst", r"\shortstack{Pred.\\Cons.}"])
        else:
            second_header.extend([r"\shortstack{In-dist.\\RMSE}", r"\shortstack{Worst\\RMSE}", r"\shortstack{Pred.\\Drift}"])

    lines = [
        "% Auto-generated by paper/generate_result_tables.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{'c' * (3 * len(TASK_SPECS))}}}",
        r"\hline",
        " & ".join([""] + [rf"\multicolumn{{3}}{{c}}{{{spec.header}}}" for spec in TASK_SPECS]) + r" \\",
        " & ".join(second_header) + r" \\",
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}",
        r"}",
        (
            r"\caption{Main matched-resolution results. Each task is evaluated at its training resolution, "
            r"so this table isolates density-shift robustness from resolution shift. We report in-distribution RMSE, "
            r"worst-case RMSE, and average nonuniform prediction drift for both the synthetic surface-functional task "
            r"together with the two real-data benchmarks: AhmedML sparse surface-force regression and AirfRANS "
            r"sparse airfoil-force regression. Here ``In-dist.\ RMSE'' denotes RMSE under uniform sampling at the "
            r"training resolution, ``Worst RMSE'' denotes the largest RMSE across sampling modes at that same "
            r"resolution, and ``Pred.\ Drift'' denotes the mean absolute difference between a "
            r"prediction from a sparse or shifted view and the dense-reference prediction for the same object, averaged "
            r"over target components and over nonuniform sampling modes.}"
        ),
        r"\label{tab:main-results}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def classification_table_rows(
    payload: dict[str, Any],
    pointnext_payload: dict[str, Any] | None = None,
) -> list[tuple[str, str, dict[str, Any] | None]]:
    rows = [(model_name, MODEL_LABELS[model_name], payload) for model_name in MODEL_ORDER]
    if pointnext_payload is not None:
        rows.append((POINTNEXT_ROW, POINTNEXT_LABEL, pointnext_payload))
    return rows


def infer_task_kind(payload: dict[str, Any]) -> str:
    model_name = first_available_model(payload)
    if model_name is None:
        raise ValueError("Metrics payload does not contain any models.")
    aggregate = payload["models"][model_name]["metrics"]["aggregate"]
    return "classification" if "accuracy_by_count" in aggregate else "regression"


def render_sampling_breakdown_table(
    payload: dict[str, Any],
    source_name: str,
    fixed_point: int | None,
    *,
    pointnext_payload: dict[str, Any] | None = None,
    setting_description: str = "low-resolution stress setting",
    label: str = "tab:sampling-breakdown",
) -> str:
    kind = infer_task_kind(payload)
    point_key = closest_point_key(payload, preferred=fixed_point)
    if point_key is None:
        raise ValueError("Could not determine a point count for the sampling breakdown table.")

    modes = list(payload["sampling_modes"])
    uniform_key = infer_uniform_key(payload)
    summary_label = "accuracy" if kind == "classification" else "RMSE"
    metric_id = "mode_metric_cls" if kind == "classification" else "mode_metric_rmse"

    if kind == "classification":
        aggregate_key = "accuracy_by_count"
        avg_key = "avg_nonuniform_accuracy"
        worst_key = "worst_case_accuracy"
        direction = True
    else:
        aggregate_key = "rmse_by_count"
        avg_key = "avg_nonuniform_rmse"
        worst_key = "worst_case_rmse"
        direction = False

    def _format_mode_header(mode: str) -> str:
        if "_" not in mode:
            return mode
        return r"\shortstack{" + mode.replace("_", r"\\") + "}"

    columns = modes + ["Avg. Nonunif.", "Worst"]
    column_headers = [_format_mode_header(mode) for mode in modes] + ["Avg. Nonunif.", "Worst"]
    rows = regression_table_rows(pointnext_payload) if kind != "classification" else classification_table_rows(payload, pointnext_payload)

    row_values: dict[str, list[float | None]] = {}
    for row_name, _row_label, row_payload in rows:
        model_name = "pointnext" if row_name == POINTNEXT_ROW else row_name
        if row_payload is None:
            row_payload = payload
        metrics = get_model_metrics(row_payload, model_name)
        if metrics is None:
            row_values[row_name] = [None] * len(columns)
            continue
        aggregate = metrics["aggregate"]
        series = [float(aggregate[aggregate_key][point_key][mode]) for mode in modes]
        nonuniform_modes = [mode for mode in modes if mode != uniform_key]
        if not nonuniform_modes:
            nonuniform_modes = [uniform_key]
        series.append(float(aggregate[avg_key][point_key]))
        series.append(float(aggregate[worst_key][point_key]))
        row_values[row_name] = series

    best_masks_by_col = []
    for col_idx in range(len(columns)):
        values = [row_values[row_name][col_idx] for row_name, _row_label, _row_payload in rows]
        best_masks_by_col.append(compute_best_mask(values, higher_is_better=direction))

    body_lines = []
    for model_idx, (row_name, row_label, _row_payload) in enumerate(rows):
        cells = [row_label]
        for col_idx, value in enumerate(row_values[row_name]):
            formatted = format_metric(value, metric_id)
            cells.append(wrap_best(formatted, best_masks_by_col[col_idx][model_idx]))
        body_lines.append(" & ".join(cells) + r" \\")

    lines = [
        "% Auto-generated by paper/generate_result_tables.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.2pt}",
        rf"\begin{{tabular}}{{l{'c' * len(columns)}}}",
        r"\hline",
        " & ".join(["Model"] + column_headers) + r" \\",
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}",
        (
            rf"\caption{{Sampling-mode breakdown on the {source_name} benchmark at the {setting_description} "
            rf"$N={point_key}$. Entries report {summary_label} under each sampling mode, followed by the average over "
            rf"nonuniform modes and the worst-case value across all modes.}}"
        ),
        rf"\label{{{label}}}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def render_resolution_table(
    payload: dict[str, Any],
    source_name: str,
    *,
    pointnext_payload: dict[str, Any] | None = None,
    label: str = "tab:resolution-refinement",
) -> str:
    kind = infer_task_kind(payload)
    counts = point_counts(payload)
    if not counts:
        raise ValueError("Could not determine point counts for the resolution table.")
    point_keys = [str(count) for count in counts]

    if kind == "classification":
        primary_header = "Worst Acc. $\\uparrow$"
        secondary_header = "Pred. Cons. $\\uparrow$"
        primary_metric_id = "worst_acc"
        secondary_metric_id = "pred_cons"
        direction_primary = True
        direction_secondary = True
    else:
        primary_header = "Worst RMSE $\\downarrow$"
        secondary_header = "Pred. Drift $\\downarrow$"
        primary_metric_id = "worst_rmse"
        secondary_metric_id = "pred_drift"
        direction_primary = False
        direction_secondary = False

    rows = regression_table_rows(pointnext_payload) if kind != "classification" else classification_table_rows(payload, pointnext_payload)

    primary_rows: dict[str, list[float | None]] = {}
    secondary_rows: dict[str, list[float | None]] = {}
    for row_name, _row_label, row_payload in rows:
        model_name = "pointnext" if row_name == POINTNEXT_ROW else row_name
        if row_payload is None:
            row_payload = payload
        metrics = get_model_metrics(row_payload, model_name)
        if metrics is None:
            primary_rows[row_name] = [None] * len(point_keys)
            secondary_rows[row_name] = [None] * len(point_keys)
            continue
        aggregate = metrics["aggregate"]
        if kind == "classification":
            primary_rows[row_name] = [float(aggregate["worst_case_accuracy"][point_key]) for point_key in point_keys]
            secondary_rows[row_name] = [
                avg_nonuniform_from_per_setting(row_payload, model_name, "prediction_consistency", point_key)
                for point_key in point_keys
            ]
        else:
            primary_rows[row_name] = [float(aggregate["worst_case_rmse"][point_key]) for point_key in point_keys]
            secondary_rows[row_name] = [
                float(aggregate["avg_nonuniform_prediction_drift"][point_key]) for point_key in point_keys
            ]

    primary_best_masks = []
    secondary_best_masks = []
    for col_idx in range(len(point_keys)):
        primary_values = [primary_rows[row_name][col_idx] for row_name, _row_label, _row_payload in rows]
        secondary_values = [secondary_rows[row_name][col_idx] for row_name, _row_label, _row_payload in rows]
        primary_best_masks.append(compute_best_mask(primary_values, higher_is_better=direction_primary))
        secondary_best_masks.append(compute_best_mask(secondary_values, higher_is_better=direction_secondary))

    body_lines = []
    for model_idx, (row_name, row_label, _row_payload) in enumerate(rows):
        cells = [row_label]
        for col_idx, value in enumerate(primary_rows[row_name]):
            cells.append(wrap_best(format_metric(value, primary_metric_id), primary_best_masks[col_idx][model_idx]))
        for col_idx, value in enumerate(secondary_rows[row_name]):
            cells.append(wrap_best(format_metric(value, secondary_metric_id), secondary_best_masks[col_idx][model_idx]))
        body_lines.append(" & ".join(cells) + r" \\")

    count_headers = [str(count) for count in counts]
    lines = [
        "% Auto-generated by paper/generate_result_tables.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.0pt}",
        rf"\begin{{tabular}}{{l{'c' * (2 * len(point_keys))}}}",
        r"\hline",
        " & ".join(
            [""] +
            [rf"\multicolumn{{{len(point_keys)}}}{{c}}{{{primary_header}}}"] +
            [rf"\multicolumn{{{len(point_keys)}}}{{c}}{{{secondary_header}}}"]
        ) + r" \\",
        " & ".join(["Model"] + count_headers + count_headers) + r" \\",
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}",
        (
            rf"\caption{{Resolution/refinement study on the {source_name} benchmark. The left block reports the primary "
            rf"worst-case task metric across point counts, while the right block reports the corresponding same-object "
            rf"consistency metric, namely average nonuniform prediction drift relative to the dense-reference prediction "
            rf"for the same object. This table is intended to expose whether gains from geometry-aware weighting persist "
            rf"as the discretization is refined.}}"
        ),
        rf"\label{{{label}}}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics_paths = {
        "regression": Path(args.regression_metrics),
        "ahmed": Path(args.ahmed_metrics),
        "airfrans": Path(args.airfrans_metrics),
    }
    payloads = {key: load_json(path) for key, path in metrics_paths.items()}
    pointnext_paths = {
        "regression": Path(args.regression_pointnext_metrics),
        "ahmed": Path(args.ahmed_pointnext_metrics),
        "airfrans": Path(args.airfrans_pointnext_metrics),
    }
    pointnext_payloads = {key: load_json(path) for key, path in pointnext_paths.items()}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_table = render_main_results_table(payloads, pointnext_payloads)
    write_text(output_dir / "table_main_results.tex", main_table)

    breakdown_payload = payloads.get(args.breakdown_source)
    if breakdown_payload is not None:
        breakdown_label = next(spec.label for spec in TASK_SPECS if spec.key == args.breakdown_source)
        breakdown_table = render_sampling_breakdown_table(
            breakdown_payload,
            source_name=breakdown_label,
            fixed_point=args.breakdown_point,
            pointnext_payload=pointnext_payloads.get(args.breakdown_source),
            setting_description=(
                "matched training resolution"
                if args.breakdown_source == "ahmed"
                else "matched training resolution"
            ),
            label="tab:sampling-breakdown",
        )
        write_text(output_dir / "table_sampling_breakdown.tex", breakdown_table)

    resolution_payload = payloads.get(args.resolution_source)
    if resolution_payload is not None:
        resolution_label = next(spec.label for spec in TASK_SPECS if spec.key == args.resolution_source)
        resolution_table = render_resolution_table(
            resolution_payload,
            source_name=resolution_label,
            pointnext_payload=pointnext_payloads.get(args.resolution_source),
            label="tab:resolution-refinement",
        )
        write_text(output_dir / "table_resolution_refinement.tex", resolution_table)

    airfrans_payload = payloads.get("airfrans")
    if airfrans_payload is not None:
        airfrans_breakdown = render_sampling_breakdown_table(
            airfrans_payload,
            source_name="AirfRANS",
            fixed_point=None,
            pointnext_payload=pointnext_payloads.get("airfrans"),
            setting_description="matched training resolution",
            label="tab:airfrans-sampling-breakdown",
        )
        write_text(output_dir / "table_airfrans_sampling_breakdown.tex", airfrans_breakdown)

        airfrans_resolution = render_resolution_table(
            airfrans_payload,
            source_name="AirfRANS",
            pointnext_payload=pointnext_payloads.get("airfrans"),
            label="tab:airfrans-resolution-refinement",
        )
        write_text(output_dir / "table_airfrans_resolution_refinement.tex", airfrans_resolution)

    summary_lines = [
        "Generated table templates:",
        f"- main results: {output_dir / 'table_main_results.tex'}",
    ]
    if breakdown_payload is not None:
        summary_lines.append(f"- sampling breakdown: {output_dir / 'table_sampling_breakdown.tex'}")
    else:
        summary_lines.append(f"- sampling breakdown: skipped (missing {metrics_paths[args.breakdown_source]})")
    if resolution_payload is not None:
        summary_lines.append(f"- resolution/refinement: {output_dir / 'table_resolution_refinement.tex'}")
    else:
        summary_lines.append(f"- resolution/refinement: skipped (missing {metrics_paths[args.resolution_source]})")
    if airfrans_payload is not None:
        summary_lines.append(f"- AirfRANS sampling breakdown: {output_dir / 'table_airfrans_sampling_breakdown.tex'}")
        summary_lines.append(f"- AirfRANS resolution/refinement: {output_dir / 'table_airfrans_resolution_refinement.tex'}")
    else:
        summary_lines.append(f"- AirfRANS sampling breakdown: skipped (missing {metrics_paths['airfrans']})")
        summary_lines.append(f"- AirfRANS resolution/refinement: skipped (missing {metrics_paths['airfrans']})")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
