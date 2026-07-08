#!/usr/bin/env python3
"""Runs incremental Bazel analysis benchmarks for JVM and native Bazel binaries."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid


ELAPSED_RE = re.compile(r"INFO: Elapsed time: ([0-9.]+)s")
LABEL_STYLES = {
    "pgo-jvm": ("JVM", "#3b6ea8"),
    "jvm": ("JVM", "#3b6ea8"),
    "pgo-native-nopgo": ("GraalVM Native", "#b45f37"),
    "native-nopgo": ("GraalVM Native", "#b45f37"),
    "pgo-native-pgo": ("GraalVM Native + PGO", "#26734d"),
    "native-pgo": ("GraalVM Native + PGO", "#26734d"),
}
LABEL_ORDER = {
    "pgo-jvm": 0,
    "jvm": 0,
    "pgo-native-nopgo": 1,
    "native-nopgo": 1,
    "pgo-native-pgo": 2,
    "native-pgo": 2,
}


def run(argv: list[str], cwd: Path | None = None, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, text=True, **kwargs)


def run_checked(argv: list[str], cwd: Path | None = None) -> str:
    result = run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(argv)}\n{result.stderr}"
        )
    return result.stdout


def resolve_git_path(worktree: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = worktree / path
    return path.resolve()


def validate_benchmark_worktree(worktree: Path) -> Path:
    """Requires a dedicated linked worktree before destructive preparation."""
    requested = worktree.resolve()
    try:
        top_level = Path(
            run_checked(
                ["git", "-C", str(requested), "rev-parse", "--show-toplevel"]
            ).strip()
        ).resolve()
        git_dir = resolve_git_path(
            requested,
            run_checked(
                ["git", "-C", str(requested), "rev-parse", "--absolute-git-dir"]
            ).strip(),
        )
        common_dir = resolve_git_path(
            requested,
            run_checked(
                ["git", "-C", str(requested), "rev-parse", "--git-common-dir"]
            ).strip(),
        )
        head_name = run_checked(
            ["git", "-C", str(requested), "rev-parse", "--abbrev-ref", "HEAD"]
        ).strip()
    except RuntimeError as error:
        raise ValueError(
            f"benchmark worktree is not a Git worktree: {requested}"
        ) from error

    if requested != top_level:
        raise ValueError(
            f"benchmark worktree must name its top level: {requested} != {top_level}"
        )
    if git_dir == common_dir:
        raise ValueError(
            "refusing to reset and clean the primary Git checkout; "
            f"use a dedicated linked worktree instead: {top_level}"
        )
    if head_name != "HEAD":
        raise ValueError(
            "refusing to reset and clean a branch-backed Git worktree; "
            f"use a detached worktree instead: {top_level}"
        )
    return top_level


def read_commits(path: Path) -> list[str]:
    commits = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not commits:
        raise ValueError(f"no commits found in {path}")
    return commits


def commit_subject(worktree: Path, commit: str) -> str:
    return run_checked(
        ["git", "-C", str(worktree), "show", "-s", "--format=%s", commit]
    ).strip()


def clean_worktree(worktree: Path, first_commit: str | None) -> None:
    run_checked(["git", "-C", str(worktree), "reset", "--hard", "--quiet"])
    if first_commit:
        run_checked(
            [
                "git",
                "-C",
                str(worktree),
                "checkout",
                "--quiet",
                "--detach",
                first_commit,
            ]
        )
    run_checked(["git", "-C", str(worktree), "reset", "--hard", "--quiet"])
    run_checked(["git", "-C", str(worktree), "clean", "-ffdqx"])


def shutdown(binary: Path, output_base: Path) -> None:
    if not output_base.exists():
        return
    result = run(
        [str(binary), f"--output_base={output_base}", "shutdown"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to shut down {binary} at {output_base} "
            f"(exit {result.returncode}):\n{result.stderr}"
        )


def remove_output_base(output_base: Path) -> None:
    if not output_base.exists():
        return
    # Bazel may extract read-only toolchain directories below an output base.
    run_checked(["chmod", "-R", "u+rwX", str(output_base)])
    shutil.rmtree(output_base)
    if output_base.exists():
        raise RuntimeError(f"failed to remove output base: {output_base}")


def prepare(args: argparse.Namespace) -> int:
    args.worktree = validate_benchmark_worktree(args.worktree)
    commits = read_commits(args.commits) if args.commits else []
    first_commit = commits[0] if commits else None
    for binary in args.binary:
        for output_base in args.output_base:
            shutdown(binary, output_base)
    for output_base in args.output_base:
        remove_output_base(output_base)
    if args.worktree.exists():
        clean_worktree(args.worktree, first_commit)
    return 0


def proc_status(pid: int) -> dict[str, int]:
    status_path = Path("/proc") / str(pid) / "status"
    values: dict[str, int] = {}
    try:
        for line in status_path.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmSize:", "VmHWM:")):
                key, value, *_ = line.split()
                values[key.rstrip(":").lower() + "_kb"] = int(value)
    except FileNotFoundError:
        pass
    return values


def server_pid(output_base: Path) -> int | None:
    pid_file = output_base / "server" / "server.pid.txt"
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "binary_label",
        "commit_index",
        "commit",
        "short_commit",
        "subject",
        "exit_code",
        "wall_sec",
        "bazel_elapsed_sec",
        "server_pid",
        "vmrss_kb",
        "vmhwm_kb",
        "vmsize_kb",
        "stdout_path",
        "stderr_path",
        "profile_path",
        "memory_profile_path",
    ]
    needs_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def label_display(label: str) -> str:
    return LABEL_STYLES.get(label, (label, "#6b7280"))[0]


def label_color(label: str) -> str:
    return LABEL_STYLES.get(label, (label, "#6b7280"))[1]


def label_sort_key(label: str) -> tuple[int, str]:
    return (LABEL_ORDER.get(label, 100), label)


def pct_delta(value: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return (value - baseline) / baseline * 100


def pct_cell(value: float | None, lower_is_better: bool = True) -> str:
    if value is None:
        return '<td class="neutral">n/a</td>'
    css = "neutral"
    if value != 0:
        is_good = value < 0 if lower_is_better else value > 0
        css = "good" if is_good else "bad"
    return f'<td class="{css}">{value:+.1f}%</td>'


def seconds_cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def mib_cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} MiB"


def read_result_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, object] = dict(row)
            converted["commit_index"] = int(row["commit_index"])
            converted["exit_code"] = int(row["exit_code"])
            for key in ("wall_sec", "bazel_elapsed_sec", "vmrss_kb", "vmhwm_kb"):
                converted[key] = float(row[key]) if row.get(key) else None
            rows.append(converted)
    return rows


def read_hyperfine_means(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    means: dict[str, float] = {}
    for result in data.get("results", []):
        command = result.get("command", "")
        match = re.search(r"(^| )--label ([^ ]+)", command)
        if match:
            means[match.group(2)] = float(result["mean"])
    return means


def grouped_values(
    rows: list[dict[str, object]], field: str, aggregate
) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if row["exit_code"] != 0 or row.get(field) is None:
            continue
        label = str(row["binary_label"])
        index = int(row["commit_index"])
        values.setdefault(label, {}).setdefault(index, []).append(float(row[field]))
    return {
        label: {index: aggregate(items) for index, items in by_commit.items()}
        for label, by_commit in values.items()
    }


def run_summaries(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_label_run: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        if row["exit_code"] != 0:
            continue
        by_label_run.setdefault(str(row["binary_label"]), {}).setdefault(
            str(row["run_id"]), []
        ).append(row)

    summaries: dict[str, dict[str, float]] = {}
    for label, by_run in by_label_run.items():
        elapsed_sums = []
        wall_sums = []
        peak_rss = []
        final_rss = []
        peak_hwm = []
        for run_rows in by_run.values():
            run_rows.sort(key=lambda r: int(r["commit_index"]))
            elapsed_sums.append(
                sum(float(r["bazel_elapsed_sec"] or 0) for r in run_rows)
            )
            wall_sums.append(sum(float(r["wall_sec"] or 0) for r in run_rows))
            rss_values = [
                float(r["vmrss_kb"]) / 1024
                for r in run_rows
                if r.get("vmrss_kb") is not None
            ]
            hwm_values = [
                float(r["vmhwm_kb"]) / 1024
                for r in run_rows
                if r.get("vmhwm_kb") is not None
            ]
            if rss_values:
                peak_rss.append(max(rss_values))
                final_rss.append(rss_values[-1])
            if hwm_values:
                peak_hwm.append(max(hwm_values))
        summaries[label] = {
            "elapsed_sum": statistics.mean(elapsed_sums),
            "wall_sum": statistics.mean(wall_sums),
            "peak_rss": statistics.mean(peak_rss),
            "final_rss": statistics.mean(final_rss),
            "peak_hwm": statistics.mean(peak_hwm),
            "runs": float(len(by_run)),
        }
    return summaries


def parse_memory_profile(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        parts = line.split(":")
        if len(parts) != 4:
            continue
        _phase, area, metric, value = parts
        if area not in ("heap", "non-heap"):
            continue
        key = f"{area}_{metric}"
        try:
            values[key] = float(value) / 1024 / 1024
        except ValueError:
            pass
    return values


def final_memory_profiles(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_label_run: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        if row["exit_code"] != 0:
            continue
        label = str(row["binary_label"])
        run_id = str(row["run_id"])
        current = by_label_run.setdefault(label, {}).get(run_id)
        if current is None or int(row["commit_index"]) > int(current["commit_index"]):
            by_label_run[label][run_id] = row

    result: dict[str, dict[str, float]] = {}
    for label, by_run in by_label_run.items():
        values: dict[str, list[float]] = {}
        for row in by_run.values():
            profile = parse_memory_profile(Path(str(row["memory_profile_path"])))
            for key, value in profile.items():
                values.setdefault(key, []).append(value)
        result[label] = {
            key: statistics.mean(items) for key, items in values.items() if items
        }
    return result


def svg_line_chart(
    title: str,
    series: dict[str, dict[int, float]],
    *,
    ylabel_unit: str,
    include_commit_1: bool = True,
) -> str:
    labels = sorted(series, key=label_sort_key)
    indexes = sorted({index for values in series.values() for index in values})
    if not include_commit_1:
        indexes = [index for index in indexes if index != 1]
    if not indexes:
        return ""

    all_values = [
        series[label][index]
        for label in labels
        for index in indexes
        if index in series[label]
    ]
    if not all_values:
        return ""
    max_value = max(all_values)
    y_max = max_value * 1.08 if max_value > 0 else 1
    if y_max == 0:
        y_max = 1

    left, right, top, bottom = 60.0, 900.0, 24.0, 238.0
    width, height = right - left, bottom - top

    def x(index: int) -> float:
        if len(indexes) == 1:
            return left + width / 2
        return left + width * indexes.index(index) / (len(indexes) - 1)

    def y(value: float) -> float:
        return bottom - height * value / y_max

    parts = [
        f'<svg viewBox="0 0 920 280" role="img" aria-label="{html.escape(title)}">',
        f'<text x="60" y="18" class="chart-title">{html.escape(title)}</text>',
    ]
    for i in range(5):
        value = y_max * i / 4
        y_pos = bottom - height * i / 4
        parts.append(
            f'<line x1="{left:.0f}" y1="{y_pos:.1f}" x2="{right:.0f}" '
            f'y2="{y_pos:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="8" y="{y_pos + 4:.1f}" class="axis">{value:.1f}</text>'
        )
    parts.append(
        f'<line x1="{left:.0f}" y1="{bottom:.0f}" x2="{right:.0f}" '
        f'y2="{bottom:.0f}" class="axis-line"/>'
    )
    parts.append(
        f'<line x1="{left:.0f}" y1="{top:.0f}" x2="{left:.0f}" '
        f'y2="{bottom:.0f}" class="axis-line"/>'
    )

    for label in labels:
        points = [
            (index, series[label][index])
            for index in indexes
            if index in series[label]
        ]
        if not points:
            continue
        color = label_color(label)
        polyline = " ".join(f"{x(index):.1f},{y(value):.1f}" for index, value in points)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="2.4"/>'
        )
        for index, value in points:
            title_text = (
                f"{label_display(label)} commit {index}: {value:.3f} {ylabel_unit}"
            )
            parts.append(
                f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="2.6" '
                f'fill="{color}"><title>{html.escape(title_text)}</title></circle>'
            )

    tick_indexes = [indexes[0]]
    for candidate in (5, 10, 15, 20):
        if candidate in indexes and candidate not in tick_indexes:
            tick_indexes.append(candidate)
    if indexes[-1] not in tick_indexes:
        tick_indexes.append(indexes[-1])
    for index in tick_indexes:
        parts.append(
            f'<text x="{x(index):.1f}" y="268" class="axis" '
            f'text-anchor="middle">{index}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def summary_table(
    labels: list[str],
    summaries: dict[str, dict[str, float]],
    hyperfine_means: dict[str, float],
) -> str:
    baseline = labels[0] if labels else ""
    native = labels[1] if len(labels) > 1 else ""
    pgo = labels[2] if len(labels) > 2 else ""
    metrics = [
        ("Hyperfine full sequence mean", hyperfine_means, seconds_cell),
        ("Runner Bazel elapsed sum mean", {k: v["elapsed_sum"] for k, v in summaries.items()}, seconds_cell),
        ("Runner wall sum mean", {k: v["wall_sum"] for k, v in summaries.items()}, seconds_cell),
        ("Peak RSS mean", {k: v["peak_rss"] for k, v in summaries.items()}, mib_cell),
        ("Final RSS mean", {k: v["final_rss"] for k, v in summaries.items()}, mib_cell),
        ("Peak HWM mean", {k: v["peak_hwm"] for k, v in summaries.items()}, mib_cell),
    ]
    header = "".join(f"<th>{html.escape(label_display(label))}</th>" for label in labels)
    rows = []
    for metric_name, values, formatter in metrics:
        cells = [f"<td>{html.escape(metric_name)}</td>"]
        for label in labels:
            cells.append(f"<td>{formatter(values.get(label))}</td>")
        base_value = values.get(baseline)
        native_value = values.get(native)
        pgo_value = values.get(pgo)
        cells.append(pct_cell(pct_delta(native_value, base_value) if native_value and base_value else None))
        cells.append(pct_cell(pct_delta(pgo_value, base_value) if pgo_value and base_value else None))
        cells.append(pct_cell(pct_delta(pgo_value, native_value) if pgo_value and native_value else None))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr><th>Metric</th>'
        + header
        + "<th>Native vs JVM</th><th>PGO vs JVM</th><th>PGO vs Native</th>"
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def final_memory_table(labels: list[str], profiles: dict[str, dict[str, float]]) -> str:
    metric_names = [
        ("Final heap used after forced GC", "heap_used"),
        ("Final heap committed after forced GC", "heap_commited"),
        ("Final max heap", "heap_max"),
        ("Final non-heap committed", "non-heap_commited"),
    ]
    header = "".join(f"<th>{html.escape(label_display(label))}</th>" for label in labels)
    rows = []
    for title, key in metric_names:
        cells = [f"<td>{html.escape(title)}</td>"]
        for label in labels:
            value = profiles.get(label, {}).get(key)
            cells.append(f"<td>{mib_cell(value)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr><th>Metric</th>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def medians_table(
    rows: list[dict[str, object]], labels: list[str], elapsed: dict[str, dict[int, float]]
) -> str:
    representative: dict[int, dict[str, object]] = {}
    for row in rows:
        representative.setdefault(int(row["commit_index"]), row)

    rows_html = []
    for index in sorted(representative):
        row = representative[index]
        cells = [
            f"<td>{index}</td>",
            f'<td><code>{html.escape(str(row["short_commit"]))}</code></td>',
            f"<td>{html.escape(str(row['subject']))}</td>",
        ]
        for label in labels:
            cells.append(f"<td>{seconds_cell(elapsed.get(label, {}).get(index))}</td>")
        if len(labels) >= 3:
            native_value = elapsed.get(labels[1], {}).get(index)
            pgo_value = elapsed.get(labels[2], {}).get(index)
            cells.append(
                pct_cell(
                    pct_delta(pgo_value, native_value)
                    if pgo_value is not None and native_value is not None
                    else None
                )
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(
        f"<th>{html.escape(label_display(label))} elapsed</th>" for label in labels
    )
    return (
        '<div class="table-scroll"><table><thead><tr><th>#</th><th>Commit</th><th>Subject</th>'
        + header
        + "<th>PGO vs Native</th></tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
    )


def generate_report(args: argparse.Namespace) -> int:
    rows = read_result_rows(args.results_csv)
    labels = sorted({str(row["binary_label"]) for row in rows}, key=label_sort_key)
    elapsed = grouped_values(rows, "bazel_elapsed_sec", statistics.median)
    rss = grouped_values(
        [
            {
                **row,
                "vmrss_mib": float(row["vmrss_kb"]) / 1024
                if row.get("vmrss_kb") is not None
                else None,
            }
            for row in rows
        ],
        "vmrss_mib",
        statistics.mean,
    )
    summaries = run_summaries(rows)
    hyperfine_means = read_hyperfine_means(args.hyperfine_json)
    memory_profiles = final_memory_profiles(rows)
    successes = sum(1 for row in rows if row["exit_code"] == 0)
    failures = len(rows) - successes
    runs_per_binary = int(min(s["runs"] for s in summaries.values())) if summaries else 0

    style = (
        ":root{color-scheme:light;--text:#17202a;--muted:#5f6b7a;"
        "--line:#d6dce5;--good:#0b6b3a;--bad:#9b2c2c;--neutral:#5f6b7a;"
        "--bg:#fff;--panel:#f7f9fc}body{margin:0;font:14px/1.45 "
        "system-ui,-apple-system,Segoe UI,sans-serif;color:var(--text);"
        "background:var(--bg)}main{max-width:1180px;margin:0 auto;"
        "padding:32px 28px 56px}h1{font-size:30px;line-height:1.15;"
        "margin:0 0 8px;letter-spacing:0}h2{font-size:20px;margin:32px 0 10px}"
        "p{color:var(--muted);margin:6px 0 16px}code{font-family:ui-monospace,"
        "SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}"
        ".table-scroll{width:100%;overflow-x:auto;margin:12px 0 20px}"
        ".table-scroll table{min-width:720px;margin:0}"
        "table{border-collapse:collapse;width:100%;"
        "background:white}th,td{border:1px solid var(--line);padding:8px 10px;"
        "text-align:right;vertical-align:top}th:first-child,td:first-child,"
        "td:nth-child(2),td:nth-child(3){text-align:left}th{background:var(--panel);"
        "font-weight:650}.good{color:var(--good);font-weight:650}"
        ".bad{color:var(--bad);font-weight:650}.neutral{color:var(--neutral);"
        "font-weight:650}.note{border-left:4px solid #6b7c93;background:var(--panel);"
        "padding:12px 14px;margin:16px 0;color:var(--text)}.grid{stroke:#e8edf4;"
        "stroke-width:1}.axis-line{stroke:#9aa6b2;stroke-width:1}.axis{fill:#647082;"
        "font-size:11px}.chart-title{fill:#17202a;font-size:14px;font-weight:650}"
        ".legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 20px}"
        ".swatch{display:inline-block;width:12px;height:12px;border-radius:2px;"
        "margin-right:6px;vertical-align:-1px}svg{width:100%;height:auto;"
        "border:1px solid var(--line);background:white;margin:10px 0 18px}"
    )
    legend = "".join(
        '<span><span class="swatch" style="background:'
        + label_color(label)
        + '"></span>'
        + html.escape(label_display(label))
        + "</span>"
        for label in labels
    )
    note = (
        f"All measured per-commit invocations passed: {successes} rows, "
        f"{runs_per_binary} runs per binary."
    )
    if failures:
        note += f" Failures: {failures}."
    if args.note:
        note += " " + args.note

    artifacts = [
        ("Report directory", str(args.output.parent)),
        ("Per-commit CSV", str(args.results_csv)),
    ]
    if args.hyperfine_json:
        artifacts.insert(1, ("Hyperfine JSON", str(args.hyperfine_json)))
    if args.pgo_profile:
        artifacts.append(("PGO profile", str(args.pgo_profile)))
    artifacts_table = (
        '<div class="table-scroll"><table><tbody>'
        + "".join(
            f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(value)}</code></td></tr>"
            for name, value in artifacts
        )
        + "</tbody></table></div>"
    )

    document = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{html.escape(args.title)}</title>",
            '<link rel="icon" href="data:,">',
            f"<style>{style}</style></head>",
            "<body><main>",
            f"<h1>{html.escape(args.title)}</h1>",
            f"<p>{html.escape(args.description)}</p>" if args.description else "",
            f'<div class="note">{html.escape(note)}</div>',
            "<h2>Summary</h2>",
            summary_table(labels, summaries, hyperfine_means),
            "<h2>Final Memory Profile</h2>",
            "<p>These values come from Bazel <code>--memory_profile</code>, "
            "which forces GC before recording final heap usage.</p>",
            final_memory_table(labels, memory_profiles),
            "<h2>Per-Commit Charts</h2>",
            f'<div class="legend">{legend}</div>',
            svg_line_chart(
                "Median Bazel Elapsed Time By Commit",
                elapsed,
                ylabel_unit="s",
            ),
            svg_line_chart(
                "Median Bazel Elapsed Time By Commit, Excluding Cold Commit 1",
                elapsed,
                ylabel_unit="s",
                include_commit_1=False,
            ),
            svg_line_chart("Mean Server RSS By Commit", rss, ylabel_unit="MiB"),
            "<h2>Per-Commit Medians</h2>",
            medians_table(rows, labels, elapsed),
            "<h2>Artifacts</h2>",
            artifacts_table,
            "</main></body></html>",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    return 0


def benchmark(args: argparse.Namespace) -> int:
    args.worktree = validate_benchmark_worktree(args.worktree)
    commits = read_commits(args.commits)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.disk_cache.mkdir(parents=True, exist_ok=True)
    args.repository_cache.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.label}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    log_dir = args.artifacts / "logs" / run_id
    profile_dir = args.artifacts / "profiles" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, commit in enumerate(commits, start=1):
        clean_worktree(args.worktree, commit)
        subject = commit_subject(args.worktree, commit)
        stdout_path = log_dir / f"{index:02d}-{commit[:12]}.stdout.log"
        stderr_path = log_dir / f"{index:02d}-{commit[:12]}.stderr.log"
        profile_path = profile_dir / f"{index:02d}-{commit[:12]}.profile.gz"
        memory_profile_path = profile_dir / f"{index:02d}-{commit[:12]}.memory.txt"
        command = [
            str(args.binary),
            *args.startup_option,
            f"--output_base={args.output_base}",
            "build",
            "--nobuild",
            "--color=no",
            "--curses=no",
            "--show_progress_rate_limit=60",
            f"--disk_cache={args.disk_cache}",
            f"--repository_cache={args.repository_cache}",
            f"--profile={profile_path}",
            f"--memory_profile={memory_profile_path}",
            args.target,
        ]
        start = time.monotonic()
        result = run(
            command,
            cwd=args.worktree,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        wall_sec = time.monotonic() - start
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        combined_output = result.stdout + "\n" + result.stderr
        elapsed_match = ELAPSED_RE.search(combined_output)
        pid = server_pid(args.output_base)
        memory = proc_status(pid) if pid is not None else {}
        rows.append(
            {
                "run_id": run_id,
                "binary_label": args.label,
                "commit_index": index,
                "commit": commit,
                "short_commit": commit[:12],
                "subject": subject,
                "exit_code": result.returncode,
                "wall_sec": f"{wall_sec:.6f}",
                "bazel_elapsed_sec": elapsed_match.group(1) if elapsed_match else "",
                "server_pid": pid or "",
                "vmrss_kb": memory.get("vmrss_kb", ""),
                "vmhwm_kb": memory.get("vmhwm_kb", ""),
                "vmsize_kb": memory.get("vmsize_kb", ""),
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "profile_path": profile_path,
                "memory_profile_path": memory_profile_path,
            }
        )
        if result.returncode != 0:
            append_rows(args.results_csv, rows)
            sys.stderr.write(
                f"{args.label} failed on {commit[:12]} with exit {result.returncode}\n"
            )
            sys.stderr.write(result.stderr[-4000:])
            return result.returncode
    append_rows(args.results_csv, rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--worktree", type=Path, required=True)
    prepare_parser.add_argument("--commits", type=Path)
    prepare_parser.add_argument("--binary", type=Path, action="append", default=[])
    prepare_parser.add_argument("--output-base", type=Path, action="append", default=[])
    prepare_parser.set_defaults(func=prepare)

    bench_parser = subparsers.add_parser("run")
    bench_parser.add_argument("--label", required=True)
    bench_parser.add_argument("--binary", type=Path, required=True)
    bench_parser.add_argument("--startup-option", action="append", default=[])
    bench_parser.add_argument("--worktree", type=Path, required=True)
    bench_parser.add_argument("--output-base", type=Path, required=True)
    bench_parser.add_argument("--commits", type=Path, required=True)
    bench_parser.add_argument("--artifacts", type=Path, required=True)
    bench_parser.add_argument("--results-csv", type=Path, required=True)
    bench_parser.add_argument("--disk-cache", type=Path, required=True)
    bench_parser.add_argument("--repository-cache", type=Path, required=True)
    bench_parser.add_argument("--target", default="//src:bazel-bin-dev")
    bench_parser.set_defaults(func=benchmark)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--results-csv", type=Path, required=True)
    report_parser.add_argument("--hyperfine-json", type=Path)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--title", default="Bazel GraalVM Native Image Benchmark")
    report_parser.add_argument("--description", default="")
    report_parser.add_argument("--note", default="")
    report_parser.add_argument("--pgo-profile", type=Path)
    report_parser.set_defaults(func=generate_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
