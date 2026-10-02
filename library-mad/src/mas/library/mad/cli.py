#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""``mad-maslab`` — compile MAD problems into MAS apps, and rerun them.

Two verbs, matching the two halves of the port:

    mad-maslab compile <config.yaml> --out <dir> [--index N]
    mad-maslab rerun   <app-dir> --infra-ref <ref>

``compile`` is offline and deterministic: no model is called, so it is the
cheap way to inspect exactly what each seat would be told. ``rerun`` drives
the compiled app through mas-lab and writes the debate result plus the
restricted scratchpad log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from mas.library.mad.compiler import LABELS_FILENAME, MAS_FILENAME, compile_problem
from mas.library.mad.domain.config import MadConfig
from mas.library.mad.domain.problems import load_problems

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mad-maslab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compile_cmd = sub.add_parser("compile", help="Compile MAD problems into MAS apps.")
    compile_cmd.add_argument("config", type=Path, help="MAD run configuration YAML.")
    compile_cmd.add_argument("--out", type=Path, required=True, help="Output directory.")
    compile_cmd.add_argument("--index", type=int, default=0, help="Problem index (default 0).")
    compile_cmd.add_argument("--count", type=int, default=1, help="How many problems to compile.")
    compile_cmd.add_argument("--moderator-model", default=None, help="Model for the moderator.")

    rerun_cmd = sub.add_parser("rerun", help="Run one compiled app and write its artifacts.")
    rerun_cmd.add_argument("app", type=Path, help="Compiled application directory.")
    rerun_cmd.add_argument("--infra-ref", action="append", default=[], help="mas-ctl infra ref.")
    rerun_cmd.add_argument("--out", type=Path, default=None, help="Artifact directory.")

    args = parser.parse_args(argv)
    if args.command == "compile":
        return _compile(args)
    return _rerun(args)


def _compile(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    problems = load_problems(_resolve_input(config, args.config), limit=None)
    selected = problems[args.index : args.index + max(1, args.count)]
    if not selected:
        print(f"error: no problem at index {args.index}", file=sys.stderr)
        return 2

    for problem in selected:
        target = args.out if len(selected) == 1 else args.out / f"q{problem.index:04d}"
        app = compile_problem(
            config, problem, out_dir=target, moderator_model=args.moderator_model
        )
        print(f"{problem.task_id}: {app.mas_manifest} ({len(app.seats)} seats)")
    return 0


def _rerun(args: argparse.Namespace) -> int:
    from mas.ctl.executor.run_mas import build_running_mas
    from mas.ctl.ui.turn_result import turn_failed
    from mas.library.mad.domain import sink

    app_dir = args.app
    manifest = app_dir / MAS_FILENAME
    if not manifest.is_file():
        print(f"error: {manifest} not found", file=sys.stderr)
        return 2

    out_dir = args.out or (app_dir / "artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    seats = _seat_ids(manifest)
    sink.clear()

    mas = build_running_mas(
        manifest,
        infra_refs=list(args.infra_ref),
        manifest_dir=app_dir,
    )
    try:
        result = mas.ask("Run the configured MAD debate.")
    finally:
        mas.close()

    if result is None or turn_failed(result):
        print("error: debate turn failed", file=sys.stderr)
        return 1
    text = result.text

    (out_dir / "debate.json").write_text(_pretty(text), encoding="utf-8")
    leftovers = {seat: sink.pending(seat) for seat in seats if sink.pending(seat)}
    if leftovers:
        print(f"warning: unread restricted records {leftovers}", file=sys.stderr)
    print(f"wrote {out_dir / 'debate.json'}")
    if (app_dir / LABELS_FILENAME).is_file():
        print(f"labels: {app_dir / LABELS_FILENAME}")
    return 0


def _pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"
    except (TypeError, ValueError):
        return (text or "") + "\n"


def _seat_ids(manifest: Path) -> list[str]:
    document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    agents = ((document.get("spec") or {}).get("agency") or {}).get("agents") or []
    return [str(row.get("id")) for row in agents if isinstance(row, dict) and row.get("id")]


def _load_config(path: Path) -> MadConfig:
    return MadConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _resolve_input(config: MadConfig, config_path: Path) -> Path:
    """MAD dataset paths are written relative to the config file."""
    dataset = Path(config.dataset.input)
    if dataset.is_absolute():
        return dataset
    return (config_path.parent / dataset).resolve()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
