"""Command-line interface: `hoopstats <command> ...`."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict


def _cmd_probe(a: argparse.Namespace) -> None:
    from .video import probe

    for v in a.videos:
        print(asdict(probe(v)))


def _cmd_proxy(a: argparse.Namespace) -> None:
    from .video import make_proxy

    print(make_proxy(a.video, a.out, height=a.height))


def _cmd_evaluate(a: argparse.Namespace) -> None:
    from .evaluation import evaluate, format_report
    from .outputs import read_events_csv

    r = evaluate(read_events_csv(a.gt), read_events_csv(a.pred), a.rules, a.tolerance)
    print(format_report(r))


def _cmd_analyze(a: argparse.Namespace) -> None:
    from .pipeline import RunConfig, analyze

    analyze(RunConfig(video=a.video, calib=a.calib, out_dir=a.out, rules=a.rules,
                      cache_dir=a.cache_dir, render=a.render))


def _cmd_calibrate(a: argparse.Namespace) -> None:
    from .calibration.tool import run_calibration_tool

    run_calibration_tool(a.video, a.out, a.court)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hoopstats", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="print video metadata")
    s.add_argument("videos", nargs="+")
    s.set_defaults(func=_cmd_probe)

    s = sub.add_parser("proxy", help="make an H.264 working copy (e.g. 1080p) of a video")
    s.add_argument("video")
    s.add_argument("out")
    s.add_argument("--height", type=int, default=1080)
    s.set_defaults(func=_cmd_proxy)

    s = sub.add_parser("calibrate", help="interactive calibration tool (Phase 1)")
    s.add_argument("video")
    s.add_argument("--out", required=True)
    s.add_argument("--court", default="fiba_3x3")
    s.set_defaults(func=_cmd_calibrate)

    s = sub.add_parser("analyze", help="run the full pipeline on a video")
    s.add_argument("video")
    s.add_argument("--calib", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rules", choices=["5v5", "3x3"], default="3x3")
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--render", action="store_true")
    s.set_defaults(func=_cmd_analyze)

    s = sub.add_parser("evaluate", help="score predicted events against ground-truth labels")
    s.add_argument("--gt", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--rules", choices=["5v5", "3x3"], default="3x3")
    s.add_argument("--tolerance", type=float, default=1.0)
    s.set_defaults(func=_cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except NotImplementedError as e:
        print(f"not implemented yet: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
