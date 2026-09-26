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

    run_calibration_tool(a.video, a.out, a.court, init=a.init, t=a.time, cache_dir=a.cache_dir,
                         port=a.port, open_browser=not a.no_browser)


def _cmd_label(a: argparse.Namespace) -> None:
    from .labeling import run_label_server

    run_label_server(a.video, a.out, cache_dir=a.cache_dir, port=a.port, open_browser=not a.no_browser)


def _cmd_ball_review(a: argparse.Namespace) -> None:
    from .labeling.ball_review import run_ball_review

    run_ball_review(a.videos or None, port=a.port, open_browser=not a.no_browser)


def _cmd_ball_dataset(a: argparse.Namespace) -> None:
    import json

    from .training.ball_dataset import build_ball_dataset

    print(json.dumps(build_ball_dataset(a.out, a.labels_dir), indent=1))


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
    s.add_argument("--out", help="default: calib/<video stem>.json")
    s.add_argument("--court", default="fiba_3x3")
    s.add_argument("--init", help="start from 'trackid3x3-outdoor' or an existing calib.json")
    s.add_argument("--time", type=float, default=1.0, help="reference frame time in seconds")
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=_cmd_calibrate)

    s = sub.add_parser("label", help="label shot events in the browser (writes labels/<video>.csv)")
    s.add_argument("video")
    s.add_argument("--out", help="label CSV path (default: labels/<video stem>.csv)")
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=_cmd_label)

    s = sub.add_parser("ball-review", help="review / correct pre-annotated ball boxes in the browser")
    s.add_argument("videos", nargs="*", help="default: all videos under data/ball_frames/")
    s.add_argument("--port", type=int, default=8767)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=_cmd_ball_review)

    s = sub.add_parser("ball-dataset", help="build ball_dataset.zip (YOLO format) from reviewed ball labels")
    s.add_argument("--out", default="data/ball_dataset")
    s.add_argument("--labels-dir", default="labels/ball")
    s.set_defaults(func=_cmd_ball_dataset)

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
