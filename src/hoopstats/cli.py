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
    from pathlib import Path

    from .pipeline import RunConfig, analyze

    rng = tuple(float(v) for v in a.render_range.split(":")) if a.render_range else None
    analyze(RunConfig(video=a.video, calib=a.calib or f"calib/{Path(a.video).stem}.json", out_dir=a.out,
                      rules=a.rules, cache_dir=a.cache_dir, ball_model=a.ball_model, make_model=a.make_model,
                      render=a.render or rng is not None, render_range=rng))


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


def _cmd_detect(a: argparse.Namespace) -> None:
    from pathlib import Path

    from .detection.runner import run_detection

    calib = a.calib or f"calib/{Path(a.video).stem}.json"
    run_detection(a.video, calib, a.ball_model, a.person_model, a.cache_dir, a.start, a.end,
                  a.person_every, a.chunk, device=a.device)


def _cmd_track_ball(a: argparse.Namespace) -> None:
    from pathlib import Path

    import numpy as np

    from .detection.runner import load_detections
    from .tracking import track_ball
    from .video import probe

    cache = Path(a.cache_dir) / Path(a.video).stem
    ball, _ = load_detections(cache)
    out = track_ball(ball, probe(a.video).n_frames)
    np.savez_compressed(cache / "ball_track.npz", **out)
    n = int((out["state"] > 0).sum())
    print(f"ball positions in {n} frames ({int((out['state'] == 2).sum())} interpolated) -> {cache / 'ball_track.npz'}")


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

    s = sub.add_parser("detect", help="run ball + player detection over a video (cached, resumable)")
    s.add_argument("video")
    s.add_argument("--calib", help="default: calib/<video stem>.json")
    s.add_argument("--ball-model", default="data/models/ball_v1.pt")
    s.add_argument("--person-model", default="data/models/yolo11s.pt")
    s.add_argument("--cache-dir", default="cache")
    s.add_argument("--start", type=int, default=0, help="first frame")
    s.add_argument("--end", type=int, default=None, help="last frame (exclusive)")
    s.add_argument("--person-every", type=int, default=3)
    s.add_argument("--chunk", type=int, default=900)
    s.add_argument("--device", default=None, help="mps / cuda / cpu (default: best available)")
    s.set_defaults(func=_cmd_detect)

    s = sub.add_parser("track-ball", help="link cached ball detections into one ball position per frame")
    s.add_argument("video")
    s.add_argument("--cache-dir", default="cache")
    s.set_defaults(func=_cmd_track_ball)

    s = sub.add_parser("analyze", help="run the full pipeline on a video")
    s.add_argument("video")
    s.add_argument("--calib", help="default: calib/<video stem>.json")
    s.add_argument("--out", required=True)
    s.add_argument("--ball-model", default="data/models/ball_v1.pt")
    s.add_argument("--make-model", default="models/make_model.json")
    s.add_argument("--render-range", help="render only this part, in seconds, e.g. 60:120 (implies --render)")
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
