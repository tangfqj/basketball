# Shot Labeling Guide

Consistent labels matter more than fast labels: evaluation (requirements §7.3) is only as good as
the ground truth. Follow these conventions for every video.

## Running the tool

```bash
uv run hoopstats label data/outdoor/IMG_0104.MOV
```

- Opens `http://127.0.0.1:8765/` in your browser; labels are saved automatically to
  `labels/IMG_0104.csv` (+ `IMG_0104.meta.json`) after every change.
- The first run on a 4K video creates a 720p working copy in `cache/` (one-time, a few minutes).
- Stop with Ctrl+C in the terminal. Re-running the command continues where you left off.
- Commit the label files to git when a video is done.

## Workflow per video

1. Write down the two team colors (A and B) in the sidebar. Keep A/B fixed for the whole video.
2. Watch at 1× or 1.5× (`]`). When a shot happens, pause, step back with `←` to the **release frame**
   (the last frame the ball touches the shooter's hand), press `S`.
3. Set team (`A`/`B`), zone (`1` inside arc, `2` beyond arc, `3` free throw), result (`M`/`X`).
4. Continue. Rows with a red `?` are incomplete; the counter in the bottom bar must be 0 when done.
5. Final pass: `,` / `.` jump through all shots to double-check them.

## What counts (requirements §5)

| Situation | Label |
|---|---|
| Jump shot, layup, dunk, hook, floater | shot |
| Tip-in / put-back (ball tipped toward the rim) | shot, release = the tip |
| Blocked shot | shot, missed |
| Airball | shot, missed |
| Pass, lob pass that is caught, loose ball | **not** a shot |
| Shot while fouled | shot, labelled by physical outcome |
| Shot after the whistle / during a dead ball (warm-up, practice shot) | **not** a shot; add nothing |

**Zone:** decided by the shooter's feet at the last ground contact before release. A foot on the line
is inside the arc. If feet are hidden, use your best judgement and write `feet hidden` in the note.

**Made:** the ball passes down through the rim. A rim-out is missed.

**Unsure?** Label your best guess and write a short note (e.g. `unclear make`, `release hidden`). Notes
are kept in the CSV and help error analysis later.
