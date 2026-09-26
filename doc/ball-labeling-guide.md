# Ball Box Review Guide

574 frames from IMG_0104–0108 were pre-annotated by the pretrained detector. Your job is to accept,
correct or reject each proposal. Expect ~4 s per frame (~40 min in total); you can stop anytime —
every decision is saved immediately to `labels/ball/<video>.json`.

```bash
uv run hoopstats ball-review
```

## What you see

- **Big view**: a zoomed part of the 4K frame, centred on the model's best guess (or the rim).
- **Yellow dashed box** = the model's proposal (with its confidence). **Green box** = the box that will be saved.
- **Whole frame** (right): click anywhere on it to look there — use it when the ball is elsewhere.

## Decisions

| Situation | Action |
|---|---|
| Green/yellow box is tight on the ball | **Enter** |
| Box is on something else, or loose | find the ball, **drag** a tight box (or **click** its centre for a standard-size box), **Enter** |
| Wrong proposal, but another yellow box is the ball | **Tab** to it, **Enter** |
| Ball not visible: hidden behind a player/backboard, out of frame, or less than half visible | **N** |
| Can't tell | **S** (skipped frames are left out of training) |

## Rules

- Box the ball **tightly**, including when a player holds it or it is in the net.
- A motion-blurred ball: box the visible (blurred) ball shape.
- Only **the game ball**. If there is a second ball (e.g. warm-up at the side), leave it unboxed and note
  nothing — but prefer **S** if you are unsure which one is the game ball.
- Frames are ordered by video and time; `F` jumps to the first undecided frame.

When done, tell Claude; the training set is then built with `uv run hoopstats ball-dataset`
(→ `data/ball_dataset.zip`, ~0.5 GB) and trained in Colab with `notebooks/train_ball_detector.ipynb`.
