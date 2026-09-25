"""Team assignment by jersey/bib colour clustering (Phase 5)."""

from .color import TeamModel, color_feature, fit_team_model, torso_crop, vote

__all__ = ["TeamModel", "assign_teams", "color_feature", "fit_team_model", "torso_crop", "vote"]


def assign_teams(video_path: str, player_detections, calibration, court):
    """Pipeline entry point: needs player tracks from Phase 2 (TODO); see scripts/eval_team_clustering.py."""
    raise NotImplementedError("Phase 5: wire team clustering into the pipeline once player tracks exist")
