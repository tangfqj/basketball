from hoopstats.labeling.shooter_review import flag_reason, load_decisions, save_decisions, team_mapping

PLAYERS = {"1": {"number": 10, "team": "A"}, "3": {"number": 6, "team": "B"}, "4": {"number": 4, "team": "A"}}


def item(label_team, cands):
    return {"label_team": label_team, "proposal": cands[0][0] if cands else None, "candidates": cands}


def test_team_mapping_follows_majority():
    items = [item("B", [[1, 0.1, 9]]), item("B", [[4, 0.0, 9]]), item("A", [[3, 0.2, 9]])]
    assert team_mapping(items, PLAYERS) == {"A": "B", "B": "A"}
    assert team_mapping([item("A", [[1, 0.1, 9]])], PLAYERS) == {"A": "A", "B": "B"}


def test_flag_reason():
    m = {"A": "A", "B": "B"}
    assert flag_reason(item("A", []), PLAYERS, m) == "no proposal"
    assert flag_reason(item("B", [[1, 0.1, 9]]), PLAYERS, m) == "team differs from shot label"
    assert flag_reason(item("A", [[1, 0.1, 9], [4, 0.15, 9]]), PLAYERS, m) == "close second candidate"
    assert flag_reason(item("A", [[1, 0.1, 9], [4, 0.9, 9]]), PLAYERS, m) is None


def test_decisions_roundtrip(tmp_path):
    save_decisions("V", {"10": {"status": "none"}, "2": {"status": "ok", "track": 1}}, tmp_path)
    d = load_decisions("V", tmp_path)
    assert list(d) == ["2", "10"] and d["2"]["track"] == 1
    assert load_decisions("missing", tmp_path) == {}
