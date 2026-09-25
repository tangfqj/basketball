from hoopstats.labeling.server import read_labels, write_labels
from hoopstats.outputs import read_events_csv


def test_label_file_roundtrip_and_compatible_with_evaluation(tmp_path):
    p = tmp_path / "x.csv"
    rows = [
        {"timestamp_s": "12.5", "frame": "375", "team": "B", "zone": "beyond_arc", "made": "1", "note": ""},
        {"timestamp_s": "3.0", "frame": "90", "team": "A", "zone": "inside_arc", "made": "0", "note": "blocked"},
    ]
    write_labels(p, rows)
    back = read_labels(p)
    assert [r["event_id"] for r in back] == ["0", "1"]          # re-numbered in time order
    assert back[0]["note"] == "blocked"
    evs = read_events_csv(p)
    assert evs[1].made and evs[1].zone.value == "beyond_arc"
