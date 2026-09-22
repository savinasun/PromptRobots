"""Trial directories are named for their outcome, and errored runs are parked in runs/recycled."""
import re

from utils.logging_utils import (MAP_FILES, RECYCLED_DIR, SUCCEEDED_DIR, TrialLogger,
                                     rebuild_map)

STAMP = re.compile(r"^\d{8}_\d{6}_(success|fail)(_\d+)?$")


def _run(root, status):
    logger = TrialLogger(str(root), "pick up the block")
    born = logger.dir.name
    logger.finish({"status": status, "log_dir": str(logger.dir)})
    return born, logger.dir


def test_directory_is_born_pessimistic(tmp_path):
    """A killed run runs none of our code, so the name it was born with is the name it keeps."""
    logger = TrialLogger(str(tmp_path), "pick up the block")
    assert logger.dir.name.endswith("_fail") and STAMP.match(logger.dir.name)


def test_done_is_promoted_to_success(tmp_path):
    born, final = _run(tmp_path, "done")
    assert born.endswith("_fail") and final.name.endswith("_success")
    assert final.parent == tmp_path / SUCCEEDED_DIR and STAMP.match(final.name)


def test_every_non_success_is_recycled(tmp_path):
    """Only `done` is success: a deliberate give_up did not accomplish the goal either."""
    for status in ("give_up", "too_many_rejections", "waypoints_exhausted", "timeout", "aborted", "error"):
        _, final = _run(tmp_path, status)
        assert final.name.endswith("_fail") or re.search(r"_fail_\d+$", final.name)
        assert final.parent == tmp_path / RECYCLED_DIR, f"{status} belongs in recycled"
        assert (final / "summary.json").exists()


def test_finished_runs_leave_the_log_root_empty(tmp_path):
    """Every finished run is filed; only a killed run is ever left loose in the log root."""
    for status in ("done", "give_up", "error", "done"):
        _run(tmp_path, status)
    assert sorted(d.name for d in tmp_path.iterdir() if d.is_dir()) == [RECYCLED_DIR, SUCCEEDED_DIR]
    assert sorted(f.name for f in tmp_path.iterdir() if f.is_file()) == sorted(MAP_FILES.values())
    # STAMP, not endswith: runs colliding within one second are suffixed, e.g. `..._success_2`.
    filed = {d.name: STAMP.match(d.name).group(1) for p in (SUCCEEDED_DIR, RECYCLED_DIR)
             for d in (tmp_path / p).iterdir()}
    assert sorted(filed.values()) == ["fail", "fail", "success", "success"]

    killed = TrialLogger(str(tmp_path), "killed")      # never calls finish()
    assert killed.dir.parent == tmp_path and killed.dir.name.endswith("_fail")


def test_summary_records_the_final_location(tmp_path):
    import json

    _, final = _run(tmp_path, "error")
    assert json.loads((final / "summary.json").read_text())["log_dir"] == str(final)


def test_collisions_within_one_second_are_suffixed(tmp_path):
    names = {_run(tmp_path, "done")[1].name for _ in range(3)}
    assert len(names) == 3


def test_explicit_name_is_left_alone(tmp_path):
    logger = TrialLogger(str(tmp_path), "pick up the block", name="calibration")
    logger.finish({"status": "error"})
    assert logger.dir == tmp_path / "calibration"


def _index(root, folder):
    import json

    return json.loads((root / MAP_FILES[folder]).read_text())


def test_filing_updates_the_folder_index(tmp_path):
    _, ok = _run(tmp_path, "done")
    _, bad = _run(tmp_path, "give_up")
    assert _index(tmp_path, SUCCEEDED_DIR) == [ok.name]
    assert _index(tmp_path, RECYCLED_DIR) == [bad.name]


def test_index_appends_in_filing_order(tmp_path):
    names = [_run(tmp_path, "done")[1].name for _ in range(3)]
    assert _index(tmp_path, SUCCEEDED_DIR) == names


def test_a_corrupt_index_does_not_fail_the_trial(tmp_path):
    (tmp_path / MAP_FILES[SUCCEEDED_DIR]).write_text("{ not json")
    _, ok = _run(tmp_path, "done")
    assert _index(tmp_path, SUCCEEDED_DIR) == [ok.name]


def test_rebuild_map_restores_a_deleted_index(tmp_path):
    names = sorted(_run(tmp_path, "error")[1].name for _ in range(2))
    (tmp_path / MAP_FILES[RECYCLED_DIR]).unlink()
    assert rebuild_map(tmp_path, RECYCLED_DIR) == names
    assert _index(tmp_path, RECYCLED_DIR) == names
