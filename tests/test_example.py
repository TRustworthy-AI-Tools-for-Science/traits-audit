from traits_audit._example import _ensure_cal_demo_dir


def test_ensure_cal_demo_dir_creates_parents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fig_dir = _ensure_cal_demo_dir()

    assert fig_dir == tmp_path / "_results" / "cal_demo"
    assert fig_dir.is_dir()
