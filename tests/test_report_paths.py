"""Exporters must reject source/output collisions before touching run receipts."""
from pathlib import Path
import sys

import pytest

from scripts import build_adaptive_report, build_pressure_report, build_report
from scripts.report_paths import validate_report_output


def invoke(exporter, source, output, monkeypatch):
    if exporter == "adaptive":
        return build_adaptive_report.build(source, output, no_plots=True)
    if exporter == "pressure":
        return build_pressure_report.build(source, output)
    monkeypatch.setattr(sys, "argv", ["build_report.py", "--run-root", str(source),
                                     "--output-dir", str(output), "--no-matrix", "--no-plots"])
    return build_report.main()


@pytest.mark.parametrize("exporter", ["adaptive", "pressure", "fixed"])
@pytest.mark.parametrize("relationship", ["same", "nested", "ancestor", "symlink"])
def test_all_exporters_reject_resolved_source_overlap_before_any_writes(tmp_path, monkeypatch, exporter, relationship):
    source = tmp_path / "run"
    source.mkdir()
    summary = source / "summary.json"
    summary.write_bytes(b'{"status":"original immutable receipt"}\n')
    if relationship == "same":
        output = source
    elif relationship == "nested":
        output = source / "new-report"
    elif relationship == "ancestor":
        output = tmp_path
    else:
        output = tmp_path / "linked-report"
        output.symlink_to(source, target_is_directory=True)
    before = summary.read_bytes()
    with pytest.raises(ValueError, match="must not overlap"):
        invoke(exporter, source, output, monkeypatch)
    assert summary.read_bytes() == before
    assert list(source.iterdir()) == [summary]


def test_existing_artifact_symlink_cannot_redirect_writes_into_a_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "report"
    source.mkdir()
    output.mkdir()
    (output / "artifacts").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        validate_report_output(output, source)
    assert not list(source.iterdir())


def test_fixed_exporter_also_protects_judge_and_data_inputs(tmp_path, monkeypatch):
    for option in ("--judge-dir", "--data-dir"):
        source = tmp_path / option[2:]
        source.mkdir()
        marker = source / "summary.json"
        marker.write_bytes(b"original input")
        monkeypatch.setattr(sys, "argv", ["build_report.py", option, str(source),
                                         "--output-dir", str(source), "--no-matrix", "--no-plots"])
        with pytest.raises(ValueError, match="must not overlap"):
            build_report.main()
        assert marker.read_bytes() == b"original input"


def test_separate_sibling_report_path_is_allowed(tmp_path):
    output = tmp_path / "reports" / "result"
    validate_report_output(output, tmp_path / "runs" / "result", tmp_path / "data")
    assert not output.exists()
