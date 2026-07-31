from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize(
    "trained_seg",
    [False, True],
    ids=["standard", "trained"],
)
def test_cli_outputs(tmp_path, trained_seg):
    input_dir = Path(__file__).parent / "data" / "input"
    trained_seg_args = ["--trained-seg"] if trained_seg else []

    result = subprocess.run(
        [
            "dna2graph-cli",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path),
            *trained_seg_args,
            "--save-mask",
            "--save-report",
            "--save-graph",
            "--save-segmentation-rois",
            "--save-bbox-rois",
            "--save-lin-decomp-rois",
            "--clean-cache",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ERROR:" not in result.stderr

    output_dir = tmp_path / "sample_input"
    expected_outputs = [
        output_dir / "sample_input_mask.png",
        output_dir / "sample_input_report.csv",
        output_dir / "sample_input_graph.h5",
        output_dir / "sample_input_segmentation_rois.zip",
        output_dir / "sample_input_bbox_rois.zip",
        output_dir / "sample_input_lin_decomp_rois.zip",
    ]

    for output in expected_outputs:
        assert output.is_file(), (
            f"Missing expected output: {output}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
