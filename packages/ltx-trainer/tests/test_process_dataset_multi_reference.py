from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from process_dataset import _REFERENCE_VIDEO_OUTPUTS, _resolve_columns  # noqa: E402


def test_resolves_two_ordered_reference_video_columns() -> None:
    roles = _resolve_columns(
        {"video", "reference_video", "reference_video_1", "caption"},
    )

    assert roles["reference_video"] == "reference_video"
    assert roles["reference_video_1"] == "reference_video_1"
    assert _REFERENCE_VIDEO_OUTPUTS == (
        ("reference_video", "reference_latents"),
        ("reference_video_1", "reference_latents_1"),
    )
