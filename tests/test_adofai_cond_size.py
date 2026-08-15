"""ADOFAI train configs must use the difficulty-only encoder cond width.

whisper_*_v2.yaml ships cond_size 384 (diff+mapper+song_position).
ADOFAI turns mapper/style/year/song-position off, so concat is 128.
Whisper conv1 is n_mels+cond_size: 80+384=464 vs 80+128=208.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _model_section_cond_size(yaml_path: Path) -> int | None:
    text = yaml_path.read_text(encoding="utf-8")
    in_model = False
    for line in text.splitlines():
        if line.startswith("model:"):
            in_model = True
            continue
        if in_model:
            if line and not line[0].isspace() and not line.startswith("#"):
                in_model = False
                continue
            stripped = line.strip()
            if stripped.startswith("cond_size:"):
                return int(stripped.split(":", 1)[1].strip().split()[0])
    return None


def test_cond_size_helper_difficulty_only_is_128():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "osuT5"))
    from osuT5.cond_size import cond_size_from_embeds

    assert (
        cond_size_from_embeds(
            do_difficulty_embed=True,
            do_mapper_embed=False,
            do_song_position_embed=False,
            do_style_embed=False,
            cond_dim=128,
        )
        == 128
    )
    assert (
        cond_size_from_embeds(
            do_difficulty_embed=True,
            do_mapper_embed=True,
            do_song_position_embed=True,
            do_style_embed=False,
            cond_dim=128,
        )
        == 384
    )


def test_adofai_v31_yaml_cond_size_is_difficulty_only():
    path = REPO_ROOT / "configs" / "train" / "adofai_v31.yaml"
    assert _model_section_cond_size(path) == 128


def test_adofai_whisper_tiny_yaml_cond_size_is_difficulty_only():
    path = REPO_ROOT / "configs" / "train" / "adofai_whisper_tiny.yaml"
    assert _model_section_cond_size(path) == 128


if __name__ == "__main__":
    test_cond_size_helper_difficulty_only_is_128()
    print("cond_size helper: ok")
    test_adofai_v31_yaml_cond_size_is_difficulty_only()
    print("adofai_v31 cond_size: ok")
    test_adofai_whisper_tiny_yaml_cond_size_is_difficulty_only()
    print("adofai_whisper_tiny cond_size: ok")
