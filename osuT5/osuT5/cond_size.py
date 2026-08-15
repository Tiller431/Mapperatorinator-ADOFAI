"""Encoder cond width from the embeds that are actually concatenated.

Whisper v2 yamls ship ``cond_size: 384`` for difficulty+mapper+song_position
(3 × 128). ADOFAI turns mapper/style/year/song-position off and keeps only
the difficulty RBF, so the real concat width is ``cond_dim`` (128).
``n_mels + cond_size`` must match that concat or Whisper conv1 is 464 vs 208.
"""

from __future__ import annotations


def cond_size_from_embeds(
    *,
    do_difficulty_embed: bool = False,
    do_mapper_embed: bool = False,
    do_song_position_embed: bool = False,
    do_style_embed: bool = False,
    cond_dim: int = 128,
    style_dim: int = 0,
) -> int:
    size = 0
    if do_style_embed:
        size += style_dim
    if do_difficulty_embed:
        size += cond_dim
    if do_mapper_embed:
        size += cond_dim
    if do_song_position_embed:
        size += cond_dim
    return size
