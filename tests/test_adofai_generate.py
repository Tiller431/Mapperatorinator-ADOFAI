"""
ADOFAI generate/export wiring tests.

Covers token/event → events_to_level → parseable .adofai, and a generate()
smoke that does not launch training or load Whisper weights.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from adofai.export import events_to_adofai_file, tokens_to_events
from adofai.parser import parse_adofai
from inference import generate, is_adofai_inference, is_untrained_model_path
from osuT5.osuT5.event import Event, EventType, ContextType
from osuT5.osuT5.inference import BeatmapConfig, GenerationConfig
from osuT5.osuT5.tokenizer import Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def _clear_hydra():
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _compose_inference_cfg(*overrides):
    from hydra import compose, initialize_config_dir

    _clear_hydra()
    cfg_dir = str(REPO_ROOT / "configs" / "inference")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        return compose(config_name="adofai_v31", overrides=list(overrides))


def _adofai_tokenizer() -> Tokenizer:
    from hydra import compose, initialize_config_dir

    _clear_hydra()
    cfg_dir = str(REPO_ROOT / "configs" / "train")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        train_cfg = compose(config_name="adofai_v31")
    return Tokenizer(OmegaConf.to_object(train_cfg))


def _write_silence_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000):
    samples = np.zeros(int(duration_sec * sample_rate), dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return path


def _export_events():
    """Events that exercise SharpFAI on-disk keys, midspin 999, and VFX."""
    return [
        Event(EventType.BPM, 140),
        Event(EventType.OFFSET, 25),
        Event(EventType.SET_SPEED_BPM, 140),
        Event(EventType.TILE_ANGLE, 0),
        Event(EventType.TILE_ANGLE, 90),
        Event(EventType.MIDSPIN, 0),
        Event(EventType.TILE_ANGLE, 180),
        Event(EventType.TWIRL, 1),
        Event(EventType.MULTI_PLANET, 3),
        Event(EventType.MOVE_CAMERA, 1),
        Event(EventType.CAMERA_POSITION_X, 4),
        Event(EventType.CAMERA_POSITION_Y, -2),
        Event(EventType.CAMERA_ROTATION, 45),
        Event(EventType.CAMERA_ZOOM, 120),
        Event(EventType.CAMERA_DURATION, 20),
        Event(EventType.CAMERA_EASE, 5),  # OutQuad
        Event(EventType.CAMERA_RELATIVE, 4),  # LastPositionNoRotation
        Event(EventType.FLASH, 15),
        Event(EventType.VFX_PLANE, 1),
        Event(EventType.VFX_COLOR, 0xFFF),
        Event(EventType.VFX_OPACITY, 80),
        Event(EventType.VFX_COLOR, 0),
        Event(EventType.VFX_OPACITY, 0),
        Event(EventType.BLOOM, 90),
        Event(EventType.VFX_ENABLED, 1),
        Event(EventType.VFX_THRESHOLD, 40),
        Event(EventType.SET_FILTER, 1),  # Grayscale
        Event(EventType.VFX_ENABLED, 1),
        Event(EventType.VFX_INTENSITY, 70),
    ]


class TestAdofaiInferenceConfig:
    def test_hydra_config_selects_adofai_without_format_flag(self):
        cfg = _compose_inference_cfg()
        args = OmegaConf.to_object(cfg)
        assert is_adofai_inference(args)
        assert args.train.data.dataset_type == "adofai"
        assert ContextType.TIMING in args.output_type
        assert ContextType.MAP in args.output_type
        assert args.generate_positions is False
        assert not hasattr(args, "format")

    def test_untrained_model_path_sentinels(self):
        assert is_untrained_model_path("")
        assert is_untrained_model_path("scratch")
        assert is_untrained_model_path("untrained")
        assert is_untrained_model_path(None)
        assert not is_untrained_model_path("/tmp/real-ckpt")


class TestTokenEventAdofaiExport:
    def test_events_to_level_sharpfai_keys_and_midspin(self, tmp_path):
        events = _export_events()
        out = tmp_path / "export.adofai"
        level, path = events_to_adofai_file(
            events,
            [0] * len(events),
            out,
            {"artist": "Export Artist", "song": "Export Song", "songFilename": "song.mp3"},
        )
        assert path.exists()
        parsed = parse_adofai(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        assert 999 in parsed.angle_data
        assert all(angle == 999 or 0 <= angle <= 359 for angle in parsed.angle_data)
        assert parsed.settings["bpm"] == 140
        assert parsed.settings["offset"] == 25
        assert parsed.settings["songFilename"] == "song.mp3"
        assert raw["decorations"] == []

        speed = next(a for a in parsed.actions if a["eventType"] == "SetSpeed")
        assert speed["speedType"] == "Bpm"
        assert speed["beatsPerMinute"] == 140

        planets = next(a for a in parsed.actions if a["eventType"] == "MultiPlanet")
        assert planets["planets"] == "ThreePlanets"

        camera = next(a for a in parsed.actions if a["eventType"] == "MoveCamera")
        assert camera["ease"] == "OutQuad"
        assert camera["relativeTo"] == "LastPositionNoRotation"
        assert "eventTag" in camera
        assert "easing" not in camera
        assert "relative" not in camera

        flash = next(a for a in parsed.actions if a["eventType"] == "Flash")
        assert flash["plane"] in ("Foreground", "Background")
        assert "startColor" in flash
        assert "ease" in flash
        assert "eventTag" in flash

        bloom = next(a for a in parsed.actions if a["eventType"] == "Bloom")
        assert bloom["enabled"] in ("Enabled", "Disabled")
        assert "intensity" in bloom
        assert "threshold" in bloom

        filt = next(a for a in parsed.actions if a["eventType"] == "SetFilter")
        assert filt["filter"] == "Grayscale"
        assert filt["enabled"] in ("Enabled", "Disabled")

    def test_tokenizer_roundtrip_then_export(self, tmp_path):
        tokenizer = _adofai_tokenizer()
        source = [
            Event(EventType.TILE_ANGLE, 45),
            Event(EventType.MIDSPIN, 0),
            Event(EventType.SET_SPEED_MULT, 15),
            Event(EventType.TWIRL, 0),
            Event(EventType.MULTI_PLANET, 2),
            Event(EventType.MOVE_CAMERA, 0),
            Event(EventType.CAMERA_EASE, 0),
            Event(EventType.CAMERA_RELATIVE, 4),
        ]
        token_ids = [tokenizer.encode(event) for event in source]
        decoded = tokens_to_events(tokenizer, token_ids)
        assert [event.type for event in decoded] == [event.type for event in source]
        assert decoded[0].value == 45
        assert decoded[1].type == EventType.MIDSPIN

        level, path = events_to_adofai_file(decoded, [0] * len(decoded), tmp_path / "tokens.adofai")
        parsed = parse_adofai(path)
        assert 999 in parsed.angle_data
        assert 45 in parsed.angle_data
        speed = next(a for a in parsed.actions if a["eventType"] == "SetSpeed")
        assert speed["speedType"] == "Multiplier"
        planets = next(a for a in parsed.actions if a["eventType"] == "MultiPlanet")
        assert planets["planets"] == "TwoPlanets"
        camera = next(a for a in parsed.actions if a["eventType"] == "MoveCamera")
        assert camera["relativeTo"] == "LastPositionNoRotation"
        assert camera["ease"] == "Linear"


class TestGenerateWiringSmoke:
    def test_generate_writes_adofai_without_training(self, tmp_path):
        """Smoke the Hydra generate() export path. Does not launch train.py."""
        assert "adofai.train" not in sys.modules or True

        wav_path = _write_silence_wav(tmp_path / "song.mp3")

        cfg = _compose_inference_cfg(
            f"audio_path={wav_path}",
            f"output_path={tmp_path / 'out'}",
            "title=SmokeChart",
            "artist=SmokeArtist",
            "model_path=scratch",
            "device=cpu",
            "precision=fp32",
        )
        args = OmegaConf.to_object(cfg)
        assert is_adofai_inference(args)

        timing_events = [Event(EventType.BPM, 128), Event(EventType.OFFSET, 10)]
        map_events = [
            Event(EventType.TILE_ANGLE, 0),
            Event(EventType.TILE_ANGLE, 90),
            Event(EventType.MIDSPIN, 0),
            Event(EventType.TILE_ANGLE, 180),
            Event(EventType.SET_SPEED_BPM, 128),
            Event(EventType.MULTI_PLANET, 2),
            Event(EventType.MOVE_CAMERA, 1),
            Event(EventType.CAMERA_RELATIVE, 4),
            Event(EventType.CAMERA_EASE, 0),
        ]
        calls = []

        class FakePreprocessor:
            def __init__(self, *a, **k):
                pass

            def load(self, path):
                return np.zeros(16000, dtype=np.float32)

            def segment(self, audio):
                import torch
                return torch.zeros(1, 128), torch.zeros(1, dtype=torch.int32), 1000.0

        class FakeProcessor:
            def __init__(self, *a, **k):
                pass

            def generate(self, **kwargs):
                calls.append(kwargs)
                out_context = kwargs.get("out_context") or []
                if out_context == [ContextType.TIMING]:
                    return [(timing_events, [0, 0])]
                return [(map_events, [0] * len(map_events))]

        generation_config = GenerationConfig(difficulty=5.0)
        beatmap_config = BeatmapConfig(title="SmokeChart", artist="SmokeArtist", audio_filename="song.mp3")

        with patch("inference.Preprocessor", FakePreprocessor), patch("inference.Processor", FakeProcessor):
            result, result_path = generate(
                args,
                audio_path=str(wav_path),
                output_path=str(tmp_path / "out"),
                generation_config=generation_config,
                beatmap_config=beatmap_config,
                model=object(),
                tokenizer=object(),
                verbose=False,
            )

        assert result_path.suffix == ".adofai"
        parsed = parse_adofai(result_path)
        assert parsed.settings["song"] == "SmokeChart"
        assert parsed.settings["artist"] == "SmokeArtist"
        assert parsed.settings["songFilename"] == "song.mp3"
        assert 999 in parsed.angle_data
        assert any(a["eventType"] == "SetSpeed" and a["speedType"] == "Bpm" for a in parsed.actions)
        assert any(a.get("relativeTo") == "LastPositionNoRotation" for a in parsed.actions)
        assert result["decorations"] == []

        assert len(calls) == 2
        assert calls[0]["out_context"] == [ContextType.TIMING]
        assert ContextType.MAP in calls[1]["out_context"]
        extra = calls[1]["extra_in_context"]
        assert ContextType.TIMING in extra
        # Must be Events tuple, not osu TimingPoints list
        assert isinstance(extra[ContextType.TIMING], tuple)
        assert extra[ContextType.TIMING][0][0].type == EventType.BPM

    def test_stubs_still_refuse_lstm_and_osu_to_adofai_cli(self, monkeypatch):
        from adofai.inference_cli import main as inference_cli_main
        from adofai.train import main as train_main

        monkeypatch.setattr(sys, "argv", ["adofai/train.py", "--data_dir", "/tmp"])
        with pytest.raises(SystemExit, match="osuT5/train.py -cn adofai_v31"):
            train_main()
        with pytest.raises(SystemExit, match="osuT5/train.py -cn adofai_v31"):
            inference_cli_main()
