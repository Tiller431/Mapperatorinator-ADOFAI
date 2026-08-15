from __future__ import annotations

import dataclasses
from enum import Enum


class EventType(Enum):
    TIME_SHIFT = "t"
    SNAPPING = "snap"
    DISTANCE = "dist"
    NEW_COMBO = "new_combo"
    HITSOUND = "hitsound"
    VOLUME = "volume"
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    BEZIER_ANCHOR = "bezier_anchor"
    PERFECT_ANCHOR = "perfect_anchor"
    CATMULL_ANCHOR = "catmull_anchor"
    RED_ANCHOR = "red_anchor"
    LAST_ANCHOR = "last_anchor"
    SLIDER_END = "slider_end"
    BEAT = "beat"
    MEASURE = "measure"
    TIMING_POINT = "timing_point"
    GAMEMODE = "gamemode"
    STYLE = "style"
    DIFFICULTY = "difficulty"
    MAPPER = "mapper"
    CS = "cs"
    YEAR = "year"
    HITSOUNDED = "hitsounded"
    SONG_LENGTH = "song_length"
    SONG_POSITION = "song_position"
    GLOBAL_SV = "global_sv"
    MANIA_KEYCOUNT = "keycount"
    HOLD_NOTE_RATIO = "hold_note_ratio"
    SCROLL_SPEED_RATIO = "scroll_speed_ratio"
    DESCRIPTOR = "descriptor"
    POS_X = "pos_x"
    POS_Y = "pos_y"
    POS = "pos"
    KIAI = "kiai"
    MANIA_COLUMN = "column"
    HOLD_NOTE = "hold_note"
    HOLD_NOTE_END = "hold_note_end"
    SCROLL_SPEED_CHANGE = "scroll_speed_change"
    SCROLL_SPEED = "scroll_speed"
    DRUMROLL = "drumroll"
    DRUMROLL_END = "drumroll_end"
    DENDEN = "denden"
    DENDEN_END = "denden_end"
    CONTROL = "control"
    
    # ADOFAI event types
    TILE_ANGLE = "adofai_angle"
    MIDSPIN = "adofai_midspin"
    SET_SPEED_BPM = "adofai_speed_bpm"
    SET_SPEED_MULT = "adofai_speed_mult"
    PAUSE = "adofai_pause"
    HOLD = "adofai_hold"
    TWIRL = "adofai_twirl"
    MULTI_PLANET = "adofai_multiplanet"
    CHECKPOINT = "adofai_checkpoint"
    AUTO_PLAY_TILES = "adofai_autoplay"
    SET_PLANET_ROTATION = "adofai_planet_rot"
    FREE_ROAM = "adofai_freeroam"
    FREE_ROAM_TWIRL = "adofai_freeroam_twirl"
    FREE_ROAM_REMOVE = "adofai_freeroam_remove"
    SCALE_MARGIN = "adofai_scale_margin"
    SCALE_RADIUS = "adofai_scale_radius"
    MULTITAP = "adofai_multitap"
    HIDE = "adofai_hide"
    KILL_PLAYER = "adofai_kill_player"
    POSITION_TRACK = "adofai_pos_track"
    MOVE_TRACK = "adofai_move_track"
    COLOR_TRACK = "adofai_color_track"
    ANIMATE_TRACK = "adofai_anim_track"
    MOVE_CAMERA = "adofai_move_camera"
    SET_HITSOUND = "adofai_set_hitsound"
    PLAY_SOUND = "adofai_play_sound"
    SET_HOLD_SOUND = "adofai_set_hold_sound"
    REPEAT_EVENTS = "adofai_repeat_events"
    SET_CONDITIONAL_EVENTS = "adofai_set_cond"
    SET_INPUT_EVENT = "adofai_set_input"
    FLASH = "adofai_flash"
    BLOOM = "adofai_bloom"
    SHAKE_SCREEN = "adofai_shake_screen"
    SET_FILTER = "adofai_set_filter"
    BPM = "adofai_bpm"
    OFFSET = "adofai_offset"


class ContextType(Enum):
    NONE = "none"
    TIMING = "timing"
    NO_HS = "no_hs"
    GD = "gd"
    MAP = "map"
    KIAI = "kiai"
    SV = "sv"


@dataclasses.dataclass
class EventRange:
    type: EventType
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    type: EventType
    value: int = 0

    def __repr__(self) -> str:
        return f"{self.type.value}{self.value}"

    def __str__(self) -> str:
        return f"{self.type.value}{self.value}"
