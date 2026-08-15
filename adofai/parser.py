"""
Parser for ADOFAI (.adofai) level files.

Handles reading and writing ADOFAI chart files, which are JSON-like with quirks
(trailing commas, non-standard formatting). Supports both angleData (modern) and
pathData (legacy) formats.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import dataclasses

try:
    import json5
except ImportError:  # pragma: no cover - optional, cleanup is the fallback
    json5 = None

# Illegal in JSON strings; keep tab/LF/CR. Workshop files sometimes embed these.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Unquoted object keys (JSON5). Only after { [ or , so values stay intact.
_UNQUOTED_KEY = re.compile(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):')


# Legacy pathData character mapping to angles
PATH_DATA_MAP = {
    "R": 0, "p": 15, "J": 30, "E": 45, "T": 60, "o": 75,
    "U": 90, "q": 105, "G": 120, "Q": 135, "H": 150, "W": 165,
    "L": 180, "x": 195, "N": 210, "Z": 225, "F": 240, "V": 255,
    "D": 270, "Y": 285, "B": 300, "C": 315, "M": 330, "A": 345,
    "!": 999  # Midspin
}


@dataclasses.dataclass
class AdofaiLevel:
    """
    Represents a complete ADOFAI level.
    
    Attributes:
        settings: Global level settings (bpm, offset, audio file, metadata, etc.)
        angle_data: List of tile angles (0-359 degrees, or 999 for midspin)
        actions: List of event dictionaries (floor-indexed events like SetSpeed, Twirl, etc.)
        decorations: List of decoration dictionaries (optional, v1 ignores these)
    """
    settings: dict[str, Any]
    angle_data: list[int]
    actions: list[dict[str, Any]]
    decorations: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "settings": self.settings,
            "angleData": self.angle_data,
            "actions": self.actions,
            "decorations": self.decorations,
        }


_STRING_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def _escape_raw_breaks_in_strings(content: str) -> str:
    """Turn raw C0 controls inside quotes into JSON escapes (levelDesc/author).

    Python json.loads(strict=True) rejects literal TAB/LF/CR in strings.
    3469661239__main embeds a raw newline then a raw tab in levelDesc.
    """
    out: list[str] = []
    in_string = False
    escape = False
    quote = ""
    for ch in content:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == quote:
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            code = ord(ch)
            if code < 32:
                out.append(f"\\u{code:04x}")
                continue
            out.append(ch)
            continue
        if ch in "\"'":
            in_string = True
            quote = ch
        out.append(ch)
    return "".join(out)


def _mask_quoted_strings(content: str) -> tuple[str, list[str]]:
    """Replace quoted strings with placeholders so regexes cannot see inside them.

    `_UNQUOTED_KEY` otherwise matches `,https:` inside artistLinks URL lists
    (2723436598 and four more) and rewrites them to `,"https"://...`.
    """
    held: list[str] = []
    out: list[str] = []
    in_string = False
    escape = False
    quote = ""
    buf: list[str] = []
    for ch in content:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
                held.append("".join(buf))
                out.append(f"\x00{len(held) - 1}\x00")
                buf = []
            continue
        if ch in "\"'":
            in_string = True
            quote = ch
            buf = [ch]
            continue
        out.append(ch)
    if buf:
        held.append("".join(buf))
        out.append(f"\x00{len(held) - 1}\x00")
    return "".join(out), held


def _restore_quoted_strings(content: str, held: list[str]) -> str:
    return _STRING_PLACEHOLDER.sub(lambda m: held[int(m.group(1))], content)


def _clean_json_string(content: str) -> str:
    """Make Workshop .adofai text acceptable to json.loads / json5.loads.

    Real files mix trailing commas, unquoted keys, raw control chars, raw
    newlines/tabs in strings, double commas, and missing commas between values
    (2346220412__main line 449/450; `]\\n\"decorations\"`).
    artistLinks often stores comma-joined https URLs in one string; key-quoting
    must not run inside that string. JSON5 covers trailing commas / unquoted
    keys / comments; the rest needs this cleanup. Cleanup must stay valid
    without json5 (trainer CPU encode).
    """
    content = _CONTROL_CHARS.sub("", content)
    content = _escape_raw_breaks_in_strings(content)
    masked, held = _mask_quoted_strings(content)
    masked = _UNQUOTED_KEY.sub(r'\1"\2"\3:', masked)
    # Insert missing commas between adjacent values: `}\n{`, `]\n"key"`,
    # `"url1" "url2"`, `"url" [`  (placeholders stand in for quoted strings)
    masked = re.sub(r'([}\]])(\s*)([{\[\x00])', r"\1,\2\3", masked)
    masked = re.sub(r'(\x00\d+\x00)(\s*)(\x00)', r"\1,\2\3", masked)
    masked = re.sub(r'(\x00\d+\x00)(\s*)([{\[])', r"\1,\2\3", masked)
    # Double commas: `"difficulty": 1, ,` / `4,,`
    masked = re.sub(r",(\s*),+", r",\1", masked)
    # Trailing commas before } or ]
    masked = re.sub(r",(\s*[}\]])", r"\1", masked)
    return _restore_quoted_strings(masked, held)


def _loads_adofai_json(content: str) -> Any:
    """Parse ADOFAI JSON with json, then json5, then the same on cleaned text."""
    cleaned = _clean_json_string(content)
    last_error: Exception | None = None
    for candidate in (content, cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        if json5 is not None:
            try:
                return json5.loads(candidate)
            except Exception as exc:  # json5 raises ValueError / JSON5DecodeError
                last_error = exc
    raise ValueError(f"Failed to parse ADOFAI file as JSON: {last_error}")


def parse_adofai(file_path: str | Path) -> AdofaiLevel:
    """
    Parse an ADOFAI level file.
    
    Supports both angleData (modern) and pathData (legacy) formats.
    Handles Workshop JSON quirks: UTF-8 BOM, trailing commas, unquoted keys,
    missing commas between objects/strings, raw control chars / newlines / tabs
    in strings, and comma-joined https URLs in artistLinks (json5 + cleanup).
    
    Args:
        file_path: Path to .adofai file
        
    Returns:
        AdofaiLevel object with parsed data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file content is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ADOFAI file not found: {file_path}")
    
    # Read file content with utf-8-sig to strip UTF-8 BOM if present
    # Many Workshop .adofai files start with BOM
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    data = _loads_adofai_json(content)
    
    # Extract settings (required)
    if "settings" not in data:
        raise ValueError("ADOFAI file missing 'settings' field")
    settings = data["settings"]
    
    # Extract angle data (prefer angleData, fall back to pathData)
    angle_data = []
    if "angleData" in data:
        angle_data = data["angleData"]
    elif "pathData" in data:
        # Convert legacy pathData to angleData
        path_data = data["pathData"]
        angle_data = [PATH_DATA_MAP.get(char, 0) for char in path_data]
    else:
        raise ValueError("ADOFAI file missing both 'angleData' and 'pathData'")
    
    # Extract actions (default to empty if not present)
    actions = data.get("actions", [])
    
    # Extract decorations (default to empty if not present)
    decorations = data.get("decorations", [])
    
    return AdofaiLevel(
        settings=settings,
        angle_data=angle_data,
        actions=actions,
        decorations=decorations
    )


def write_adofai(level: AdofaiLevel, file_path: str | Path, indent: int = 4) -> None:
    """
    Write an ADOFAI level to file.
    
    Always writes angleData format (modern standard).
    Produces valid JSON with proper formatting.
    
    Args:
        level: AdofaiLevel object to write
        file_path: Output path for .adofai file
        indent: JSON indentation (default 4 spaces)
    """
    file_path = Path(file_path)
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    data = level.to_dict()
    
    # Write JSON with indentation
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def path_data_to_angle_data(path_data: str) -> list[int]:
    """
    Convert legacy pathData string to angleData list.
    
    Args:
        path_data: String like "RRULDR!" (legacy format)
        
    Returns:
        List of angles like [0, 0, 90, 180, 270, 0, 999]
    """
    return [PATH_DATA_MAP.get(char, 0) for char in path_data]


def angle_data_to_path_data(angle_data: list[int]) -> str:
    """
    Convert angleData list to legacy pathData string.
    
    Note: Only exact angle matches are converted; intermediate angles
    default to 'R' (0 degrees).
    
    Args:
        angle_data: List of angles
        
    Returns:
        pathData string
    """
    # Reverse lookup
    angle_to_char = {v: k for k, v in PATH_DATA_MAP.items()}
    return ''.join(angle_to_char.get(angle, 'R') for angle in angle_data)
