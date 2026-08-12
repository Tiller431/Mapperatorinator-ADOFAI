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


def _clean_json_string(content: str) -> str:
    """
    Clean up non-standard JSON formatting common in .adofai files.
    
    Handles:
    - Trailing commas before closing brackets/braces
    - Multiple trailing commas
    """
    # Remove trailing commas before closing brackets/braces
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return content


def parse_adofai(file_path: str | Path) -> AdofaiLevel:
    """
    Parse an ADOFAI level file.
    
    Supports both angleData (modern) and pathData (legacy) formats.
    Handles non-standard JSON formatting (trailing commas, etc.).
    
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
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean up non-standard JSON
    content = _clean_json_string(content)
    
    # Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse ADOFAI file as JSON: {e}")
    
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
