"""Utilities for extracting color priors from metadata and normalizing color terms."""
import json
import re
from pathlib import Path
from typing import Optional, List, Dict

RESOURCES_DIR = Path(__file__).parent.parent.parent.parent / "resources"

# Caches
_alias_cache: Optional[Dict[str, str]] = None
_detail_colors_cache: Optional[List[str]] = None

# Material-to-color priors (dominant material -> likely color)
MATERIAL_COLOR_PRIORS: Dict[str, str] = {
    "MANGO WOOD": "Natural",
    "RECYCLED WOOD": "Natural",
    "ELM WOOD": "Natural",
    "ACACIA WOOD": "Natural",
    "TEAK WOOD": "Natural",
    "TEAK": "Natural",
    "PINE WOOD": "Natural",
    "BAMBOO": "Natural",
    "RATTAN": "Natural",
    "SEAGRASS": "Natural",
    "JUTE": "Natural",
    "WATER HYACINTH": "Natural",
    "DRIFTWOOD": "Natural",
    "IRON": "Black",
    "ALUMINIUM": "Silver",
    "ALUMINUM": "Silver",
    "STAINLESS STEEL": "Silver",
    "BRASS": "Brass",
    "COPPER": "Copper",
    # GLASS intentionally omitted — can be transparent OR colored; let the LLM decide from the image.
    "MARBLE": "White",
}


def _load_aliases() -> Dict[str, str]:
    """Load color aliases from JSON."""
    global _alias_cache
    if _alias_cache is None:
        filepath = RESOURCES_DIR / "color_aliases.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                _alias_cache = json.load(f)
        else:
            _alias_cache = {}
    return _alias_cache


def _load_detail_colors() -> List[str]:
    """Load canonical detail color names."""
    global _detail_colors_cache
    if _detail_colors_cache is None:
        filepath = RESOURCES_DIR / "color_detail_values.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            _detail_colors_cache = json.load(f)
    return _detail_colors_cache


def normalize_color_name(raw_color: Optional[str]) -> Optional[str]:
    """
    Normalize a raw color string to a canonical detail color name.
    
    Tries in order:
    1. Exact match against canonical list
    2. Case-insensitive match
    3. Alias lookup (lowercase)
    4. Returns None if no match
    """
    if not raw_color or raw_color.lower() in ("null", "none", ""):
        return None
    
    detail_colors = _load_detail_colors()
    aliases = _load_aliases()
    
    # Strip whitespace
    cleaned = raw_color.strip()
    
    # 1. Exact match
    if cleaned in detail_colors:
        return cleaned
    
    # 2. Case-insensitive match
    lower = cleaned.lower()
    for canonical in detail_colors:
        if canonical.lower() == lower:
            return canonical
    
    # 3. Alias lookup
    alias_result = aliases.get(lower)
    if alias_result and alias_result in detail_colors:
        return alias_result
    
    # 4. Partial match: check if any canonical name is contained in the raw string
    #    (e.g. "dark brown color" -> "Dark brown")
    for canonical in sorted(detail_colors, key=len, reverse=True):
        if canonical.lower() in lower:
            return canonical
    
    return None


def extract_text_color_hints(description: Optional[str]) -> List[str]:
    """
    Extract color hints from supplier_reference_description text.
    
    Scans text for known color names and aliases, returns matches
    in order of appearance.
    """
    if not description:
        return []
    
    detail_colors = _load_detail_colors()
    aliases = _load_aliases()
    
    text_lower = description.lower()
    found = []
    seen = set()
    
    # Check canonical names (longest first to avoid partial matches)
    for canonical in sorted(detail_colors, key=len, reverse=True):
        if canonical == "Multi":
            continue
        pattern = r'\b' + re.escape(canonical.lower()) + r'\b'
        if re.search(pattern, text_lower) and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    
    # Check aliases
    for alias, canonical in sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text_lower) and canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
    
    return found


def extract_material_color_prior(materials: Optional[str]) -> Optional[str]:
    """
    Extract the most likely color prior from a materials string.
    
    Parses format like "MANGO WOOD (70.00%), IRON (30.00%)" and returns
    the color prior for the highest-percentage material, if one exists.
    """
    if not materials:
        return None
    
    # Parse "MATERIAL (XX.XX%)" pairs
    pattern = r'([A-Z][A-Z\s]+?)\s*\((\d+(?:\.\d+)?)%\)'
    matches = re.findall(pattern, materials.upper())
    
    if not matches:
        return None
    
    # Sort by percentage descending
    parsed = [(name.strip(), float(pct)) for name, pct in matches]
    parsed.sort(key=lambda x: x[1], reverse=True)
    
    # Return prior for highest-percentage material that has a mapping
    for material_name, pct in parsed:
        if pct < 20.0:
            break  # Skip minor materials
        prior = MATERIAL_COLOR_PRIORS.get(material_name)
        if prior:
            return prior
    
    return None


def build_prior_hints_block(
    supplier_reference_description: Optional[str] = None,
    materials: Optional[str] = None
) -> str:
    """
    Build a formatted PRIOR HINTS block for injection into prompts.
    Returns empty string if no useful hints are found.
    """
    parts = []
    
    text_hints = extract_text_color_hints(supplier_reference_description)
    material_prior = extract_material_color_prior(materials)
    
    if supplier_reference_description:
        parts.append(f"Product description: {supplier_reference_description}")
    if materials:
        parts.append(f"Materials: {materials}")
    if text_hints:
        parts.append(f"Colors mentioned in text: {', '.join(text_hints)}")
    if material_prior:
        parts.append(f"Material suggests dominant color: {material_prior}")
    
    if not parts:
        return ""
    
    return "\nPRIOR HINTS (use to disambiguate when image is ambiguous; image takes precedence):\n" + "\n".join(f"- {p}" for p in parts) + "\n"
