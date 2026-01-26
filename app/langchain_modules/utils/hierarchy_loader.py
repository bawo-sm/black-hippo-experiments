"""Utility functions for loading and filtering values using hierarchy mappings."""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Path to hierarchy mappings file
RESOURCES_DIR = Path(__file__).parent.parent.parent.parent / "resources"
HIERARCHY_MAPPINGS_FILE = RESOURCES_DIR / "hierarchy_mappings.json"

# Cache for loaded hierarchy mappings
_hierarchy_cache: Optional[Dict[str, Any]] = None


def _load_hierarchy_mappings() -> Dict[str, Any]:
    """Load hierarchy mappings from JSON file with caching."""
    global _hierarchy_cache
    if _hierarchy_cache is None:
        if not HIERARCHY_MAPPINGS_FILE.exists():
            raise FileNotFoundError(
                f"Hierarchy mappings file not found: {HIERARCHY_MAPPINGS_FILE}"
            )
        with open(HIERARCHY_MAPPINGS_FILE, 'r', encoding='utf-8') as f:
            _hierarchy_cache = json.load(f)
    return _hierarchy_cache


def get_valid_sub_values(main_value: str) -> List[str]:
    """
    Get valid sub values for a given main category.
    
    Args:
        main_value: The main category value
        
    Returns:
        List of valid sub values for the main category.
        If main_value is "Unspecified" or not found, returns empty list.
    """
    if main_value == "Unspecified":
        return []
    
    hierarchy = _load_hierarchy_mappings()
    main_to_sub = hierarchy.get("main_to_sub", {})
    return main_to_sub.get(main_value, [])


def get_valid_detail_values(sub_value: str) -> List[str]:
    """
    Get valid detail values for a given sub category.
    
    Args:
        sub_value: The sub category value
        
    Returns:
        List of valid detail values for the sub category.
        If sub_value is "Unspecified" or not found, returns empty list.
    """
    if sub_value == "Unspecified":
        return []
    
    hierarchy = _load_hierarchy_mappings()
    sub_to_detail = hierarchy.get("sub_to_detail", {})
    return sub_to_detail.get(sub_value, [])


def get_valid_level4_values(detail_value: str) -> List[str]:
    """
    Get valid level4 values for a given detail category.
    
    Args:
        detail_value: The detail category value
        
    Returns:
        List of valid level4 values for the detail category.
        If detail_value is "Unspecified" or not found, returns empty list.
    """
    if detail_value == "Unspecified":
        return []
    
    hierarchy = _load_hierarchy_mappings()
    detail_to_level4 = hierarchy.get("detail_to_level4", {})
    return detail_to_level4.get(detail_value, [])


def filter_values_by_hierarchy(all_values: List[str], valid_values: List[str]) -> List[str]:
    """
    Filter a list of values to only include those that are valid according to hierarchy.
    
    Args:
        all_values: Complete list of all possible values
        valid_values: List of valid values from hierarchy mapping
        
    Returns:
        Filtered list containing only valid values.
        If valid_values is empty, returns all_values (no filtering).
    """
    if not valid_values:
        return all_values
    
    # Return intersection of all_values and valid_values
    valid_set = set(valid_values)
    return [v for v in all_values if v in valid_set]


def load_json_values(filename: str) -> List[str]:
    """
    Load values from a JSON file in the resources directory.
    
    Args:
        filename: Name of the JSON file (e.g., "main_values.json")
        
    Returns:
        List of values from the JSON file
    """
    filepath = RESOURCES_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Values file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

