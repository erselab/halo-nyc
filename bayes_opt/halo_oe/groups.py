"""Group an inventory's native sub-categories into configurable super-categories.

Each inventory in ``nyc_ch4_emissions.h5`` stores many native sub-categories whose
labels differ between inventories (EDGAR/EPA/Pittsburgh use different taxonomies).
To solve for a small, identifiable set of process groups (natural gas, landfill,
wastewater, …) that are comparable across inventories, sub-categories are mapped
into groups by **case-insensitive keyword matching** on their labels.

The mapping is fully configurable: a ``[category_groups]`` config section provides
``group = keyword, keyword, ...`` lines; the first group whose any keyword is a
substring of a sub-category label claims it, and anything unmatched falls into
``other``. A sensible default is provided when no config section is given. The
assignment is returned so it can be printed and verified.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_KEYWORD_MAP", "keyword_map_from_config", "assign_groups",
           "group_indices"]

# Default keyword -> group mapping (ordered; first match wins). Lowercase keywords.
DEFAULT_KEYWORD_MAP = {
    "natural_gas": ["ng_", "natural gas", "fuel exploitation gas"],
    "petroleum":   ["petroleum", "oil", "refineries", "refining"],
    "coal":        ["coal"],
    "landfill":    ["landfill", "solid waste", "waste incineration", "composting",
                    "waste burning"],
    "wastewater":  ["wastewater", "waste water"],
    "agriculture": ["enteric", "manure", "agricultur", "rice", "field burning",
                    "soils"],
    "combustion":  ["combustion", "stationary", "mobile", "power industry",
                    "buildings", "manufacturing", "iron and steel", "aviation",
                    "shipping", "railways", "transport", "chemical"],
}


def keyword_map_from_config(cfg) -> dict[str, list[str]]:
    """Build a group->keywords mapping from a ``[category_groups]`` config section.

    Returns :data:`DEFAULT_KEYWORD_MAP` if the section is absent or empty.
    """
    section = cfg.section("category_groups")
    if not section:
        return DEFAULT_KEYWORD_MAP
    return {group: [k.strip().lower() for k in value.split(",") if k.strip()]
            for group, value in section.items()}


def assign_groups(labels, keyword_map=DEFAULT_KEYWORD_MAP) -> dict[str, str]:
    """Assign each sub-category label to a group (first keyword match; else 'other')."""
    out = {}
    for label in labels:
        low = str(label).lower()
        group = "other"
        for grp, keywords in keyword_map.items():
            if any(kw in low for kw in keywords):
                group = grp
                break
        out[label] = group
    return out


def group_indices(labels, keyword_map=DEFAULT_KEYWORD_MAP):
    """Map labels to groups and collect the sub-category indices per group.

    Returns ``(indices, assignment)`` where ``indices`` is ``{group: [row indices
    into the inventory array]}`` for non-empty groups only (in keyword-map order,
    with ``other`` last if used), and ``assignment`` is ``{label: group}``.
    """
    assignment = assign_groups(labels, keyword_map)
    order = list(keyword_map.keys()) + ["other"]
    indices: dict[str, list[int]] = {g: [] for g in order}
    for i, label in enumerate(labels):
        indices[assignment[label]].append(i)
    indices = {g: np.array(ix, dtype=int) for g, ix in indices.items() if ix}
    return indices, assignment
