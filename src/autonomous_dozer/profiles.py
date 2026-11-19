"""Load sourced dimensions without coupling algorithms to a particular machine."""
import json
from pathlib import Path
from .domain import Blade


def load_blade(path: str | Path) -> Blade:
    profile = json.loads(Path(path).read_text())
    dims, assumed = profile["published_dimensions"], profile["simulation_assumptions"]
    return Blade(dims["blade_width_m"], dims["blade_face_height_m"],
                 dims["rated_blade_capacity_m3"], assumed["effective_capacity_factor"],
                 assumed["max_push_distance_m"])
