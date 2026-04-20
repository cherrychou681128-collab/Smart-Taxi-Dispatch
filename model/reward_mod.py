from __future__ import annotations
from pathlib import Path
import json
import time
from typing import Dict, Any

DEFAULT_STATE = {
    "driver_points": 0.0,
    "bias": 0.0,
    "updated_at": None,
    "events_processed": 0
}


def load_state(state_path: Path) -> Dict[str, Any]:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return dict(DEFAULT_STATE)


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def points_to_bias(points: float) -> float:
    b = points / 50.0
    return max(-5.0, min(5.0, b))


def update_state_from_events(state: Dict[str, Any], events_path: Path) -> Dict[str, Any]:

    if not events_path.exists():
        state["bias"] = points_to_bias(float(state.get("driver_points", 0.0)))
        return state

    processed = int(state.get("events_processed", 0))
    lines = events_path.read_text(encoding="utf-8").splitlines()

    if processed >= len(lines):
        state["bias"] = points_to_bias(float(state.get("driver_points", 0.0)))
        return state

    for line in lines[processed:]:
        if not line.strip():
            continue
        ev = json.loads(line)
        chosen = ev.get("chosen", {})
        best = ev.get("best", {})

        pred = float(chosen.get("pred", 0.0))
        dist = float(chosen.get("dist_km", 0.0))
        chosen_score = float(chosen.get("score", 0.0))
        best_score = float(best.get("score", chosen_score))

        r_demand = pred * 8.0
        p_distance = -dist * 1.2
        p_low_demand = -3.0 if pred < 0.2 else 0.0
        gap = max(0.0, best_score - chosen_score)
        p_gap = -gap * 1.5

        delta = r_demand + p_distance + p_low_demand + p_gap
        delta = max(-15.0, min(15.0, delta))

        state["driver_points"] = float(state.get("driver_points", 0.0)) + delta

    state["events_processed"] = len(lines)
    state["bias"] = points_to_bias(float(state.get("driver_points", 0.0)))
    return state


def get_bias(outputs_dir: Path, enable: bool = True) -> float:

    if not enable:
        return 0.0
    state_path = outputs_dir / "reward_state.json"
    events_path = outputs_dir / "reward_events.jsonl"

    state = load_state(state_path)
    state = update_state_from_events(state, events_path)
    save_state(state_path, state)
    return float(state.get("bias", 0.0))


def apply_bias_to_score(base_score: float, pred: float, dist_km: float, bias: float) -> float:

    wDemand = 1.0 + 0.25 * bias
    wDist = 0.6 - 0.10 * bias
    return base_score + pred * wDemand - dist_km * wDist
