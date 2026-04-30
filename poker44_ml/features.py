from __future__ import annotations

from collections import Counter
from typing import Any

import math
import statistics


MEANINGFUL_ACTIONS = ("call", "check", "bet", "raise", "fold")
AGGRESSIVE_ACTIONS = ("bet", "raise")
PASSIVE_ACTIONS = ("call", "check")
POSITION_NAMES = ("button", "small_blind", "big_blind", "early", "middle", "late", "unknown")
POSTFLOP_STREETS = ("flop", "turn", "river")
TIMING_KEYS = (
    "elapsed_ms",
    "decision_ms",
    "time_ms",
    "action_ms",
    "duration_ms",
    "response_ms",
    "seconds_to_act",
    "time_to_act",
)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(statistics.fmean(values)),
        f"{prefix}_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
    }


def coefficient_of_variation(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    if not positives:
        return 0.0
    mean = float(statistics.fmean(positives))
    return safe_div(float(statistics.pstdev(positives)) if len(positives) > 1 else 0.0, mean)


def _linear_slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = float(statistics.fmean(values))
    numerator = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values))
    denominator = sum((idx - x_mean) ** 2 for idx in range(n))
    return safe_div(numerator, denominator)


def _safe_correlation(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) <= 1:
        return 0.0
    mean_a = float(statistics.fmean(a))
    mean_b = float(statistics.fmean(b))
    var_a = sum((value - mean_a) ** 2 for value in a)
    var_b = sum((value - mean_b) ** 2 for value in b)
    if var_a <= 0.0 or var_b <= 0.0:
        return 0.0
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    return safe_div(cov, math.sqrt(var_a * var_b))


def _action_counts(actions: list[dict[str, Any]]) -> Counter[str]:
    return Counter((action.get("action_type") or "").lower() for action in actions)


def _street_count(streets: list[Any], actions: list[dict[str, Any]]) -> float:
    if streets:
        return float(len(streets))
    street_names = {
        (action.get("street") or "").lower()
        for action in actions
        if action.get("street")
    }
    street_names.discard("")
    return float(len(street_names))


def _street_depth_ratio(street_count: float) -> float:
    # Hold'em typically has preflop, flop, turn, river.
    return clamp01(street_count / 4.0)


def _position_name(seat: int | None, button_seat: int | None, n_players: int) -> str:
    if not seat or not button_seat or n_players < 2:
        return "unknown"
    distance = (seat - button_seat) % max(n_players, 1)
    if distance == 0:
        return "button"
    if distance == 1:
        return "small_blind"
    if distance == 2:
        return "big_blind"
    if n_players <= 3:
        return "late" if distance >= n_players - 1 else "middle"
    if distance <= 3:
        return "early"
    if distance >= n_players - 1:
        return "late"
    return "middle"


def _seat_action_profile(actions: list[dict[str, Any]], seat: int | None) -> dict[str, float]:
    if not seat:
        return {
            "hero_action_count": 0.0,
            "hero_vpip_proxy": 0.0,
            "hero_raise_freq": 0.0,
            "hero_fold_freq": 0.0,
            "hero_aggression_ratio": 0.0,
        }

    seat_actions = [action for action in actions if action.get("actor_seat") == seat]
    counts = _action_counts(seat_actions)
    meaningful = max(1, sum(counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS))
    aggressive = sum(counts.get(kind, 0) for kind in AGGRESSIVE_ACTIONS)
    passive = sum(counts.get(kind, 0) for kind in PASSIVE_ACTIONS)
    vpip = counts.get("call", 0) + counts.get("bet", 0) + counts.get("raise", 0)
    return {
        "hero_action_count": float(len(seat_actions)),
        "hero_vpip_proxy": safe_div(vpip, meaningful),
        "hero_raise_freq": safe_div(counts.get("raise", 0), meaningful),
        "hero_fold_freq": safe_div(counts.get("fold", 0), meaningful),
        "hero_aggression_ratio": safe_div(aggressive, aggressive + passive),
    }


def _normalized_entropy(values: list[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    total = sum(positives)
    if total <= 0.0 or len(positives) <= 1:
        return 0.0
    probs = [value / total for value in positives]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
    return safe_div(entropy, math.log(len(probs)))


def _first_postblind_preflop_action(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for action in actions:
        action_type = (action.get("action_type") or "").lower()
        if (action.get("street") or "").lower() != "preflop":
            continue
        if action_type in {"small_blind", "big_blind", "ante", "straddle"}:
            continue
        return action
    return None


def _street_openers(actions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    openers: dict[str, dict[str, Any]] = {}
    for action in actions:
        street = (action.get("street") or "").lower()
        if street and street not in openers:
            openers[street] = action
    return openers


def _action_transition_entropy(action_types: list[str]) -> float:
    if len(action_types) <= 1:
        return 0.0
    transitions = Counter(zip(action_types[:-1], action_types[1:]))
    return _normalized_entropy(list(transitions.values()))


def _action_timing_value(action: dict[str, Any]) -> float | None:
    for key in TIMING_KEYS:
        if key not in action:
            continue
        value = safe_float(action.get(key), default=-1.0)
        if value < 0.0:
            continue
        if key.endswith("_ms") or key in {"elapsed_ms", "decision_ms", "time_ms", "action_ms", "duration_ms", "response_ms"}:
            value = value / 1000.0
        return value
    return None


def _timing_micro_features(actions: list[dict[str, Any]]) -> dict[str, float]:
    timings: list[float] = []
    pot_before: list[float] = []
    amount_sizes: list[float] = []
    street_depths: list[float] = []
    for action in actions:
        timing = _action_timing_value(action)
        if timing is None:
            continue
        timings.append(timing)
        pot_before.append(safe_float(action.get("pot_before")))
        amount_sizes.append(safe_float(action.get("normalized_amount_bb")))
        street = (action.get("street") or "").lower()
        street_depths.append({"preflop": 1.0, "flop": 2.0, "turn": 3.0, "river": 4.0}.get(street, 0.0))

    if not timings:
        return {
            "timing_available": 0.0,
            "decision_time_mean": 0.0,
            "decision_time_std": 0.0,
            "decision_time_cv": 0.0,
            "decision_time_min": 0.0,
            "decision_time_max": 0.0,
            "fast_action_ratio": 0.0,
            "slow_action_ratio": 0.0,
            "timing_entropy": 0.0,
            "timing_slope": 0.0,
            "timing_pot_correlation": 0.0,
            "timing_bet_size_correlation": 0.0,
            "timing_street_correlation": 0.0,
        }

    fast_threshold = 1.5
    slow_threshold = 8.0
    buckets = Counter(min(10, int(value // 1.5)) for value in timings)
    summary = summarize(timings, "decision_time")
    return {
        "timing_available": 1.0,
        "decision_time_mean": summary["decision_time_mean"],
        "decision_time_std": summary["decision_time_std"],
        "decision_time_cv": coefficient_of_variation(timings),
        "decision_time_min": summary["decision_time_min"],
        "decision_time_max": summary["decision_time_max"],
        "fast_action_ratio": safe_div(sum(1 for value in timings if value <= fast_threshold), len(timings)),
        "slow_action_ratio": safe_div(sum(1 for value in timings if value >= slow_threshold), len(timings)),
        "timing_entropy": _normalized_entropy(list(buckets.values())),
        "timing_slope": _linear_slope(timings),
        "timing_pot_correlation": _safe_correlation(timings, pot_before),
        "timing_bet_size_correlation": _safe_correlation(timings, amount_sizes),
        "timing_street_correlation": _safe_correlation(timings, street_depths),
    }


def _decision_optimality_features(actions: list[dict[str, Any]], hero_position: str) -> dict[str, float]:
    decisions = [
        action
        for action in actions
        if (action.get("action_type") or "").lower() in MEANINGFUL_ACTIONS
    ]
    if not decisions:
        return {
            "pot_odds_call_pressure_mean": 0.0,
            "fold_under_good_odds_ratio": 0.0,
            "call_under_bad_odds_ratio": 0.0,
            "oversized_bet_ratio": 0.0,
            "thin_bet_ratio": 0.0,
            "position_adjusted_aggression": 0.0,
            "street_adjusted_aggression": 0.0,
            "passive_when_checked_to_ratio": 0.0,
        }

    pot_odds: list[float] = []
    good_odds_folds = 0
    bad_odds_calls = 0
    oversized_bets = 0
    thin_bets = 0
    checked_to_passive = 0
    aggressive_weighted = 0.0
    street_weighted = 0.0
    position_weight = {
        "button": 1.20,
        "late": 1.15,
        "middle": 1.00,
        "early": 0.85,
        "small_blind": 0.80,
        "big_blind": 0.90,
        "unknown": 1.00,
    }.get(hero_position, 1.0)

    for action in decisions:
        action_type = (action.get("action_type") or "").lower()
        pot_before_bb = safe_float(action.get("pot_before"))
        amount_bb = safe_float(action.get("normalized_amount_bb"))
        call_to_bb = safe_float(action.get("call_to"))
        raise_to_bb = safe_float(action.get("raise_to"))
        required_call = call_to_bb if call_to_bb > 0.0 else amount_bb
        odds = safe_div(required_call, pot_before_bb + required_call)
        if required_call > 0.0:
            pot_odds.append(odds)
            if action_type == "fold" and odds <= 0.18:
                good_odds_folds += 1
            if action_type == "call" and odds >= 0.38:
                bad_odds_calls += 1

        bet_fraction = safe_div(max(amount_bb, raise_to_bb), pot_before_bb if pot_before_bb > 0.0 else 1.0)
        if action_type in AGGRESSIVE_ACTIONS:
            if bet_fraction >= 1.25:
                oversized_bets += 1
            if 0.0 < bet_fraction <= 0.20:
                thin_bets += 1
            aggressive_weighted += position_weight

        street = (action.get("street") or "").lower()
        street_weight = {"preflop": 0.85, "flop": 1.00, "turn": 1.10, "river": 1.20}.get(street, 1.0)
        if action_type in AGGRESSIVE_ACTIONS:
            street_weighted += street_weight
        if action_type in PASSIVE_ACTIONS and pot_before_bb <= 0.0:
            checked_to_passive += 1

    n = len(decisions)
    aggressive_count = sum(1 for action in decisions if (action.get("action_type") or "").lower() in AGGRESSIVE_ACTIONS)
    return {
        "pot_odds_call_pressure_mean": float(statistics.fmean(pot_odds)) if pot_odds else 0.0,
        "fold_under_good_odds_ratio": safe_div(good_odds_folds, n),
        "call_under_bad_odds_ratio": safe_div(bad_odds_calls, n),
        "oversized_bet_ratio": safe_div(oversized_bets, max(aggressive_count, 1)),
        "thin_bet_ratio": safe_div(thin_bets, max(aggressive_count, 1)),
        "position_adjusted_aggression": safe_div(aggressive_weighted, n),
        "street_adjusted_aggression": safe_div(street_weighted, n),
        "passive_when_checked_to_ratio": safe_div(checked_to_passive, n),
    }


def hand_features(hand: dict[str, Any]) -> dict[str, float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}
    metadata = hand.get("metadata") or {}

    counts = _action_counts(actions)
    action_types = [(action.get("action_type") or "").lower() for action in actions]
    meaningful = max(1, sum(counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS))
    aggressive = sum(counts.get(kind, 0) for kind in AGGRESSIVE_ACTIONS)
    passive = sum(counts.get(kind, 0) for kind in PASSIVE_ACTIONS)

    amounts_bb = [
        safe_float(action.get("normalized_amount_bb"))
        for action in actions
        if action.get("normalized_amount_bb") is not None
    ]
    bet_like_sizes = [
        safe_float(action.get("normalized_amount_bb"))
        for action in actions
        if (action.get("action_type") or "").lower() in AGGRESSIVE_ACTIONS
        and action.get("normalized_amount_bb") is not None
    ]
    pot_after_values = [
        safe_float(action.get("pot_after"))
        for action in actions
        if action.get("pot_after") is not None
    ]
    actor_seats = [
        int(action.get("actor_seat"))
        for action in actions
        if action.get("actor_seat") is not None
    ]
    player_stacks = [
        safe_float(player.get("starting_stack"))
        for player in players
        if player.get("starting_stack") is not None
    ]
    revealed = sum(1 for player in players if player.get("showed_hand"))

    street_count = _street_count(streets, actions)
    button_seat = metadata.get("button_seat")
    hero_seat = metadata.get("hero_seat")
    bb_size = safe_float(metadata.get("bb"), 1.0) or 1.0
    hero_position = _position_name(hero_seat, button_seat, len(players))
    hero_profile = _seat_action_profile(actions, hero_seat)
    timing_profile = _timing_micro_features(actions)
    optimality_profile = _decision_optimality_features(actions, hero_position)

    final_pot = safe_float(outcome.get("total_pot")) or (max(pot_after_values) if pot_after_values else 0.0)
    final_pot_bb = safe_div(final_pot, bb_size)
    player_stacks_bb = [safe_div(stack, bb_size) for stack in player_stacks]
    street_action_counts = Counter(
        (action.get("street") or "").lower()
        for action in actions
        if action.get("street")
    )
    actor_counts = Counter(actor_seats)
    meaningful_action_count = float(sum(counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS))
    size_buckets = Counter(int(round(size * 2.0)) for size in bet_like_sizes if size > 0.0)
    preflop_opener = _first_postblind_preflop_action(actions)
    street_openers = _street_openers(actions)
    donk_bet_count = 0
    donk_bet_opportunities = 0
    previous_street_actor: int | None = None
    for street in POSTFLOP_STREETS:
        opener = street_openers.get(street)
        if opener is None:
            continue
        opener_type = (opener.get("action_type") or "").lower()
        opener_actor = opener.get("actor_seat")
        if opener_type in AGGRESSIVE_ACTIONS:
            donk_bet_opportunities += 1
            if previous_street_actor is not None and opener_actor != previous_street_actor:
                donk_bet_count += 1
        street_actions = [a for a in actions if (a.get("street") or "").lower() == street]
        aggressive_actors = [
            a.get("actor_seat")
            for a in street_actions
            if (a.get("action_type") or "").lower() in AGGRESSIVE_ACTIONS
        ]
        previous_street_actor = aggressive_actors[-1] if aggressive_actors else previous_street_actor
    limp_flag = 0.0
    if preflop_opener is not None and (preflop_opener.get("action_type") or "").lower() == "call":
        limp_flag = 1.0
    first_action_aggressive = 1.0 if action_types and action_types[0] in AGGRESSIVE_ACTIONS else 0.0

    feats = {
        "n_actions": float(len(actions)),
        "n_players": float(len(players)),
        "n_streets": street_count,
        "street_depth_ratio": _street_depth_ratio(street_count),
        "showdown": 1.0 if outcome.get("showdown") else 0.0,
        "revealed_players_ratio": safe_div(revealed, max(len(players), 1)),
        "call_ratio": safe_div(counts.get("call", 0), meaningful),
        "check_ratio": safe_div(counts.get("check", 0), meaningful),
        "bet_ratio": safe_div(counts.get("bet", 0), meaningful),
        "raise_ratio": safe_div(counts.get("raise", 0), meaningful),
        "fold_ratio": safe_div(counts.get("fold", 0), meaningful),
        "passive_ratio": safe_div(passive, meaningful),
        "aggression_ratio": safe_div(aggressive, aggressive + passive),
        "vpip_proxy": safe_div(
            counts.get("call", 0) + counts.get("bet", 0) + counts.get("raise", 0),
            meaningful,
        ),
        "raise_frequency": safe_div(counts.get("raise", 0), meaningful),
        "fold_to_action_tendency": safe_div(counts.get("fold", 0), aggressive + counts.get("call", 0)),
        "bet_like_count": float(aggressive),
        "avg_action_size_bb": float(statistics.fmean(amounts_bb)) if amounts_bb else 0.0,
        "avg_bet_size_bb": float(statistics.fmean(bet_like_sizes)) if bet_like_sizes else 0.0,
        "bet_size_std_bb": float(statistics.pstdev(bet_like_sizes)) if len(bet_like_sizes) > 1 else 0.0,
        "max_bet_size_bb": float(max(bet_like_sizes)) if bet_like_sizes else 0.0,
        "action_size_std_bb": float(statistics.pstdev(amounts_bb)) if len(amounts_bb) > 1 else 0.0,
        "bet_size_bucket_entropy": _normalized_entropy(list(size_buckets.values())),
        "final_pot": final_pot,
        "final_pot_bb": final_pot_bb,
        "avg_starting_stack": float(statistics.fmean(player_stacks)) if player_stacks else 0.0,
        "avg_starting_stack_bb": float(statistics.fmean(player_stacks_bb)) if player_stacks_bb else 0.0,
        "short_stack_ratio": safe_div(sum(1 for stack in player_stacks_bb if stack <= 20.0), max(len(player_stacks_bb), 1)),
        "all_in_like_ratio": safe_div(sum(1 for size in bet_like_sizes if size >= 20.0), max(len(bet_like_sizes), 1)),
        "preflop_action_ratio": safe_div(street_action_counts.get("preflop", 0), max(len(actions), 1)),
        "flop_action_ratio": safe_div(street_action_counts.get("flop", 0), max(len(actions), 1)),
        "turn_action_ratio": safe_div(street_action_counts.get("turn", 0), max(len(actions), 1)),
        "river_action_ratio": safe_div(street_action_counts.get("river", 0), max(len(actions), 1)),
        "action_entropy": _normalized_entropy([counts.get(kind, 0) for kind in MEANINGFUL_ACTIONS]),
        "street_entropy": _normalized_entropy(list(street_action_counts.values())),
        "action_transition_entropy": _action_transition_entropy(action_types),
        "actor_count_ratio": safe_div(len(actor_counts), max(len(players), 1)),
        "actor_concentration": safe_div(max(actor_counts.values()) if actor_counts else 0.0, max(len(actions), 1)),
        "heads_up_flag": 1.0 if len(players) == 2 else 0.0,
        "full_ring_flag": 1.0 if len(players) >= 6 else 0.0,
        "deep_stack_flag": 1.0 if (player_stacks_bb and statistics.fmean(player_stacks_bb) >= 100.0) else 0.0,
        "aggressive_to_pot_ratio": safe_div(sum(bet_like_sizes), final_pot_bb if final_pot_bb > 0 else 1.0),
        "meaningful_action_count": meaningful_action_count,
        "donk_bet_rate": safe_div(donk_bet_count, donk_bet_opportunities),
        "limp_flag": limp_flag,
        "first_action_aggressive": first_action_aggressive,
        "hero_on_button": 1.0 if hero_position == "button" else 0.0,
        "hero_in_blinds": 1.0 if hero_position in {"small_blind", "big_blind"} else 0.0,
        "hero_early_position": 1.0 if hero_position == "early" else 0.0,
        "hero_middle_position": 1.0 if hero_position == "middle" else 0.0,
        "hero_late_position": 1.0 if hero_position == "late" else 0.0,
        "hero_position_known": 1.0 if hero_position != "unknown" else 0.0,
    }
    feats.update(hero_profile)
    feats.update(timing_profile)
    feats.update(optimality_profile)
    return feats


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    if not chunk:
        return {"chunk_size": 0.0}

    per_hand = [hand_features(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())

    out = {"chunk_size": float(len(chunk)), "hand_count": float(len(chunk))}
    for name in feature_names:
        values = [row[name] for row in per_hand]
        out.update(summarize(values, name))

    vpip_vals = [row["vpip_proxy"] for row in per_hand]
    aggr_vals = [row["aggression_ratio"] for row in per_hand]
    raise_vals = [row["raise_frequency"] for row in per_hand]
    fold_vals = [row["fold_to_action_tendency"] for row in per_hand]
    stack_vals = [row["avg_starting_stack_bb"] for row in per_hand]
    pot_vals = [row["final_pot_bb"] for row in per_hand]
    aggression_vals = [row["aggression_ratio"] for row in per_hand]
    action_entropy_vals = [row["action_entropy"] for row in per_hand]
    bet_cv_vals = [
        safe_div(row["bet_size_std_bb"], row["avg_bet_size_bb"] if row["avg_bet_size_bb"] > 0 else 1.0)
        for row in per_hand
    ]
    decision_time_vals = [row["decision_time_mean"] for row in per_hand if row["timing_available"] > 0.0]
    fast_action_vals = [row["fast_action_ratio"] for row in per_hand]
    slow_action_vals = [row["slow_action_ratio"] for row in per_hand]
    pot_odds_vals = [row["pot_odds_call_pressure_mean"] for row in per_hand]
    fold_good_odds_vals = [row["fold_under_good_odds_ratio"] for row in per_hand]
    call_bad_odds_vals = [row["call_under_bad_odds_ratio"] for row in per_hand]
    position_aggr_vals = [row["position_adjusted_aggression"] for row in per_hand]
    street_aggr_vals = [row["street_adjusted_aggression"] for row in per_hand]

    out["consistency_score"] = safe_div(
        1.0,
        1.0
        + (statistics.pstdev(vpip_vals) if len(vpip_vals) > 1 else 0.0)
        + (statistics.pstdev(aggr_vals) if len(aggr_vals) > 1 else 0.0)
        + (statistics.pstdev(raise_vals) if len(raise_vals) > 1 else 0.0),
    )
    out["behavioral_consistency_score"] = safe_div(
        1.0,
        1.0
        + coefficient_of_variation(vpip_vals)
        + coefficient_of_variation(aggression_vals)
        + coefficient_of_variation(raise_vals)
        + coefficient_of_variation(bet_cv_vals)
        + coefficient_of_variation(action_entropy_vals),
    )
    out["vpip_consistency_cv"] = coefficient_of_variation(vpip_vals)
    out["aggression_consistency_cv"] = coefficient_of_variation(aggression_vals)
    out["raise_frequency_consistency_cv"] = coefficient_of_variation(raise_vals)
    out["bet_sizing_consistency_cv"] = coefficient_of_variation(bet_cv_vals)
    out["action_entropy_consistency_cv"] = coefficient_of_variation(action_entropy_vals)
    out["behavioral_pattern_slope_vpip"] = _linear_slope(vpip_vals)
    out["behavioral_pattern_slope_aggression"] = _linear_slope(aggression_vals)
    out["vpip_aggr_gap_mean"] = float(statistics.fmean(v - a for v, a in zip(vpip_vals, aggr_vals)))
    out["fold_raise_gap_mean"] = float(statistics.fmean(f - r for f, r in zip(fold_vals, raise_vals)))
    out["showdown_rate"] = float(statistics.fmean(row["showdown"] for row in per_hand))
    out["deep_street_rate"] = float(statistics.fmean(1.0 if row["n_streets"] >= 3.0 else 0.0 for row in per_hand))
    out["avg_players"] = float(statistics.fmean(row["n_players"] for row in per_hand))
    out["avg_actions"] = float(statistics.fmean(row["n_actions"] for row in per_hand))
    out["avg_streets"] = float(statistics.fmean(row["n_streets"] for row in per_hand))
    out["avg_stack_to_pot_ratio"] = float(
        statistics.fmean(safe_div(stack, pot if pot > 0 else 1.0) for stack, pot in zip(stack_vals, pot_vals))
    )
    out["action_entropy_mean"] = float(statistics.fmean(row["action_entropy"] for row in per_hand))
    out["street_entropy_mean"] = float(statistics.fmean(row["street_entropy"] for row in per_hand))
    out["actor_concentration_mean"] = float(statistics.fmean(row["actor_concentration"] for row in per_hand))
    out["bet_size_cv_mean"] = float(
        statistics.fmean(
            safe_div(row["bet_size_std_bb"], row["avg_bet_size_bb"] if row["avg_bet_size_bb"] > 0 else 1.0)
            for row in per_hand
        )
    )
    out["timing_available_rate"] = safe_div(len(decision_time_vals), len(per_hand))
    out["decision_time_chunk_mean"] = float(statistics.fmean(decision_time_vals)) if decision_time_vals else 0.0
    out["decision_time_chunk_cv"] = coefficient_of_variation(decision_time_vals)
    out["fast_action_rate_mean"] = float(statistics.fmean(fast_action_vals))
    out["slow_action_rate_mean"] = float(statistics.fmean(slow_action_vals))
    out["timing_regularity_score"] = safe_div(
        1.0,
        1.0 + out["decision_time_chunk_cv"] + coefficient_of_variation(fast_action_vals),
    )
    out["timing_to_aggression_correlation"] = _safe_correlation(
        [row["decision_time_mean"] for row in per_hand],
        aggression_vals,
    )
    out["timing_to_pot_odds_correlation"] = _safe_correlation(
        [row["decision_time_mean"] for row in per_hand],
        pot_odds_vals,
    )
    out["decision_optimality_score"] = safe_div(
        1.0,
        1.0
        + float(statistics.fmean(fold_good_odds_vals))
        + float(statistics.fmean(call_bad_odds_vals))
        + abs(float(statistics.fmean(position_aggr_vals)) - float(statistics.fmean(street_aggr_vals))),
    )
    out["pot_odds_pressure_mean"] = float(statistics.fmean(pot_odds_vals))
    out["fold_good_odds_rate_mean"] = float(statistics.fmean(fold_good_odds_vals))
    out["call_bad_odds_rate_mean"] = float(statistics.fmean(call_bad_odds_vals))
    out["position_aggression_alignment_mean"] = float(statistics.fmean(position_aggr_vals))
    out["street_aggression_alignment_mean"] = float(statistics.fmean(street_aggr_vals))
    out["position_street_aggression_gap"] = abs(
        out["position_aggression_alignment_mean"] - out["street_aggression_alignment_mean"]
    )
    out["donk_bet_rate_mean"] = float(statistics.fmean(row["donk_bet_rate"] for row in per_hand))
    out["limp_rate"] = float(statistics.fmean(row["limp_flag"] for row in per_hand))
    out["first_action_aggressive_rate"] = float(
        statistics.fmean(row["first_action_aggressive"] for row in per_hand)
    )
    out["bet_size_bucket_entropy_mean"] = float(
        statistics.fmean(row["bet_size_bucket_entropy"] for row in per_hand)
    )
    out["action_transition_entropy_mean"] = float(
        statistics.fmean(row["action_transition_entropy"] for row in per_hand)
    )
    out["log_chunk_size"] = math.log1p(len(chunk))
    return out
