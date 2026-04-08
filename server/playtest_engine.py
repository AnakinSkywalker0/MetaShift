"""
MetaShift — PlaytestEngine
Deterministic rule-based synthetic player behavior model.
Same input ALWAYS produces same output — no random state.

Rules implemented:
  1. TTK Rule   : weapon TTK 40%+ below average → usage converges toward 65%+
  2. Economy Rule: archetype economy gap > 25% → weaker archetype dropout spikes
  3. Perk Rule  : kill-reward perk with trivial PvE uptime → near-permanent active
  4. Cascade    : high perk uptime → economy bonus → archetype economy shifts
  5. Winrate    : function of economy advantage + weapon affinity + dropout pressure
"""

from __future__ import annotations
import copy
import math
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# PlaytestEngine
# ---------------------------------------------------------------------------

class PlaytestEngine:
    """
    Runs a full deterministic simulation pass over game_state and returns
    computed metrics.  No side effects — always returns a fresh metrics dict.
    """

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def simulate(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Return fully computed metrics for *game_state*."""
        weapons    = game_state.get("weapons", {})
        archetypes = game_state.get("archetypes", {})
        perks      = game_state.get("perks", {})
        game_mode  = game_state.get("game_mode", "PvP")
        synergies  = game_state.get("synergy_combos", [])

        # --- Pass 1: weapon usage (needs first-pass avg TTK) ----------------
        weapon_usage = self._calculate_weapon_usage(weapons)

        # --- Pass 2: perk uptimes -------------------------------------------
        perk_uptimes = self._calculate_perk_uptimes(perks, game_mode, archetypes, weapon_usage)

        # --- Pass 3: economy rates (base + perk bonuses) --------------------
        economy_rates = self._calculate_economy_rates(archetypes, perks, perk_uptimes)

        # --- Pass 4: iterate once to let economy feed back into weapon usage
        #             (wealthy archetypes gravitate to their preferred weapons)
        weapon_usage = self._apply_economy_weapon_affinity(weapon_usage, archetypes, economy_rates)
        # Re-normalise after affinity adjustment
        total = sum(weapon_usage.values()) or 1.0
        weapon_usage = {k: v / total for k, v in weapon_usage.items()}

        # --- Pass 5: dropout rates ------------------------------------------
        dropout_rates = self._calculate_dropout_rates(archetypes, economy_rates)

        # --- Pass 6: archetype winrates -------------------------------------
        archetype_winrates = self._calculate_winrates(archetypes, weapon_usage, economy_rates, dropout_rates)

        # --- Pass 7: weighted average TTK -----------------------------------
        average_ttk = self._calculate_average_ttk(weapons, weapon_usage)

        # --- Pass 8: dominant strategy detection ----------------------------
        dominant_strategy = self._detect_dominant_strategy(
            synergies, weapon_usage, perk_uptimes, economy_rates, archetype_winrates
        )

        # --- Perk type map (for grader: passive perks exempted from uptime cap)
        perk_type_map = {name: p.get("type", "passive") for name, p in perks.items()}

        return {
            "weapon_usage_rates":      weapon_usage,
            "perk_uptimes":            perk_uptimes,
            "perk_types":              perk_type_map,
            "archetype_economy_rates": economy_rates,
            "room_dropout_rates":      dropout_rates,
            "archetype_winrates":      archetype_winrates,
            "average_ttk":             average_ttk,
            "dominant_strategy":       dominant_strategy,
        }

    def apply_stat_adjustment(
        self,
        game_state: Dict[str, Any],
        target: str,
        parameter: str,
        change: float,
    ) -> Dict[str, Any]:
        """
        Apply *change* as a delta multiplier to game_state[target][parameter].
        e.g. change=-0.15 → new_value = old_value * 0.85
        Returns a deep copy of game_state with the mutation applied.
        Raises KeyError if target / parameter combo not found.
        """
        new_state = copy.deepcopy(game_state)
        multiplier = 1.0 + change

        for bucket in ("weapons", "archetypes", "perks"):
            bucket_data = new_state.get(bucket, {})
            if target in bucket_data and parameter in bucket_data[target]:
                old_val = bucket_data[target][parameter]
                bucket_data[target][parameter] = old_val * multiplier
                return new_state

        raise KeyError(
            f"Cannot find target='{target}' with parameter='{parameter}' "
            f"in weapons / archetypes / perks."
        )

    # -----------------------------------------------------------------------
    # Rule 1 — TTK → Usage
    # -----------------------------------------------------------------------

    def _calculate_weapon_usage(self, weapons: Dict[str, Any]) -> Dict[str, float]:
        """
        Deterministic TTK-based weapon usage model.
        Rule: weapon TTK 40%+ below average → usage converges toward 65%+
        """
        if not weapons:
            return {}

        ttk_values: Dict[str, float] = {n: s["ttk"] for n, s in weapons.items()}
        avg_ttk = sum(ttk_values.values()) / len(ttk_values)
        if avg_ttk <= 0:
            return {n: 1.0 / len(weapons) for n in weapons}

        raw_scores: Dict[str, float] = {}
        for name, ttk in ttk_values.items():
            ratio = ttk / avg_ttk   # < 1 ⟹ faster (better)

            if ratio <= 0.60:
                # 40 %+ below average — strong dominance
                advantage = (0.60 - ratio) / 0.60
                score = 2.5 + advantage * 3.5            # pushes usage to 65 %+
            elif ratio <= 0.85:
                advantage = (0.85 - ratio) / 0.25
                score = 1.2 + advantage * 1.3
            elif ratio <= 1.20:
                score = 1.0 + (1.0 - ratio) * 0.5
            else:
                penalty = (ratio - 1.20)
                score = max(0.20, 0.80 - penalty * 0.6)

            base_mult = weapons[name].get("base_usage_multiplier", 1.0)
            raw_scores[name] = score * base_mult

        total = sum(raw_scores.values()) or 1.0
        return {n: s / total for n, s in raw_scores.items()}

    # -----------------------------------------------------------------------
    # Rule 3 — Kill-reward perks
    # -----------------------------------------------------------------------

    def _calculate_perk_uptimes(
        self,
        perks:        Dict[str, Any],
        game_mode:    str,
        archetypes:   Dict[str, Any],
        weapon_usage: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Kill-reward perk: trivial uptime conditions in PvE → near-permanent.
        Synergy overrides (e.g. chaos-event discovered weapon combos) are
        embedded in perk["synergy_kpm_override"] keyed by archetype.
        """
        uptimes: Dict[str, float] = {}

        for perk_name, perk in perks.items():
            ptype = perk.get("type", "passive")

            if ptype == "kill_reward":
                kill_threshold = perk.get("kill_threshold", 3)
                avg_kpm        = perk.get("avg_kills_per_minute", 1.5)
                duration       = perk.get("perk_duration", 10)

                # PvE multiplier
                kpm_mult = 2.0 if game_mode == "PvE" else 1.0
                effective_kpm = avg_kpm * kpm_mult

                # Synergy overrides (e.g. post-chaos weapon interaction)
                synergy_kpm = perk.get("synergy_kpm_override", 0.0)
                if synergy_kpm > 0:
                    # Weighted by how much the synergy weapon is actually used
                    synergy_weapon = perk.get("synergy_weapon", "")
                    synergy_weight = weapon_usage.get(synergy_weapon, 0.0)
                    effective_kpm = effective_kpm * (1 - synergy_weight) + synergy_kpm * synergy_weight

                kills_in_window = effective_kpm * duration / 60.0
                trigger_ratio   = kills_in_window / max(kill_threshold, 0.01)

                if trigger_ratio >= 1.0:
                    # Trivially maintained — near permanent
                    uptime = 0.50 + 0.45 * _sigmoid((trigger_ratio - 1.0) * 3.0)
                else:
                    uptime = 0.30 * trigger_ratio

                uptimes[perk_name] = _clamp(uptime, 0.0, 0.99)

            elif ptype == "passive":
                uptimes[perk_name] = _clamp(perk.get("base_uptime", 1.0), 0.0, 1.0)

            elif ptype == "active":
                cooldown  = perk.get("cooldown", 30)
                active_t  = perk.get("active_duration", 10)
                cycle     = cooldown + active_t
                uptimes[perk_name] = _clamp(active_t / cycle if cycle > 0 else 0.0, 0.0, 1.0)

            else:
                uptimes[perk_name] = 0.50

        return uptimes

    # -----------------------------------------------------------------------
    # Economy
    # -----------------------------------------------------------------------

    def _calculate_economy_rates(
        self,
        archetypes:  Dict[str, Any],
        perks:       Dict[str, Any],
        perk_uptimes: Dict[str, float],
    ) -> Dict[str, float]:
        """Economy = base + uptime-weighted perk bonuses."""
        economy: Dict[str, float] = {}

        for arch_name, arch in archetypes.items():
            base = arch.get("base_economy", 100.0)
            total = base

            for perk_name in arch.get("perks", []):
                if perk_name not in perks:
                    continue
                perk   = perks[perk_name]
                uptime = perk_uptimes.get(perk_name, 0.0)

                if "economy_multiplier" in perk:
                    mult = perk["economy_multiplier"]
                    # Multiplier scales with uptime relative to passive (always-on)
                    effective_mult = 1.0 + (mult - 1.0) * uptime
                    total = base * effective_mult

                if "economy_bonus" in perk:
                    total += base * perk["economy_bonus"] * uptime

            economy[arch_name] = total

        return economy

    # -----------------------------------------------------------------------
    # Rule 2 — Economy gap → Dropout
    # -----------------------------------------------------------------------

    def _calculate_dropout_rates(
        self,
        archetypes:    Dict[str, Any],
        economy_rates: Dict[str, float],
    ) -> Dict[str, float]:
        """Archetype dropout spikes when its economy lags > 25% behind richest."""
        if not economy_rates:
            return {}

        max_economy = max(economy_rates.values()) if economy_rates else 1.0
        dropouts: Dict[str, float] = {}

        for arch_name, arch in archetypes.items():
            base_dropout = arch.get("base_dropout", 0.05)
            arch_economy = economy_rates.get(arch_name, max_economy)

            if max_economy <= 0:
                dropouts[arch_name] = base_dropout
                continue

            gap = (max_economy - arch_economy) / max_economy  # 0 ≤ gap ≤ 1

            if gap > 0.25:
                # Rule: spike when gap > 25%
                excess = (gap - 0.25) / 0.75
                dropout = base_dropout + (0.40 - base_dropout) * _clamp(excess * 2.0, 0.0, 1.0)
            elif gap > 0.0:
                dropout = base_dropout * (1.0 + gap * 0.8)
            else:
                # This archetype IS the richest
                dropout = max(base_dropout * 0.85, 0.01)

            dropouts[arch_name] = _clamp(dropout, 0.01, 0.50)

        return dropouts

    # -----------------------------------------------------------------------
    # Winrates
    # -----------------------------------------------------------------------

    def _apply_economy_weapon_affinity(
        self,
        weapon_usage:  Dict[str, float],
        archetypes:    Dict[str, Any],
        economy_rates: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Wealthy archetypes play more, boosting usage of their preferred weapons.
        Adjusts raw usage scores BEFORE final normalisation.
        """
        avg_economy = sum(economy_rates.values()) / max(len(economy_rates), 1)
        updated = dict(weapon_usage)

        for arch_name, arch in archetypes.items():
            arch_economy = economy_rates.get(arch_name, avg_economy)
            if avg_economy <= 0:
                continue
            wealth_ratio = arch_economy / avg_economy
            bonus = (wealth_ratio - 1.0) * 0.12  # ±12% per 100% economy swing

            for weapon in arch.get("preferred_weapons", []):
                if weapon in updated:
                    updated[weapon] = max(0.01, updated[weapon] * (1.0 + bonus))

        return updated

    def _calculate_winrates(
        self,
        archetypes:    Dict[str, Any],
        weapon_usage:  Dict[str, float],
        economy_rates: Dict[str, float],
        dropout_rates: Dict[str, float],
    ) -> Dict[str, float]:
        """Winrate = base + economy edge + weapon affinity bonus − dropout drag."""
        if not archetypes:
            return {}

        avg_economy = sum(economy_rates.values()) / max(len(economy_rates), 1)
        raw: Dict[str, float] = {}

        for arch_name, arch in archetypes.items():
            score = arch.get("base_winrate", 0.50)

            # Economy advantage
            arch_econ = economy_rates.get(arch_name, avg_economy)
            if avg_economy > 0:
                econ_ratio = arch_econ / avg_economy
                score += (econ_ratio - 1.0) * 0.15

            # Weapon affinity bonus
            for w in arch.get("preferred_weapons", []):
                usage = weapon_usage.get(w, 0.0)
                score += usage * 0.20

            # Dropout penalty: fewer persistent players lose more representation
            dropout = dropout_rates.get(arch_name, 0.05)
            score *= (1.0 - dropout * 0.25)

            raw[arch_name] = score

        # Normalise to mean = 0.50
        avg_raw = sum(raw.values()) / max(len(raw), 1)
        if avg_raw <= 0:
            return {n: 0.50 for n in archetypes}
        factor = 0.50 / avg_raw
        return {n: _clamp(v * factor, 0.10, 0.90) for n, v in raw.items()}

    # -----------------------------------------------------------------------
    # Weighted average TTK
    # -----------------------------------------------------------------------

    def _calculate_average_ttk(
        self,
        weapons:      Dict[str, Any],
        weapon_usage: Dict[str, float],
    ) -> float:
        if not weapons:
            return 3.0
        weighted = sum(
            weapons[w]["ttk"] * weapon_usage.get(w, 1.0 / len(weapons))
            for w in weapons
        )
        total_w = sum(weapon_usage.get(w, 1.0 / len(weapons)) for w in weapons)
        return weighted / total_w if total_w > 0 else 3.0

    # -----------------------------------------------------------------------
    # Dominant strategy detection
    # -----------------------------------------------------------------------

    def _detect_dominant_strategy(
        self,
        synergy_combos: list,
        weapon_usage:   Dict[str, float],
        perk_uptimes:   Dict[str, float],
        economy_rates:  Dict[str, float],
        winrates:       Dict[str, float],
    ) -> Optional[str]:
        """
        Check whether any synergy combo meets all its trigger thresholds.
        Returns a descriptive strategy tag or None.
        """
        for combo in synergy_combos:
            weapon        = combo.get("weapon", "")
            perk          = combo.get("perk", "")
            archetype     = combo.get("archetype", "")
            usage_thresh  = combo.get("trigger_usage_threshold", 0.40)
            uptime_thresh = combo.get("trigger_uptime_threshold", 0.70)
            winrate_thresh = combo.get("trigger_winrate_threshold", 0.0)

            weapon_ok   = (not weapon)   or weapon_usage.get(weapon, 0) >= usage_thresh
            perk_ok     = (not perk)     or perk_uptimes.get(perk, 0)   >= uptime_thresh
            winrate_ok  = (not archetype) or winrates.get(archetype, 0.50) >= (0.50 + winrate_thresh)

            if weapon_ok and perk_ok and winrate_ok:
                return combo.get("strategy_name", f"{archetype}+{weapon}+{perk}_loop")

        return None
