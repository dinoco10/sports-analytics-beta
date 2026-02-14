"""
═══════════════════════════════════════════════════════════════════════
Model Weights — YOUR baseball philosophy encoded as numbers.

RULES:
1. Main weights must sum to 1.00
2. Pitching underlying sub-weights must sum to 1.00
3. If you change one weight, adjust others to compensate
═══════════════════════════════════════════════════════════════════════
"""


def get_main_weights():
    """
    Main component weights for the power rankings model.
    These determine how much each team attribute matters.
    """
    return {
        # ── Offense ───────────────────────────────
        'lineup_contact': 0.13,      # OBP/contacto > power (TU PRIORIDAD)
        'lineup_power': 0.09,        # HR, SLG, run production
        'speed_baserunning': 0.09,   # Atletismo, robos, bases extra
        
        # ── Pitching ─────────────────────────────
        'bullpen_depth': 0.13,       # MUY importante (ganar 2-1)
        'pitching_underlying': 0.12, # DATA: K-BB%, xERA gap, BABIP, etc.
        'rotation_strength': 0.08,   # Eye test de rotación
        
        # ── Defense & Depth ──────────────────────
        'defense': 0.11,             # Defensa general
        'depth': 0.07,               # Sobrevivir 162 juegos
        
        # ── Intangibles ──────────────────────────
        'momentum': 0.06,            # Vibes, trayectoria
        'farm_system': 0.05,         # Call-ups potenciales
        'manager_coaching': 0.02,    # Bajo (difícil de evaluar)
        
        # ── Personal ─────────────────────────────
        'personal_weight': 0.05,     # Tu bias, eye test
    }


def get_pitching_underlying_sub_weights():
    """
    Sub-weights WITHIN the pitching_underlying component.
    These determine which underlying metric matters most.
    
    The effective weight of each metric in the total model is:
    effective = sub_weight × main_weight('pitching_underlying') × SOURCE_WEIGHTS['user_criteria']
    
    Example: K_BB_pct effective = 0.20 × 0.12 × 0.20 = 0.0048 (0.48%)
    All 8 metrics combined = 0.12 × 0.20 = 0.024 (2.4% of total model)
    """
    return {
        # ── Tier 1: Most Predictive (53%) ─────────
        'K_BB_pct': 0.20,        # Best single pitcher skill metric
        'xERA_gap': 0.18,        # Luck/regression detector
        'FIP_era_gap': 0.15,     # ERA sustainability
        
        # ── Tier 2: Very Important (33%) ──────────
        'BABIP_allowed': 0.13,   # Balls in play luck
        'HR_FB_pct': 0.12,       # Most volatile year-to-year
        'GB_pct': 0.08,          # Stable skill metric
        
        # ── Tier 3: Complementary (14%) ───────────
        'barrel_pct': 0.08,      # Contact quality allowed
        'swstr_pct': 0.06,       # Stuff quality proxy
    }


def validate_weights():
    """Run this to check your weights are valid."""
    main = get_main_weights()
    sub = get_pitching_underlying_sub_weights()
    
    main_total = sum(main.values())
    sub_total = sum(sub.values())
    
    errors = []
    
    if abs(main_total - 1.0) > 0.01:
        errors.append(f"Main weights sum to {main_total:.3f}, should be 1.00")
    if abs(sub_total - 1.0) > 0.01:
        errors.append(f"Sub weights sum to {sub_total:.3f}, should be 1.00")
    if any(v < 0 for v in main.values()):
        errors.append("Negative main weights found")
    if any(v < 0 for v in sub.values()):
        errors.append("Negative sub weights found")
    
    if errors:
        for e in errors:
            print(f"❌ {e}")
        return False
    
    print("✅ All weights valid")
    print(f"   Main: {main_total:.2f} | Sub: {sub_total:.2f}")
    return True


if __name__ == "__main__":
    validate_weights()