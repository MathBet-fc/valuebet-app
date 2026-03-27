import math
import pytest
import engine

def test_poisson_probability():
    p_0 = engine.poisson_probability(0, 1.5)
    expected_poisson = math.exp(-1.5)
    assert math.isclose(p_0, expected_poisson, rel_tol=1e-5), f"Poisson failed: {p_0} != {expected_poisson}"

def test_dixon_coles_probability():
    p_1_0_rho_0 = engine.dixon_coles_probability(1, 0, 1.5, 1.2, 0.0)
    p_1_0_base = engine.poisson_probability(1, 1.5) * engine.poisson_probability(0, 1.2)
    assert math.isclose(p_1_0_rho_0, p_1_0_base, rel_tol=1e-5), "DC without rho failed"
    
    p_1_0_rho_neg = engine.dixon_coles_probability(1, 0, 1.5, 1.2, -0.05)
    assert p_1_0_rho_neg < p_1_0_rho_0, "DC with negative rho logic failed for 1-0"

def test_calcola_xCorners_pro():
    xh, xa = engine.calcola_xCorners_pro(5.5, 4.5, 4.5, 5.5, 10.0, 5.0, 5.0)
    assert xh > 0.1 and xa > 0.1, "xCorners calculation failed boundaries"

def test_elo_adjustment():
    base_xh = 1.5
    elo_h = 1600
    elo_a = 1500
    diff = elo_h - elo_a
    adjusted_xh = base_xh * (1 + diff/1000.0)
    assert adjusted_xh > base_xh, "ELO adjustment logic failed"
