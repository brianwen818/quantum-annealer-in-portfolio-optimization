import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qaportfolio import backtest, mvo, qubo  # noqa: E402
from qaportfolio.schedule import first_friday  # noqa: E402


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n = 8
    F = rng.normal(size=(n, 2)) * 0.05
    cov = F @ F.T + np.diag(rng.uniform(0.005, 0.03, n))
    idx = [str(i) for i in range(n)]
    return pd.Series(rng.normal(0.03, 0.05, n), index=idx), pd.DataFrame(cov, idx, idx)


@pytest.mark.parametrize("P,expected", [(20, [1, 2, 4, 8, 5]), (100, [1, 2, 4, 8, 16, 32, 37]), (1, [1])])
def test_binary_coefficients_cover_range(P, expected):
    c = qubo.binary_coefficients(P)
    assert list(c) == expected
    assert c.sum() == P


@pytest.mark.parametrize("gamma", [None, 5.0])
def test_qubo_energy_matches_objective_plus_penalty(problem, gamma):
    mu, cov = problem
    P = 20
    prob = qubo.build_qubo(mu.values, cov.values, P, gamma, penalty_factor=2.0)
    rng = np.random.default_rng(1)
    K = len(prob.coefs)
    for _ in range(20):
        b = rng.integers(0, 2, size=len(mu) * K)
        x = b.reshape(len(mu), K) @ prob.coefs
        w = x / P
        obj = w @ cov.values @ w if gamma is None else mvo.mv_objective(w, mu.values, cov.values, gamma)
        expected = obj + prob.lagrange * (x.sum() - P) ** 2
        energy = prob.bqm.energy(dict(enumerate(b)))
        assert energy == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_qubo_solution_is_feasible(problem):
    mu, cov = problem
    sol = qubo.solve(mu, cov, 20, 5.0, "neal", {"num_reads": 20, "num_sweeps": 500}, 2.0, seed=0)
    assert sol.weights.sum() == pytest.approx(1.0)
    assert np.allclose(sol.weights * 20, np.round(sol.weights * 20))
    assert sol.feasible_rate > 0.5


def test_round_to_lots_sums_to_one(problem):
    mu, cov = problem
    w = mvo.max_utility(mu, cov, 5.0)
    r = mvo.round_to_lots(w, 20)
    assert r.sum() == pytest.approx(1.0)
    assert (np.abs(r - w) <= 1 / 20 + 1e-12).all()


def test_first_friday():
    assert first_friday(2025, 3).day == 7
    assert first_friday(2025, 6).day == 6


def test_rebalance_cost():
    prev = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 1.0})
    turnover, cost = backtest.rebalance_cost(prev, target, fee=0.001, tax=0.003)
    assert turnover == pytest.approx(0.5)
    assert cost == pytest.approx(0.5 * 0.001 + 0.5 * 0.004)
