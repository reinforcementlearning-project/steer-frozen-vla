import pytest
from evaluation.summarize import wilson


def test_wilson_boundary_and_symmetry():
    lo, hi = wilson(0, 50)
    assert lo == pytest.approx(0, abs=1e-12)
    assert hi == pytest.approx(.071347599, abs=1e-8)
    a, b = wilson(50, 50)
    assert a == pytest.approx(1-hi)
    assert b == pytest.approx(1.)
    assert sum(wilson(25, 50)) == pytest.approx(1.)
