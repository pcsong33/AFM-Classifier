from pytest import approx

from dr.load import load_table
from dr.utils import fit_and_subtract
from dr.S_parameters import *


def load_test_data(ver: int):
    VER_MAP = {
        1: 'data/cells_c-srk.txt',
        2: 'data/cells_h7-29.txt'
    }
    if ver < 3:
        test_data = load_table(VER_MAP[ver])['Adhesion'].values
        M = int(np.sqrt(test_data.shape[0]))
        fitted = fit_and_subtract(test_data).reshape(M, M)
    else:
        if ver == 3:
            dim = (512, 512)
        else:
            dim = (200, 200)
        fitted = np.ones(dim)

    return fitted


fitted_1 = load_test_data(1)
fitted_2 = load_test_data(2)
fitted_3 = load_test_data(3)
fitted_4 = load_test_data(4)


def test_S_a():
    assert S_a(fitted_1) == approx(0.756, abs=1e-2), "ver-1 correctness failed"
    assert S_a(fitted_2) == approx(0.844, abs=1e-2), "ver-2 correctness failed"


def test_S_q():
    assert S_q(fitted_1) == approx(0.950, abs=1e-2), "ver-1 correctness failed"
    assert S_q(fitted_2) == approx(1.04, abs=1e-2), "ver-2 correctness failed"


def test_S_sk():
    assert S_sk(fitted_1) == approx(
        0.0277, abs=1e-2), "ver-1 correctness failed"
    assert S_sk(fitted_2) == approx(
        0.671, abs=1e-2), "ver-2 correctness failed"


def test_S_ku():
    assert S_ku(fitted_1) == approx(3.36, abs=1e-2), "ver-1 correctness failed"
    assert S_ku(fitted_2) == approx(3.12, abs=1e-2), "ver-2 correctness failed"


def test_S_z():
    assert S_z(fitted_1) == approx(12.7, abs=2e-2), "ver-1 correctness failed"
    assert S_z(fitted_2) == approx(8.49, abs=2e-2), "ver-2 correctness failed"


def test_S_v():
    assert S_v(fitted_1) == approx(-6.66, abs=2e-2), "ver-1 correctness failed"
    assert S_v(fitted_2) == approx(-3.07, abs=2e-2), "ver-2 correctness failed"


def test_S_p():
    assert S_p(fitted_1) == approx(6.05, abs=2e-2), "ver-1 correctness failed"
    assert S_p(fitted_2) == approx(5.43, abs=2e-2), "ver-2 correctness failed"


def test_S_sc():
    assert S_sc(fitted_1) == approx(2373, abs=2e-2), "ver-1 correctness failed"
    assert S_sc(fitted_2) == approx(3989, abs=2e-2), "ver-2 correctness failed"


def test_S_dq():
    assert S_dq(fitted_1) == approx(88.3, abs=3), "ver-1 correctness failed"
    assert S_dq(fitted_2) == approx(88.8, abs=3), "ver-2 correctness failed"
