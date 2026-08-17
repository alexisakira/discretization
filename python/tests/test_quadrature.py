from pathlib import Path

import numpy as np
from scipy.io import loadmat

from discretization import (
    clenshaw_curtis,
    data_driven_gaussian_quadrature,
    gauss_hermite,
    gauss_legendre,
    gaussian_mixture_quadrature,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = loadmat(REPOSITORY_ROOT / "reference_data" / "matlab_r2024b_milestone2.mat")


def test_standard_quadrature_rules_match_matlab():
    order = int(FIXTURE["quadratureOrder"].item())
    interval = tuple(FIXTURE["quadratureInterval"].reshape(-1))

    hermite_nodes, hermite_weights = gauss_hermite(order)
    legendre_nodes, legendre_weights = gauss_legendre(order, interval)
    clenshaw_nodes, clenshaw_weights = clenshaw_curtis(order, interval)

    np.testing.assert_allclose(hermite_nodes, FIXTURE["hermiteNodes"].reshape(-1), atol=1e-14)
    np.testing.assert_allclose(hermite_weights, FIXTURE["hermiteWeights"].reshape(-1), atol=1e-14)
    np.testing.assert_allclose(legendre_nodes, FIXTURE["legendreNodes"].reshape(-1), atol=1e-14)
    np.testing.assert_allclose(legendre_weights, FIXTURE["legendreWeights"].reshape(-1), atol=1e-14)
    np.testing.assert_allclose(clenshaw_nodes, FIXTURE["clenshawCurtisNodes"].reshape(-1), atol=1e-14)
    np.testing.assert_allclose(clenshaw_weights, FIXTURE["clenshawCurtisWeights"].reshape(-1), atol=1e-14)


def test_gaussian_mixture_quadrature_matches_matlab():
    nodes, weights = gaussian_mixture_quadrature(
        FIXTURE["mixtureCoeff"].reshape(-1),
        FIXTURE["mixtureMean"].reshape(-1),
        FIXTURE["mixtureStd"].reshape(-1),
        int(FIXTURE["quadratureOrder"].item()),
    )

    np.testing.assert_allclose(nodes, FIXTURE["mixtureNodes"].reshape(-1), atol=1e-11)
    np.testing.assert_allclose(weights, FIXTURE["mixtureWeights"].reshape(-1), atol=1e-11)


def test_data_driven_gaussian_quadrature_matches_matlab():
    nodes, weights = data_driven_gaussian_quadrature(
        FIXTURE["npgqData"].reshape(-1), int(FIXTURE["npgqOrder"].item())
    )

    np.testing.assert_allclose(nodes, FIXTURE["npgqNodes"].reshape(-1), atol=1e-12)
    np.testing.assert_allclose(weights, FIXTURE["npgqWeights"].reshape(-1), atol=1e-12)
