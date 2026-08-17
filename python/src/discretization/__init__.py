"""Maximum-entropy discretization of stochastic processes."""

__version__ = "0.1.0"

from .distributions import moment_matched_distribution
from .markov import (
    cir_transition_density,
    discrete_ar,
    discrete_cir,
)
from .maxent import (
    entropy_objective,
    maximum_entropy_weights,
)
from .processes import (
    discrete_gaussian_mixture_ar,
    discrete_stochastic_volatility_ar,
    discrete_var,
)
from .quadrature import (
    clenshaw_curtis,
    data_driven_gaussian_quadrature,
    gauss_hermite,
    gauss_legendre,
    gaussian_mixture_quadrature,
)

__all__ = [
    "__version__",
    "cir_transition_density",
    "clenshaw_curtis",
    "data_driven_gaussian_quadrature",
    "discrete_ar",
    "discrete_cir",
    "discrete_gaussian_mixture_ar",
    "discrete_stochastic_volatility_ar",
    "discrete_var",
    "entropy_objective",
    "gauss_hermite",
    "gauss_legendre",
    "gaussian_mixture_quadrature",
    "maximum_entropy_weights",
    "moment_matched_distribution",
]
