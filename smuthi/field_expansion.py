import numpy as np


class PlaneWaveExpansion:
    r"""A class to manage plane wave expansions like

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{j=1}^2 \int_0^{2\pi} \mathrm{d}\alpha \int_0^{2\pi} \mathrm{d}\kappa \, \kappa \,
        g_j^\pm(\kappa, \alpha) \mathbf{E}^\pm_j(\kappa, \alpha; \mathbf{r})

    Args:
        n_effective (ndarray):      :math:`n_\mathrm{eff} = \kappa / \omega`, sets PlaneWavePattern.n_effective
                                    attribute to input

        azimuthal_angles (ndarray): :math:`\alpha` , sets PlaneWavePattern.azimuthal_angles attribute to input
    """
    def __init__(self, n_effective=None, azimuthal_angles=None):
        self.n_effective = n_effective
        self.azimuthal_angles = azimuthal_angles
        self.coefficients = np.zeros((2, 2, len(self.n_effective), len(self.azimuthal_angles)), dtype=complex)


class SphericalWaveExpansion:
    r"""A class to manage plane wave expansions like

    .. math::
        \mathbf{E}(\mathbf{r}) = \sum_{j=1}^2 \int_0^{2\pi} \mathrm{d}\alpha \int_0^{2\pi} \mathrm{d}\kappa \, \kappa \,
        g_j^\pm(\kappa, \alpha) \mathbf{E}^\pm_j(\kappa, \alpha; \mathbf{r})

    Args:
        n_effective (ndarray):      :math:`n_\mathrm{eff} = \kappa / \omega`, sets PlaneWavePattern.n_effective
                                    attribute to input

        azimuthal_angles (ndarray): :math:`\alpha` , sets PlaneWavePattern.azimuthal_angles attribute to input
    """
    def __init__(self, n_effective=None, azimuthal_angles=None):
        self.n_effective = n_effective
        self.azimuthal_angles = azimuthal_angles
