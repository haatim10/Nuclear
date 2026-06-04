#!/usr/bin/env python3
"""
Ahrens et al. (BNL E734) neutrino-proton elastic NC cross section.

Standard Model, FIXED neutrino energy calculation of dsigma/dQ^2 for
    nu p   -> nu p
    nubar p -> nubar p
using ONLY the equations from Ahrens et al., Phys. Rev. D 35, 785 (1987):

    Eq. (1.7)   alpha, beta, gamma, delta couplings
    Eq. (1.9)   F1+F2 and F2 in terms of vector form factors
    Eq. (1.10)  isovector / isoscalar dipole form factors GV3, GV0, FV3, FV0
    Eq. (1.11)  GA = beta*GA3 + delta*GA0
    Eq. (1.12)  GA3 dipole (with the 1/2 factor)
    Eq. (1.14)  differential cross section
    Eq. (1.15)  structure functions A, B, C and s - u

IMPORTANT: This is a fixed-energy (monochromatic E_nu) Standard Model
calculation. It is NOT the full beam-flux-averaged reproduction of the
paper's Fig. 35 (the paper integrates over the BNL neutrino spectrum).

Output (under plots/):
    ahrens_sm_dsigma.png   -- dsigma/dQ^2 vs Q^2 for nu and nubar
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

PROJECT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_DIR / "plots"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

# numpy.trapz was renamed to numpy.trapezoid in NumPy 2.x and removed under the
# old name. Use whichever trapezoidal-integration function this NumPy provides.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# =============================================================================
# Physical constants
# =============================================================================


@dataclass(frozen=True)
class PhysicsConstants:
    """
    Inputs for nu p / nubar p neutral-current elastic scattering on a proton.

    G_F          : Fermi coupling constant [GeV^-2] (natural units hbar = c = 1)
    M_p          : proton mass [GeV]
    sin2_theta_W : weak mixing angle sin^2(theta_W)
    M_V          : vector dipole mass [GeV] (Eq. 1.10)
    M_A          : axial dipole mass [GeV]  (Eq. 1.12)
    g_A          : axial coupling g_A(0) from neutron beta decay
    kappa_p      : proton anomalous magnetic moment
    kappa_n      : neutron anomalous magnetic moment
    E_nu         : incident neutrino energy [GeV] (single fixed value)
    """

    G_F: float = 1.166e-5
    M_p: float = 0.938
    sin2_theta_W: float = 0.220
    M_V: float = 0.84
    M_A: float = 1.06
    g_A: float = 1.26
    kappa_p: float = 1.793
    kappa_n: float = -1.913
    E_nu: float = 1.25

    @property
    def hbar_c_GeV_fm(self) -> float:
        """hbar*c in GeV*fm."""
        return 0.1973269804

    @property
    def GeV2_to_cm2(self) -> float:
        """Multiply a GeV^-2 cross section by this to get cm^2: (hbar c)^2."""
        return self.hbar_c_GeV_fm**2 * 1.0e-26


CONST = PhysicsConstants()

# Q^2 grid for the theory curves [GeV^2]
Q2_MIN, Q2_MAX, Q2_NPTS = 0.1, 1.2, 250

# Approximate neutrino flux used ONLY for the flux-averaging demonstration.
# This is a generic Gaussian-like shape, NOT the real BNL E734 beam spectrum.
FLUX_E_MIN = 0.2     # GeV
FLUX_E_MAX = 5.0     # GeV
FLUX_E_CENTER = 1.25  # GeV
FLUX_E_WIDTH = 0.6   # GeV
FLUX_NPTS = 400

# sin^2(theta_W) sensitivity sweep (fixed E_nu; does not change CONST)
SIN2_SWEEP_MIN = 0.1
SIN2_SWEEP_MAX = 0.3
SIN2_SWEEP_NPTS = 9  # 0.1, 0.125, ..., 0.3


# =============================================================================
# Form factors -- transcribed exactly from Ahrens Eqs. (1.7), (1.9)-(1.12)
# =============================================================================


def vector_form_factors(
    q2: np.ndarray,
    const: PhysicsConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Proton NC vector form factors F1(Q^2), F2(Q^2).

    Eq. (1.7) couplings:
        alpha = 1 - 2 sin^2(theta_W)
        gamma = -(2/3) sin^2(theta_W)

    Eq. (1.10) dipole isovector (^3) / isoscalar (^0) form factors:
        tau = Q^2 / (4 M_p^2)
        GV3 = 0.5 (1 + kappa_p - kappa_n) / (1 + Q^2/M_V^2)^2
        GV0 = 1.5 (1 + kappa_p + kappa_n) / (1 + Q^2/M_V^2)^2
        FV3 = 0.5 (kappa_p - kappa_n) / ((1 + tau)(1 + Q^2/M_V^2)^2)
        FV0 = 1.5 (kappa_p + kappa_n) / ((1 + tau)(1 + Q^2/M_V^2)^2)

    Eq. (1.9):
        F1 + F2 = alpha GV3 + gamma GV0
        F2      = alpha FV3 + gamma FV0
        F1      = (F1 + F2) - F2
    """
    s2w = const.sin2_theta_W
    alpha = 1.0 - 2.0 * s2w
    gamma = -(2.0 / 3.0) * s2w

    tau = q2 / (4.0 * const.M_p**2)
    kp = const.kappa_p
    kn = const.kappa_n
    dip_v = 1.0 / (1.0 + q2 / const.M_V**2) ** 2

    gv3 = 0.5 * (1.0 + kp - kn) * dip_v
    gv0 = 1.5 * (1.0 + kp + kn) * dip_v
    fv3 = 0.5 * (kp - kn) * dip_v / (1.0 + tau)
    fv0 = 1.5 * (kp + kn) * dip_v / (1.0 + tau)

    f1_plus_f2 = alpha * gv3 + gamma * gv0
    f2 = alpha * fv3 + gamma * fv0
    f1 = f1_plus_f2 - f2
    return f1, f2


def axial_form_factor(
    q2: np.ndarray,
    const: PhysicsConstants,
) -> np.ndarray:
    """
    Axial form factor GA(Q^2).

    Eq. (1.7):  beta = 1, delta = 0
    Eq. (1.12): GA3 = 0.5 g_A / (1 + Q^2/M_A^2)^2
                GA0 = 0
    Eq. (1.11): GA = beta GA3 + delta GA0
    """
    beta = 1.0
    delta = 0.0
    ga3 = 0.5 * const.g_A / (1.0 + q2 / const.M_A**2) ** 2
    ga0 = 0.0
    return beta * ga3 + delta * ga0


# =============================================================================
# Structure functions A, B, C and kinematics -- Ahrens Eq. (1.15)
# =============================================================================


def kinematic_s_minus_u(
    q2: np.ndarray,
    E_nu: float,
    M_p: float,
) -> np.ndarray:
    """Eq. (1.15): s - u = 4 M_p E_nu - Q^2."""
    return 4.0 * M_p * E_nu - q2


def coefficient_a(
    q2: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    g_a: np.ndarray,
    M_p: float,
) -> np.ndarray:
    """
    Eq. (1.15):
        A = (Q^2/M_p^2) [ GA^2 (1 + Q^2/4M_p^2)
                          - F1^2 (1 - Q^2/4M_p^2)
                          + F2^2 (1 - Q^2/4M_p^2)(Q^2/4M_p^2)
                          + F1 F2 (Q^2/M_p^2) ]
    """
    tau = q2 / (4.0 * M_p**2)
    q2_mp2 = q2 / M_p**2
    return q2_mp2 * (
        g_a**2 * (1.0 + tau)
        - f1**2 * (1.0 - tau)
        + f2**2 * (1.0 - tau) * tau
        + f1 * f2 * q2_mp2
    )


def coefficient_b(
    q2: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    g_a: np.ndarray,
    M_p: float,
) -> np.ndarray:
    """Eq. (1.15): B = (Q^2/M_p^2) GA (F1 + F2)."""
    return (q2 / M_p**2) * g_a * (f1 + f2)


def coefficient_c(
    q2: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    g_a: np.ndarray,
    M_p: float,
) -> np.ndarray:
    """Eq. (1.15): C = (1/4) [ GA^2 + F1^2 + F2^2 (Q^2/4M_p^2) ]."""
    tau = q2 / (4.0 * M_p**2)
    return 0.25 * (g_a**2 + f1**2 + f2**2 * tau)


# =============================================================================
# Differential cross section -- Ahrens Eq. (1.14)
# =============================================================================


def dsigma_dq2_natural(
    q2: np.ndarray,
    const: PhysicsConstants,
    beam: Literal["nu", "nubar"],
    E_nu: float | np.ndarray | None = None,
) -> np.ndarray:
    """
    dsigma/dQ^2 in natural units [GeV^-4], Eq. (1.14):

        dsigma/dQ^2 = G_F^2 M_p^2 / (8 pi E_nu^2)
                      * [ A +/- B (s-u)/M_p^2 + C (s-u)^2/M_p^4 ]

    Plus sign for neutrinos, minus sign for antineutrinos (exact paper sign).

    E_nu : optional neutrino energy [GeV]. If None, const.E_nu is used (the
    fixed-energy calculation is unchanged). A scalar or array may be passed;
    passing a scalar q2 with an array of E_nu values (as the flux-averaging
    routine does) returns dsigma/dQ^2 evaluated at each E_nu.

    NOTE on prefactor: the paper's Eq. (1.14) shows G_F^2 * M_p^2 (the typed
    instruction read M_p); M_p^2 is used here because that is what the paper
    prints and it gives the correct GeV^-4 units for dsigma/dQ^2.
    """
    e_nu = const.E_nu if E_nu is None else E_nu

    f1, f2 = vector_form_factors(q2, const)
    g_a = axial_form_factor(q2, const)

    a = coefficient_a(q2, f1, f2, g_a, const.M_p)
    b = coefficient_b(q2, f1, f2, g_a, const.M_p)
    c = coefficient_c(q2, f1, f2, g_a, const.M_p)

    s_minus_u = kinematic_s_minus_u(q2, e_nu, const.M_p)
    sign = 1.0 if beam == "nu" else -1.0  # + for nu, - for nubar

    bracket = (
        a
        + sign * b * s_minus_u / const.M_p**2
        + c * s_minus_u**2 / const.M_p**4
    )
    prefactor = const.G_F**2 * const.M_p**2 / (8.0 * np.pi * e_nu**2)
    return prefactor * bracket


def dsigma_dq2_cm2(
    q2: np.ndarray,
    const: PhysicsConstants,
    beam: Literal["nu", "nubar"],
) -> np.ndarray:
    """dsigma/dQ^2 in cm^2/(GeV/c)^2 (natural result times (hbar c)^2)."""
    return dsigma_dq2_natural(q2, const, beam) * const.GeV2_to_cm2


# =============================================================================
# Flux-averaging demonstration (APPROXIMATION -- not the real Ahrens beam)
# =============================================================================


def neutrino_flux(e_nu: np.ndarray) -> np.ndarray:
    """
    Approximate Gaussian-like neutrino flux phi(E_nu).

    Centered at FLUX_E_CENTER with width FLUX_E_WIDTH. This is a generic
    demonstration shape, NOT the measured BNL E734 spectrum.
    """
    return np.exp(-0.5 * ((e_nu - FLUX_E_CENTER) / FLUX_E_WIDTH) ** 2)


def flux_average_dsigma(
    q2_values: np.ndarray,
    beam: Literal["nu", "nubar"],
    const: PhysicsConstants = CONST,
) -> np.ndarray:
    """
    Flux-averaged <dsigma/dQ^2>(Q^2) in natural units [GeV^-4].

    APPROXIMATION ONLY. For each Q^2 this computes

        <dsigma/dQ^2> = integral[ phi(E) dsigma/dQ^2(E, Q^2) dE ]
                        / integral[ phi(E) dE ]

    using an approximate Gaussian-like flux phi(E) (neutrino_flux) over
    E_nu in [FLUX_E_MIN, FLUX_E_MAX], integrated with numpy.trapz.

    A physical cutoff is applied: energies with E_nu <= Q^2/(4 M_p) are
    ignored (the elastic kinematics require s - u = 4 M_p E_nu - Q^2 > 0).
    """
    q2_values = np.atleast_1d(np.asarray(q2_values, dtype=float))
    e_grid = np.linspace(FLUX_E_MIN, FLUX_E_MAX, FLUX_NPTS)
    flux = neutrino_flux(e_grid)
    flux_integral = _trapz(flux, e_grid)

    result = np.empty_like(q2_values)
    for i, q2 in enumerate(q2_values):
        # dsigma/dQ^2 at fixed Q^2 across every flux energy
        dsig = dsigma_dq2_natural(float(q2), const, beam, E_nu=e_grid)
        # physical cutoff: ignore E_nu <= Q^2/(4 M_p)
        below_threshold = e_grid <= q2 / (4.0 * const.M_p)
        dsig = np.where(below_threshold, 0.0, dsig)
        numerator = _trapz(dsig * flux, e_grid)
        result[i] = numerator / flux_integral
    return result


def flux_average_dsigma_cm2(
    q2_values: np.ndarray,
    beam: Literal["nu", "nubar"],
    const: PhysicsConstants = CONST,
) -> np.ndarray:
    """Flux-averaged <dsigma/dQ^2> in cm^2/(GeV/c)^2."""
    return flux_average_dsigma(q2_values, beam, const) * const.GeV2_to_cm2


# =============================================================================
# Proton / neutron NC form factors -- KamLAND Sec. II, Eqs. (2)-(6)
# (Abe et al., KamLAND Collaboration, Phys. Rev. D 107, 072006 (2023))
#
# These provide the proton/neutron neutral-current form factors. They are then
# fed into the SAME Ahrens Eq. (1.14)/(1.15) cross-section engine above. The
# strange axial coupling gsA and strange vector form factors F1s, F2s are kept
# as EXPLICIT parameters (defaulted to zero), not silently assumed.
# =============================================================================


def nc_form_factors_kamland(
    q2: np.ndarray,
    const: PhysicsConstants,
    target: Literal["proton", "neutron"] = "proton",
    gsA: float = 0.0,
    F1s: float = 0.0,
    F2s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Proton/neutron neutral-current form factors F1_NC, F2_NC, GA_NC.

    Source: KamLAND Sec. II, Eqs. (2)-(6). The cross-section structure itself
    is still Ahrens Eq. (1.14)/(1.15); only the form factors come from KamLAND.

    Electromagnetic form factors -- DIPOLE approximation (older/common form;
    KamLAND uses the more modern BBBA05 parametrization). TODO: replace with
    BBBA05 if required.
        GD   = 1 / (1 + Q^2/M_V^2)^2
        tau  = Q^2 / (4 M^2)
        GEp  = GD ,  GMp = (1 + kappa_p) GD
        GEn  = 0  ,  GMn = kappa_n GD

    Sachs -> Dirac/Pauli (KamLAND Eqs. 5-6):
        F1 = (GE + tau GM) / (1 + tau)
        F2 = (GM - GE)     / (1 + tau)

    KamLAND Eq. (3) charged-current vector form factors:
        F_CC_1 = F1p - F1n ,  F_CC_2 = F2p - F2n

    KamLAND Eq. (4) neutral-current vector form factors (strange kept explicit):
        proton:  F1_NC = +0.5 F_CC_1 - 2 sin2thetaW F1p - 0.5 F1s
                 F2_NC = +0.5 F_CC_2 - 2 sin2thetaW F2p - 0.5 F2s
        neutron: F1_NC = -0.5 F_CC_1 - 2 sin2thetaW F1n - 0.5 F1s
                 F2_NC = -0.5 F_CC_2 - 2 sin2thetaW F2n - 0.5 F2s

    KamLAND Eq. (2) neutral-current axial form factor (gsA explicit):
        proton:  GA_NC = 0.5 (+gA - gsA) / (1 + Q^2/M_A^2)^2
        neutron: GA_NC = 0.5 (-gA - gsA) / (1 + Q^2/M_A^2)^2

    Notes
    -----
    - M is taken as const.M_p (the nucleon mass used by the Ahrens engine).
    - gsA defaults to 0.0 == baseline / no strange axial contribution.
    - F1s, F2s default to 0.0 following KamLAND's statement that the strange
      vector form factors are consistent with zero.
    """
    s2w = const.sin2_theta_W
    M = const.M_p

    gd = 1.0 / (1.0 + q2 / const.M_V**2) ** 2
    tau = q2 / (4.0 * M**2)

    # electromagnetic Sachs form factors (dipole approximation)
    gep = gd
    gmp = (1.0 + const.kappa_p) * gd
    gen = 0.0 * gd
    gmn = const.kappa_n * gd

    # Sachs -> Dirac/Pauli
    f1p = (gep + tau * gmp) / (1.0 + tau)
    f2p = (gmp - gep) / (1.0 + tau)
    f1n = (gen + tau * gmn) / (1.0 + tau)
    f2n = (gmn - gen) / (1.0 + tau)

    # KamLAND Eq. (3): charged-current vector form factors
    f_cc_1 = f1p - f1n
    f_cc_2 = f2p - f2n

    # KamLAND Eq. (4) + Eq. (2)
    dip_a = 1.0 / (1.0 + q2 / const.M_A**2) ** 2
    if target == "proton":
        f1_nc = 0.5 * f_cc_1 - 2.0 * s2w * f1p - 0.5 * F1s
        f2_nc = 0.5 * f_cc_2 - 2.0 * s2w * f2p - 0.5 * F2s
        ga_nc = 0.5 * (const.g_A - gsA) * dip_a
    elif target == "neutron":
        f1_nc = -0.5 * f_cc_1 - 2.0 * s2w * f1n - 0.5 * F1s
        f2_nc = -0.5 * f_cc_2 - 2.0 * s2w * f2n - 0.5 * F2s
        ga_nc = 0.5 * (-const.g_A - gsA) * dip_a
    else:
        raise ValueError(f"target must be 'proton' or 'neutron', got {target!r}")

    return f1_nc, f2_nc, ga_nc


def dsigma_dq2_nc_target(
    q2: np.ndarray,
    const: PhysicsConstants,
    beam: Literal["nu", "nubar"] = "nu",
    target: Literal["proton", "neutron"] = "proton",
    E_nu: float | np.ndarray | None = None,
    gsA: float = 0.0,
) -> np.ndarray:
    """
    dsigma/dQ^2 [GeV^-4] for NC elastic scattering on a proton or neutron.

    Form factors: KamLAND Sec. II (nc_form_factors_kamland).
    Cross-section engine: Ahrens Eq. (1.14)/(1.15) (same A, B, C as above).
    Plus sign for neutrino, minus sign for antineutrino.
    """
    e_nu = const.E_nu if E_nu is None else E_nu

    f1, f2, g_a = nc_form_factors_kamland(q2, const, target=target, gsA=gsA)

    a = coefficient_a(q2, f1, f2, g_a, const.M_p)
    b = coefficient_b(q2, f1, f2, g_a, const.M_p)
    c = coefficient_c(q2, f1, f2, g_a, const.M_p)

    s_minus_u = kinematic_s_minus_u(q2, e_nu, const.M_p)
    sign = 1.0 if beam == "nu" else -1.0  # + for nu, - for nubar

    bracket = (
        a
        + sign * b * s_minus_u / const.M_p**2
        + c * s_minus_u**2 / const.M_p**4
    )
    prefactor = const.G_F**2 * const.M_p**2 / (8.0 * np.pi * e_nu**2)
    return prefactor * bracket


def dsigma_dq2_nc_target_cm2(
    q2: np.ndarray,
    const: PhysicsConstants,
    beam: Literal["nu", "nubar"] = "nu",
    target: Literal["proton", "neutron"] = "proton",
    gsA: float = 0.0,
) -> np.ndarray:
    """dsigma/dQ^2 on proton/neutron in cm^2/(GeV/c)^2."""
    return (
        dsigma_dq2_nc_target(q2, const, beam=beam, target=target, gsA=gsA)
        * const.GeV2_to_cm2
    )


# =============================================================================
# Plotting
# =============================================================================


def plot_main_cross_section(const: PhysicsConstants = CONST) -> Path:
    """Standard Model dsigma/dQ^2 vs Q^2 for nu and nubar (fixed E_nu)."""
    q2 = np.linspace(Q2_MIN, Q2_MAX, Q2_NPTS)

    dsig_nu = dsigma_dq2_cm2(q2, const, "nu")
    dsig_nubar = dsigma_dq2_cm2(q2, const, "nubar")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        q2,
        dsig_nu,
        color="C0",
        lw=2,
        label=r"$\nu p \to \nu p$",
    )
    ax.plot(
        q2,
        dsig_nubar,
        color="C1",
        lw=2,
        label=r"$\bar{\nu} p \to \bar{\nu} p$",
    )

    ax.set_xlabel(r"$Q^2$ [(GeV/$c$)$^2$]")
    ax.set_ylabel(r"$d\sigma/dQ^2$ [cm$^2$/(GeV/$c$)$^2$]")
    ax.set_title(
        "Ahrens Eq. (1.14) -- Standard Model, fixed-energy calculation\n"
        rf"$E_\nu={const.E_nu}$ GeV, $\sin^2\theta_W={const.sin2_theta_W}$, "
        rf"$M_V={const.M_V}$ GeV, $M_A={const.M_A}$ GeV, $g_A={const.g_A}$"
    )
    ax.set_yscale("log")
    ax.set_xlim(Q2_MIN, Q2_MAX)
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "ahrens_sm_dsigma.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_flux_averaged(const: PhysicsConstants = CONST) -> Path:
    """
    Flux-averaged <dsigma/dQ^2> vs Q^2 for nu and nubar (APPROXIMATION).

    Uses the generic Gaussian-like flux in neutrino_flux(), NOT the real
    BNL E734 beam spectrum.
    """
    q2 = np.linspace(Q2_MIN, Q2_MAX, Q2_NPTS)

    dsig_nu = flux_average_dsigma_cm2(q2, "nu", const)
    dsig_nubar = flux_average_dsigma_cm2(q2, "nubar", const)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(q2, dsig_nu, color="C0", lw=2, label=r"$\nu p \to \nu p$")
    ax.plot(q2, dsig_nubar, color="C1", lw=2, label=r"$\bar{\nu} p \to \bar{\nu} p$")

    ax.set_xlabel(r"$Q^2$ [(GeV/$c$)$^2$]")
    ax.set_ylabel(r"$\langle d\sigma/dQ^2 \rangle$ [cm$^2$/(GeV/$c$)$^2$]")
    ax.set_title(
        "Ahrens Eq. (1.14) -- APPROXIMATE flux-averaged (NOT the real beam)\n"
        rf"Gaussian flux: center {FLUX_E_CENTER} GeV, width {FLUX_E_WIDTH} GeV, "
        rf"$E_\nu \in$ [{FLUX_E_MIN}, {FLUX_E_MAX}] GeV",
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.set_xlim(Q2_MIN, Q2_MAX)
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "ahrens_flux_averaged_approx.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_sin2_theta_w_sweep(
    base_const: PhysicsConstants = CONST,
    sin2_min: float = SIN2_SWEEP_MIN,
    sin2_max: float = SIN2_SWEEP_MAX,
    n_sin2: int = SIN2_SWEEP_NPTS,
) -> Path:
    """
    sin^2(theta_W) sensitivity: fixed E_nu, sweep sin^2(theta_W) in [0.1, 0.3].

    Only alpha and gamma in Eq. (1.7) depend on sin^2(theta_W); all other
    inputs are held at base_const values. Ahrens default (0.220) is drawn
    with a thicker black line for reference.
    """
    q2 = np.linspace(Q2_MIN, Q2_MAX, Q2_NPTS)
    sin2_values = np.linspace(sin2_min, sin2_max, n_sin2)
    cmap = plt.cm.viridis
    ahrens_sin2 = base_const.sin2_theta_W

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for i, s2w in enumerate(sin2_values):
        const = replace(base_const, sin2_theta_W=float(s2w))
        color = cmap(i / max(n_sin2 - 1, 1))
        is_default = np.isclose(s2w, ahrens_sin2)
        lw = 2.5 if is_default else 1.4
        zorder = 3 if is_default else 1
        label = rf"$\sin^2\theta_W={s2w:.3f}$" + (" (Ahrens)" if is_default else "")

        dsig_nu = dsigma_dq2_cm2(q2, const, "nu")
        dsig_nubar = dsigma_dq2_cm2(q2, const, "nubar")
        plot_kw = dict(
            color="k" if is_default else color,
            lw=lw,
            zorder=zorder,
            label=label,
        )
        axes[0].plot(q2, dsig_nu, **plot_kw)
        axes[1].plot(q2, dsig_nubar, **plot_kw)

    beam_titles = [
        r"$\nu p \to \nu p$",
        r"$\bar{\nu} p \to \bar{\nu} p$",
    ]
    for ax, title in zip(axes, beam_titles):
        ax.set_title(title)
        ax.set_xlabel(r"$Q^2$ [(GeV/$c$)$^2$]")
        ax.set_yscale("log")
        ax.set_xlim(Q2_MIN, Q2_MAX)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel(r"$d\sigma/dQ^2$ [cm$^2$/(GeV/$c$)$^2$]")
    axes[0].legend(loc="upper right", fontsize=8, title=r"$\sin^2\theta_W$")

    fig.suptitle(
        rf"$\sin^2\theta_W$ sensitivity (fixed $E_\nu={base_const.E_nu}$ GeV, "
        rf"Ahrens Eqs. 1.7--1.15)",
        fontsize=11,
    )
    fig.tight_layout()

    out = PLOTS_DIR / "ahrens_sin2_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --- Proton vs neutron NC plots (KamLAND form factors, Ahrens engine) --------


def plot_proton_vs_neutron(
    beam: Literal["nu", "nubar"],
    const: PhysicsConstants = CONST,
    gsA: float = 0.0,
) -> Path:
    """
    dsigma/dQ^2 vs Q^2 for proton vs neutron NC elastic, one beam type.

    Baseline gsA = 0.0 (no strange axial contribution).
    """
    q2 = np.linspace(Q2_MIN, Q2_MAX, Q2_NPTS)
    dsig_p = dsigma_dq2_nc_target_cm2(q2, const, beam=beam, target="proton", gsA=gsA)
    dsig_n = dsigma_dq2_nc_target_cm2(q2, const, beam=beam, target="neutron", gsA=gsA)

    beam_sym = r"\nu" if beam == "nu" else r"\bar{\nu}"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(q2, dsig_p, color="C0", lw=2, label=rf"${beam_sym} p \to {beam_sym} p$")
    ax.plot(q2, dsig_n, color="C3", lw=2, label=rf"${beam_sym} n \to {beam_sym} n$")

    ax.set_xlabel(r"$Q^2$ [(GeV/$c$)$^2$]")
    ax.set_ylabel(r"$d\sigma/dQ^2$ [cm$^2$/(GeV/$c$)$^2$]")
    ax.set_title(
        "NC elastic: proton vs neutron "
        "(KamLAND Sec. II form factors, Ahrens Eq. 1.14)\n"
        rf"$E_\nu={const.E_nu}$ GeV, $\sin^2\theta_W={const.sin2_theta_W}$, "
        rf"$g_s^A={gsA}$ (baseline)",
        fontsize=10,
    )
    ax.set_yscale("log")
    ax.set_xlim(Q2_MIN, Q2_MAX)
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fname = "nu_proton_vs_neutron.png" if beam == "nu" else "nubar_proton_vs_neutron.png"
    out = PLOTS_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_neutron_to_proton_ratio(
    const: PhysicsConstants = CONST,
    gsA: float = 0.0,
) -> Path:
    """Neutron/proton dsigma/dQ^2 ratio vs Q^2 for nu and nubar (baseline gsA)."""
    q2 = np.linspace(Q2_MIN, Q2_MAX, Q2_NPTS)

    ratio_nu = (
        dsigma_dq2_nc_target(q2, const, beam="nu", target="neutron", gsA=gsA)
        / dsigma_dq2_nc_target(q2, const, beam="nu", target="proton", gsA=gsA)
    )
    ratio_nubar = (
        dsigma_dq2_nc_target(q2, const, beam="nubar", target="neutron", gsA=gsA)
        / dsigma_dq2_nc_target(q2, const, beam="nubar", target="proton", gsA=gsA)
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(q2, ratio_nu, color="C0", lw=2, label=r"$\nu$: $n/p$")
    ax.plot(q2, ratio_nubar, color="C1", lw=2, label=r"$\bar{\nu}$: $n/p$")

    ax.set_xlabel(r"$Q^2$ [(GeV/$c$)$^2$]")
    ax.set_ylabel(r"$(d\sigma/dQ^2)_n \, / \, (d\sigma/dQ^2)_p$")
    ax.set_title(
        "Neutron-to-proton NC cross-section ratio\n"
        rf"$\sin^2\theta_W={const.sin2_theta_W}$, $g_s^A={gsA}$ (baseline)",
        fontsize=10,
    )
    ax.set_xlim(Q2_MIN, Q2_MAX)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "neutron_to_proton_ratio.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_sin2_proton_neutron(
    const: PhysicsConstants = CONST,
    gsA: float = 0.0,
) -> Path:
    """
    sin^2(theta_W) sensitivity of proton and neutron NC cross sections.

    Sweep sin2thetaW = [0.10, 0.15, 0.20, 0.22, 0.25, 0.30]. Two panels
    (proton, neutron) to keep it readable. Neutrino beam, baseline gsA.
    """
    q2 = np.linspace(Q2_MIN, Q2_MAX, Q2_NPTS)
    sin2_values = [0.10, 0.15, 0.20, 0.22, 0.25, 0.30]
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for i, s2w in enumerate(sin2_values):
        c = replace(const, sin2_theta_W=s2w)
        color = cmap(i / (len(sin2_values) - 1))
        is_default = np.isclose(s2w, const.sin2_theta_W)
        lw = 2.5 if is_default else 1.5
        label = rf"$\sin^2\theta_W={s2w:.2f}$" + (" (Ahrens)" if is_default else "")
        plot_kw = dict(color="k" if is_default else color, lw=lw,
                       zorder=3 if is_default else 1, label=label)
        axes[0].plot(
            q2, dsigma_dq2_nc_target_cm2(q2, c, "nu", "proton", gsA), **plot_kw
        )
        axes[1].plot(
            q2, dsigma_dq2_nc_target_cm2(q2, c, "nu", "neutron", gsA), **plot_kw
        )

    for ax, title in zip(axes, [r"$\nu p \to \nu p$", r"$\nu n \to \nu n$"]):
        ax.set_title(title)
        ax.set_xlabel(r"$Q^2$ [(GeV/$c$)$^2$]")
        ax.set_yscale("log")
        ax.set_xlim(Q2_MIN, Q2_MAX)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel(r"$d\sigma/dQ^2$ [cm$^2$/(GeV/$c$)$^2$]")
    axes[0].legend(loc="best", fontsize=8, title=r"$\sin^2\theta_W$")
    fig.suptitle(
        "sin^2(theta_W) sensitivity of proton/neutron NC cross sections "
        rf"($g_s^A={gsA}$ baseline)",
        fontsize=11,
    )
    fig.tight_layout()

    out = PLOTS_DIR / "sin2thetaW_sensitivity_proton_neutron.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def print_sample_table(const: PhysicsConstants = CONST, gsA: float = 0.0) -> None:
    """Print dsigma/dQ^2 [cm^2/(GeV/c)^2] at sample Q^2 for sanity-checking."""
    q2 = np.array([0.10, 0.45, 1.05])
    channels = [
        ("nu p", "nu", "proton"),
        ("nu n", "nu", "neutron"),
        ("nubar p", "nubar", "proton"),
        ("nubar n", "nubar", "neutron"),
    ]
    print(f"\nSample dsigma/dQ^2 [cm^2/(GeV/c)^2], gsA={gsA} (baseline):")
    print(f"  {'channel':10s}" + "".join(f"  Q2={x:<10.2f}" for x in q2))
    for name, beam, target in channels:
        vals = dsigma_dq2_nc_target_cm2(q2, const, beam=beam, target=target, gsA=gsA)
        print(f"  {name:10s}" + "".join(f"  {v:.4e}" for v in vals))


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    print("Physics inputs:")
    for name in PhysicsConstants.__dataclass_fields__:
        print(f"  {name:16s} = {getattr(CONST, name)}")

    out_sm = plot_main_cross_section()
    out_flux = plot_flux_averaged()
    out_sin2 = plot_sin2_theta_w_sweep()
    # Proton vs neutron NC (KamLAND form factors, Ahrens engine), baseline gsA=0
    out_nu_pn = plot_proton_vs_neutron("nu", gsA=0.0)
    out_nubar_pn = plot_proton_vs_neutron("nubar", gsA=0.0)
    out_ratio = plot_neutron_to_proton_ratio(gsA=0.0)
    out_sin2_pn = plot_sin2_proton_neutron(gsA=0.0)

    print(f"\nSaved: {out_sm}")
    print(f"Saved: {out_flux}")
    print(f"Saved: {out_sin2}")
    print(f"Saved: {out_nu_pn}")
    print(f"Saved: {out_nubar_pn}")
    print(f"Saved: {out_ratio}")
    print(f"Saved: {out_sin2_pn}")

    print_sample_table(gsA=0.0)

    print(
        "\nNote: the fixed-energy plot is the Standard Model Ahrens Eqs. "
        "(1.7)-(1.15) calculation. The flux-averaged plot uses an APPROXIMATE "
        "Gaussian flux (NOT the real Ahrens beam), only to demonstrate the "
        "flux-averaging method. Proton/neutron plots use KamLAND Sec. II form "
        "factors in the Ahrens engine, baseline g_s^A=0 (no strange axial)."
    )


if __name__ == "__main__":
    main()
