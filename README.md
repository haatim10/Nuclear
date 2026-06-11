# NuclearProject — Ahrens νp elastic & KamLAND NC form factors

Python implementation of neutral-current elastic neutrino–nucleon scattering:

- **Ahrens et al., Phys. Rev. D 35, 785 (1987)** — proton-only fixed-energy \(d\sigma/dQ^2\) (Eqs. 1.7–1.15)
- **KamLAND Collaboration, Phys. Rev. D 107, 072006 (2023), Sec. II** — proton vs neutron NC form factors (Eqs. 2–6), fed into the same Ahrens cross-section engine

Reactions computed:

- \(\nu_\mu p \to \nu_\mu p\), \(\bar\nu_\mu p \to \bar\nu_\mu p\) (Ahrens proton SM)
- \(\nu_\mu n \to \nu_\mu n\), \(\bar\nu_\mu n \to \bar\nu_\mu n\) (KamLAND neutron extension)

## Setup

```bash
cd NuclearProject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python plot_cross_section.py
```

All plots are written to `plots/`. The script also prints a sanity-check table of \(d\sigma/dQ^2\) at \(Q^2 = 0.10,\ 0.45,\ 1.05\) GeV² for \(\nu p\), \(\nu n\), \(\bar\nu p\), \(\bar\nu n\).

## Default constants

Defined in `PhysicsConstants` (`plot_cross_section.py`):

| Symbol | Value | Unit | Source / note |
|--------|-------|------|---------------|
| \(G_F\) | 1.166×10⁻⁵ | GeV⁻² | Fermi constant |
| \(M_p\) | 0.938 | GeV | Proton (nucleon) mass |
| \(\sin^2\theta_W\) | 0.220 | — | Ahrens best fit |
| \(M_V\) | 0.84 | GeV | Vector dipole mass |
| \(M_A\) | 1.06 | GeV | Axial dipole mass |
| \(g_A\) | 1.26 | — | Axial coupling at \(Q^2=0\) |
| \(\kappa_p\) | 1.793 | — | Proton anomalous magnetic moment |
| \(\kappa_n\) | −1.913 | — | Neutron anomalous magnetic moment |
| \(E_\nu\) | 1.25 | GeV | Fixed beam energy (monochromatic) |
| \(g_s^A\) | 0.0 | — | Strange axial coupling (explicit parameter; baseline only) |

**Units:** cross sections are computed in natural units (GeV⁻⁴ per GeV² for \(d\sigma/dQ^2\)) and converted to cm²/(GeV/c)² via \((\hbar c)^2 \times 10^{-26}\), with \(\hbar c = 0.1973269804\) GeV·fm.

Change inputs by editing `CONST` or passing a custom `PhysicsConstants` instance.

---

## Physics Formulas and Assumptions Used

This section documents every formula implemented in `plot_cross_section.py` as of the current version. The **cross-section structure** always comes from Ahrens; KamLAND supplies **proton/neutron NC form factors** only for the extension plots.

### 1. Ahrens fixed-energy cross-section formula (Eq. 1.14)

Differential cross section for NC elastic scattering on a nucleon:

\[
\frac{d\sigma}{dQ^2}
= \frac{G_F^2 M^2}{8\pi E_\nu^2}
\left[
A \pm B\,\frac{s-u}{M^2} + C\,\frac{(s-u)^2}{M^4}
\right]
\]

**Sign convention (Ahrens):**

- **Plus** (\(+\)) for **neutrinos** (\(\nu\))
- **Minus** (\(-\)) for **antineutrinos** (\(\bar\nu\))

**Kinematics (Eq. 1.15):**

\[
s - u = 4 M E_\nu - Q^2
\]

In code, \(M = M_p\) (proton mass). Implemented in `dsigma_dq2_natural()` and `dsigma_dq2_nc_target()`.

---

### 2. Ahrens structure functions \(A\), \(B\), \(C\) (Eq. 1.15)

These depend on the vector form factors \(F_1(Q^2)\), \(F_2(Q^2)\) and the axial form factor \(G_A(Q^2)\). Define \(\tau = Q^2/(4M^2)\).

\[
A = \frac{Q^2}{M^2}\left[
G_A^2\left(1 + \frac{Q^2}{4M^2}\right)
- F_1^2\left(1 - \frac{Q^2}{4M^2}\right)
+ F_2^2\left(1 - \frac{Q^2}{4M^2}\right)\frac{Q^2}{4M^2}
+ F_1 F_2\,\frac{Q^2}{M^2}
\right]
\]

\[
B = \frac{Q^2}{M^2}\,G_A\,(F_1 + F_2)
\]

\[
C = \frac{1}{4}\left[
G_A^2 + F_1^2 + F_2^2\,\frac{Q^2}{4M^2}
\right]
\]

Implemented in `coefficient_a()`, `coefficient_b()`, `coefficient_c()`. The **\(B\)** term is the \(\nu/\bar\nu\) interference piece; its sign flip in Eq. 1.14 is what makes the two beams differ.

---

### 3. Ahrens proton Standard Model form factors (Eqs. 1.7, 1.9, 1.10, 1.11, 1.12)

Used for the **proton-only** Ahrens plots (`vector_form_factors()`, `axial_form_factor()`).

**Weak couplings (Eq. 1.7, Standard Model):**

\[
\alpha = 1 - 2\sin^2\theta_W, \qquad
\beta = 1, \qquad
\gamma = -\tfrac{2}{3}\sin^2\theta_W, \qquad
\delta = 0
\]

**Vector form factors (Eq. 1.9), with dipole inputs (Eq. 1.10):**

\[
F_1 + F_2 = \alpha\,G_V^3 + \gamma\,G_V^0
\]

\[
F_2 = \alpha\,F_V^3 + \gamma\,F_V^0
\]

\[
F_1 = (F_1 + F_2) - F_2
\]

**Dipole isovector / isoscalar form factors (Eq. 1.10):**

\[
\tau = \frac{Q^2}{4M^2}
\]

\[
G_V^3 = \frac{1}{2}\,\frac{1 + \kappa_p - \kappa_n}{\left(1 + Q^2/M_V^2\right)^2}
\]

\[
G_V^0 = \frac{3}{2}\,\frac{1 + \kappa_p + \kappa_n}{\left(1 + Q^2/M_V^2\right)^2}
\]

\[
F_V^3 = \frac{1}{2}\,\frac{\kappa_p - \kappa_n}{(1+\tau)\left(1 + Q^2/M_V^2\right)^2}
\]

\[
F_V^0 = \frac{3}{2}\,\frac{\kappa_p + \kappa_n}{(1+\tau)\left(1 + Q^2/M_V^2\right)^2}
\]

**Axial form factor (Eqs. 1.11, 1.12, Standard Model):**

\[
G_A = \beta\,G_A^3 + \delta\,G_A^0
\]

\[
G_A^3 = \frac{1}{2}\,\frac{g_A}{\left(1 + Q^2/M_A^2\right)^2}, \qquad G_A^0 = 0
\]

So in the SM baseline: \(G_A = \tfrac{1}{2}\,g_A / (1 + Q^2/M_A^2)^2\).

---

### 4. KamLAND proton/neutron neutral-current form factors (Sec. II, Eqs. 2–4)

Used **only** for the proton-vs-neutron extension (`nc_form_factors_kamland()`). The cross-section formula remains Ahrens Eq. 1.14/1.15; only \(F_1\), \(F_2\), \(G_A\) change with target.

**Axial NC form factors (Eq. 2):**

\[
G_A^{\mathrm{NC},p}(Q^2) = \frac{1}{2}\,(+g_A - g_s^A)\left(1 + \frac{Q^2}{M_A^2}\right)^{-2}
\]

\[
G_A^{\mathrm{NC},n}(Q^2) = \frac{1}{2}\,(-g_A - g_s^A)\left(1 + \frac{Q^2}{M_A^2}\right)^{-2}
\]

**\(g_s^A\) is an explicit function parameter** (default `gsA=0.0`). It is **not** silently assumed to be zero. Baseline plots use \(g_s^A = 0\) only as a **“no strange axial contribution” reference case**, to isolate proton/neutron and \(\sin^2\theta_W\) effects before turning on strange physics.

**Charged-current vector form factors (Eq. 3):**

\[
F_1^{\mathrm{CC}} = F_{1p} - F_{1n}, \qquad
F_2^{\mathrm{CC}} = F_{2p} - F_{2n}
\]

**Neutral-current vector form factors (Eq. 4), strange vector kept explicit:**

For **proton**:

\[
F_{1,2}^{\mathrm{NC},p} = +\tfrac{1}{2}F_{1,2}^{\mathrm{CC}} - 2\sin^2\theta_W\,F_{1,2}^{p} - \tfrac{1}{2}F_{1,2}^{s}
\]

For **neutron**:

\[
F_{1,2}^{\mathrm{NC},n} = -\tfrac{1}{2}F_{1,2}^{\mathrm{CC}} - 2\sin^2\theta_W\,F_{1,2}^{n} - \tfrac{1}{2}F_{1,2}^{s}
\]

\(F_1^s\) and \(F_2^s\) are **explicit parameters** (`F1s`, `F2s`), defaulted to **zero** following KamLAND’s statement that strange vector form factors are consistent with zero. They are not hard-coded as a physical assumption.

**Cross-check:** at \(g_s^A = 0\), the KamLAND proton \(F_1, F_2, G_A\) reproduce the Ahrens proton SM values numerically (verified in code output).

---

### 5. Electromagnetic form-factor assumption (first version)

Used inside `nc_form_factors_kamland()` to build \(F_{1p,n}\), \(F_{2p,n}\) before the KamLAND NC combinations.

**Dipole approximation** (older/common; not BBBA05):

\[
G_D = \frac{1}{\left(1 + Q^2/M_V^2\right)^2}
\]

\[
G_E^p = G_D, \qquad G_M^p = (1 + \kappa_p)\,G_D
\]

\[
G_E^n = 0, \qquad G_M^n = \kappa_n\,G_D
\]

**Sachs → Dirac/Pauli (KamLAND Eqs. 5–6):**

\[
F_1 = \frac{G_E + \tau\,G_M}{1 + \tau}, \qquad
F_2 = \frac{G_M - G_E}{1 + \tau}
\]

with \(\tau = Q^2/(4M^2)\).

**Assumption stated clearly:**

- This is the **dipole EM form-factor approximation**.
- KamLAND uses the more modern **BBBA05** parametrization for electron-scattering data.
- **TODO:** replace dipole EM form factors with BBBA05 if required.

---

### 6. Flux-averaging approximation

For the demonstration flux-averaged plot only (`flux_average_dsigma()`):

\[
\left\langle \frac{d\sigma}{dQ^2} \right\rangle(Q^2)
= \frac{\displaystyle\int \Phi(E_\nu)\,\dfrac{d\sigma}{dQ^2}(Q^2, E_\nu)\,dE_\nu}
       {\displaystyle\int \Phi(E_\nu)\,dE_\nu}
\]

**Approximate flux used:**

\[
\Phi(E_\nu) = \exp\!\left[-\tfrac{1}{2}\left(\frac{E_\nu - 1.25\ \mathrm{GeV}}{0.6\ \mathrm{GeV}}\right)^2\right]
\]

integrated over \(E_\nu \in [0.2,\ 5.0]\) GeV (400-point grid, trapezoidal rule).

**Physical cutoff:** energies with \(E_\nu \le Q^2/(4M_p)\) are set to zero (elastic kinematics require \(s-u > 0\)).

**Important limitations of this flux:**

- This is a **fake Gaussian flux**, **NOT** the real BNL E734 beam spectrum.
- It is only a **demonstration of the averaging method**.
- **TODO:** replace the Gaussian with digitized Ahrens Fig. 2 / Fig. 3 flux data for a real Ahrens Fig. 35 comparison.

---

### 7. \(\sin^2\theta_W\) sensitivity

**Default:** \(\sin^2\theta_W = 0.22\) (Ahrens best fit).

**Sweep range:** 0.10 to 0.30 (9 points in `ahrens_sin2_sweep.png`; six values \([0.10, 0.15, 0.20, 0.22, 0.25, 0.30]\) in `sin2thetaW_sensitivity_proton_neutron.png`).

**Where \(\sin^2\theta_W\) enters:**

- **Ahrens proton:** through \(\alpha = 1 - 2\sin^2\theta_W\) and \(\gamma = -\tfrac{2}{3}\sin^2\theta_W\) in the vector form factors.
- **KamLAND proton/neutron:** through the \(-2\sin^2\theta_W\,F_{1,2}^{p/n}\) term in \(F_{1,2}^{\mathrm{NC}}\).

**Where it does not enter (baseline):**

- The axial \(G_A\) formulas do **not** depend on \(\sin^2\theta_W\) directly in the current SM setup (\(\delta = 0\), fixed \(g_A\), baseline \(g_s^A = 0\)).

**Note:** \(\sin^2\theta_W\) is **varied as an input parameter**; it is **not extracted from data** in this project.

---

### 8. Proton vs neutron comparison

The neutron/proton ratio plot shows:

\[
R(Q^2) = \frac{(d\sigma/dQ^2)_n}{(d\sigma/dQ^2)_p}
\]

for \(\nu\) and \(\bar\nu\) separately (`plot_neutron_to_proton_ratio()`).

This directly addresses the **difference in NC cross section between neutrons and protons**.

**Observed behavior (baseline \(g_s^A = 0\), \(\sin^2\theta_W = 0.22\), \(E_\nu = 1.25\) GeV):**

- **Neutrinos:** \(R \approx 1.3\text{–}1.5\) over the plotted \(Q^2\) range — neutron cross section is **larger** than proton.
- **Antineutrinos:** \(R\) starts above 1 at low \(Q^2\) but **decreases with \(Q^2\)** and can **drop below 1** at high \(Q^2\) (proton exceeds neutron), due to the different \(\nu/\bar\nu\) interference sign on the \(B\) term.

Sample values at \(Q^2 = 0.45\) GeV² (cm²/(GeV/c)², \(g_s^A = 0\)):

| Channel | \(d\sigma/dQ^2\) |
|---------|------------------|
| \(\nu p\) | 1.33×10⁻³⁹ |
| \(\nu n\) | 1.77×10⁻³⁹ |
| \(\bar\nu p\) | 5.41×10⁻⁴⁰ |
| \(\bar\nu n\) | 7.16×10⁻⁴⁰ |

---

### 9. Plots generated

| File | What it shows | Form factors | Notes |
|------|---------------|--------------|-------|
| `ahrens_sm_dsigma.png` | \(d\sigma/dQ^2\) vs \(Q^2\) for \(\nu p\) and \(\bar\nu p\) | Ahrens Eqs. 1.7–1.12 | Fixed \(E_\nu = 1.25\) GeV, SM |
| `ahrens_flux_averaged_approx.png` | Flux-averaged \(\langle d\sigma/dQ^2\rangle\) vs \(Q^2\) | Ahrens (proton) | **Approximate Gaussian flux only** |
| `ahrens_sin2_sweep.png` | \(\sin^2\theta_W\) sensitivity (0.1–0.3) for \(\nu p\) and \(\bar\nu p\) | Ahrens (proton) | Fixed \(E_\nu\) |
| `nu_proton_vs_neutron.png` | \(\nu p\) vs \(\nu n\) cross sections | KamLAND + Ahrens engine | Baseline \(g_s^A = 0\) |
| `nubar_proton_vs_neutron.png` | \(\bar\nu p\) vs \(\bar\nu n\) cross sections | KamLAND + Ahrens engine | Baseline \(g_s^A = 0\) |
| `neutron_to_proton_ratio.png` | \((d\sigma/dQ^2)_n / (d\sigma/dQ^2)_p\) for \(\nu\) and \(\bar\nu\) | KamLAND + Ahrens engine | Baseline \(g_s^A = 0\) |
| `sin2thetaW_sensitivity_proton_neutron.png` | \(\sin^2\theta_W\) sweep for \(\nu p\) and \(\nu n\) | KamLAND + Ahrens engine | Two panels; baseline \(g_s^A = 0\) |
| `pn_ratio_nu.png` | \(\sigma_p/\sigma_n\) vs \(Q^2\) for \(\nu\), \(\sin^2\theta_W\) sweep | KamLAND + Ahrens engine | Baseline \(g_s^A = 0\) |
| `pn_ratio_nubar.png` | \(\sigma_p/\sigma_n\) vs \(Q^2\) for \(\bar\nu\), \(\sin^2\theta_W\) sweep | KamLAND + Ahrens engine | Baseline \(g_s^A = 0\) |
| `pn_ratio_combined.png` | \(\sigma_p/\sigma_n\) vs \(Q^2\), \(\nu\) and \(\bar\nu\) panels | KamLAND + Ahrens engine | Combined view |

---

## \(\sigma_p / \sigma_n\) vs \(Q^2\) for \(\sin^2\theta_W\) sweep

Shows how the **proton-to-neutron NC cross-section ratio** depends on \(Q^2\) and on **\(\sin^2\theta_W\)**.

**Definitions:**

\[
R_\nu(Q^2) = \frac{d\sigma(\nu p)/dQ^2}{d\sigma(\nu n)/dQ^2}, \qquad
R_{\bar\nu}(Q^2) = \frac{d\sigma(\bar\nu p)/dQ^2}{d\sigma(\bar\nu n)/dQ^2}
\]

**Implementation:**

- Cross section: **Ahrens Eq. (1.14)/(1.15)** (`dsigma_dq2_nc_target`)
- Form factors: **KamLAND Sec. II** (`nc_form_factors_kamland`)
- Fixed \(E_\nu = 1.25\) GeV, baseline \(g_s^A = 0\) (no strange axial contribution)
- \(\sin^2\theta_W \in \{0.10,\ 0.15,\ 0.20,\ 0.22,\ 0.25,\ 0.30\}\); **0.22** is highlighted as the Ahrens reference value

**Note:** This is **\(\sigma_p/\sigma_n\)**, the inverse of the older `neutron_to_proton_ratio.png` plot (\(\sigma_n/\sigma_p\)).

Running `python plot_cross_section.py` prints a table of \(R_\nu\) and \(R_{\bar\nu}\) at \(Q^2 = 0.10,\ 0.45,\ 1.05\) GeV² for each \(\sin^2\theta_W\).

---

### 10. Important limitations

| Limitation | Detail |
|------------|--------|
| Fixed energy | All fixed-energy plots use **\(E_\nu = 1.25\) GeV** (monochromatic), not the BNL wide-band spectrum. |
| Not Fig. 35 | This is **not** a full reproduction of Ahrens **Fig. 35** (flux-averaged data comparison). |
| Approximate flux | The flux-averaged plot uses a **Gaussian demo flux**, not measured beam data. |
| No detector effects | No acceptance, efficiency, resolution smearing, or background subtraction. |
| No nuclear effects | Free-nucleon calculation only — no FSI, SI, Pauli blocking, Fermi motion, or nuclear deexcitation. |
| \(g_s^A\) not fitted | \(g_s^A\) is an **input parameter**; no fit to KamLAND or Ahrens strange-axial results is performed. |
| \(\sin^2\theta_W\) not extracted | \(\sin^2\theta_W\) is **varied as input**, not determined from a simultaneous fit to data. |
| Dipole EM form factors | Proton/neutron extension uses **dipole** EM \(F_{1,2}\), not BBBA05. |
| Nucleon mass | \(M = M_p\) is used for both proton and neutron targets (Ahrens engine convention). |
| Strange vector | \(F_1^s = F_2^s = 0\) by default (explicit parameters, KamLAND-consistent). |

---

## Code structure

| Function | Role | Source |
|----------|------|--------|
| `vector_form_factors()` | Ahrens proton \(F_1, F_2\) | Ahrens Eqs. 1.7, 1.9, 1.10 |
| `axial_form_factor()` | Ahrens proton \(G_A\) | Ahrens Eqs. 1.11, 1.12 |
| `coefficient_a/b/c()` | Structure functions \(A, B, C\) | Ahrens Eq. 1.15 |
| `dsigma_dq2_natural()` | Proton \(d\sigma/dQ^2\) | Ahrens Eq. 1.14 |
| `nc_form_factors_kamland()` | Proton/neutron \(F_1^{\mathrm{NC}}, F_2^{\mathrm{NC}}, G_A^{\mathrm{NC}}\) | KamLAND Sec. II |
| `dsigma_dq2_nc_target()` | Target \(d\sigma/dQ^2\) (p or n) | Ahrens engine + KamLAND FF |
| `flux_average_dsigma()` | Demo flux average | Approximate Gaussian flux |

## References

- L. A. Ahrens et al., *Phys. Rev. D* **35**, 785 (1987) — `ahrens.pdf`
- S. Abe et al. (KamLAND Collaboration), *Phys. Rev. D* **107**, 072006 (2023) — `kamland.pdf`

## Files

```
NuclearProject/
├── plot_cross_section.py   # main code
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── ahrens.pdf              # reference (not executed by script)
├── kamland.pdf             # reference (not executed by script)
├── .venv/                  # local venv (git-ignored)
└── plots/                  # generated figures (committed)
    ├── ahrens_sm_dsigma.png
    ├── ahrens_flux_averaged_approx.png
    ├── ahrens_sin2_sweep.png
    ├── nu_proton_vs_neutron.png
    ├── nubar_proton_vs_neutron.png
    ├── neutron_to_proton_ratio.png
    ├── sin2thetaW_sensitivity_proton_neutron.png
    ├── pn_ratio_nu.png
    ├── pn_ratio_nubar.png
    └── pn_ratio_combined.png
```

**Note:** `pypdf` is listed in `requirements.txt` for reading the reference PDFs locally; the plotting script does not import it.
