# Schrödingerization, QSP/QSVT and Carleman linearization: transferable capability

This repository presents a curated and reproducible selection of **Quantum Mads' prior work on quantum methods for dissipative and nonlinear dynamics and matrix functions**. It brings together three complementary capabilities:

1. **Schrödingerization:** embedding dissipative evolution in an enlarged Hermitian system and recovering physical states and operators.
2. **Quantum signal processing and singular value transformation:** constructing an inverse polynomial, synthesizing its QSP phases and validating the resulting matrix action.
3. **Carleman--Schrödingerization:** lifting quadratic differential equations to finite linear systems and mapping those lifts to enlarged Hermitian evolutions.

> **Detailed derivations and figure-by-figure explanations are kept in [`docs/`](docs/).**

## Results at a glance

| Evidence package | Configuration | Headline result |
| --- | --- | --- |
| Heat-equation state recovery | $32$ spatial and $512$ auxiliary points | Error vs direct matrix exponential decreases from `0.638%` with $e^{-\|p\|}$ to `0.0165%` with the smooth auxiliary extension |
| Heat-equation propagator recovery | Same fixed recovery coordinate | Relative Frobenius error decreases from `0.467%` to `0.0115%` |
| Heat-equation postselection | Accepted region $p\geq0$ | Probability changes from `28.9%` to `22.9%`, exposing the accuracy--postselection trade-off |
| QSP phase synthesis | $\kappa=6$, odd degree $25$, $26$ phases | Synthesized response reproduces the requested polynomial to `3.05e-13` |
| QSVT matrix-inverse action | Controlled $4\times4$ matrix | Solution error `1.43%`, residual `0.803%` and idealized postselection probability `2.18%` |
| Scalar Carleman check | $K=6$ | Absolute final-time truncation error `2.5920e-9` |
| Coupled-vector Carleman check | $K=1\rightarrow2\rightarrow3$ | Physical-state error decreases from `14.82%` to `1.559%` to `2.03e-16` |
| Coupled-vector Schrödingerization | $K=3$ lift with $39\times1{,}024=39{,}936$ enlarged dimensions | Recovered physical-state error vs direct Carleman state `9.2862e-7`; complete-lift recovery error `1.0503e-6` |

For the heat equation, smoothing improves state recovery by approximately **38.6x** and propagator reconstruction by approximately **40.6x**. Independent state-level and operator-level implementations agree to `7.69e-15`.

## Evidence packages

| Capability | What the experiment demonstrates | Reproduce | Technical note | Metrics |
| --- | --- | --- | --- | --- |
| Schrödingerization | Warped-phase embedding of a dissipative heat equation, auxiliary-profile design, state and complete-propagator recovery, and postselection diagnostics | [`run_heat_equation_demo.py`](scripts/run_heat_equation_demo.py) | [`Schrodingerization_heat-equation.md`](docs/Schrodingerization_heat-equation.md) | [`heat_equation_metrics.json`](results/heat_equation_metrics.json) |
| QSP/QSVT matrix inversion | Odd Chebyshev inverse approximation, bounded scaling, QSP phase synthesis, independent response validation and matrix inverse action | [`run_qsp_qsvt_demo.py`](scripts/run_qsp_qsvt_demo.py) | [`QSP_QSVT_matrix-inversion.md`](docs/QSP_QSVT_matrix-inversion.md) | [`qsp_qsvt_metrics.json`](results/qsp_qsvt_metrics.json) |
| Carleman--Schrödingerization | Truncated linearization of scalar and coupled-vector quadratic equations followed by Hermitian evolution and analytical validation | [`run_carleman_schrodingerization_demo.py`](scripts/run_carleman_schrodingerization_demo.py) | [`Carleman_Schrodingerization.md`](docs/Carleman_Schrodingerization.md) | [`carleman_schrodingerization_metrics.json`](results/carleman_schrodingerization_metrics.json) |

The committed figures are stored in [`results/`](results/) alongside the machine-readable metrics.

## Transferable relevance to Airbus

Together, the three evidence packages establish a reusable workflow:

$$
\boxed{
\text{dissipative or quadratic dynamics}
\longrightarrow
\text{linear, unitary-compatible representation}
\longrightarrow
\text{quantum signal or Hamiltonian transformation}
\longrightarrow
\text{validated observables}
}.
$$

In particular, the repository demonstrates the ability to:

- Map non-unitary dynamics to enlarged Hermitian evolution
- Linearize quadratic nonlinearities and quantify truncation error
- Design and discretize auxiliary-variable representations
- Construct bounded matrix-function approximations and synthesize QSP phases
- Recover states and complete operators
- Separate approximation, conditioning, recovery and postselection effects using analytical and classical matrix references

These are transferable mathematical and numerical building blocks for selecting, implementing and validating an Airbus-specific proof of concept.

## Reproducing the results

The repository uses Python 3.12 and a locked [`uv`](https://docs.astral.sh/uv/) environment:

```powershell
uv sync --locked
uv run python scripts/run_heat_equation_demo.py
uv run python scripts/run_qsp_qsvt_demo.py
uv run python scripts/run_carleman_schrodingerization_demo.py
```



## Scientific context

- S. Jin, N. Liu and Y. Yu, [Quantum simulation of partial differential equations: Applications and detailed analysis](https://doi.org/10.1103/PhysRevA.108.032603), *Physical Review A* 108, 032603 (2023).
- S. Jin, N. Liu and Y. Yu, [Quantum simulation of partial differential equations via Schrödingerization](https://doi.org/10.1103/PhysRevLett.133.230602), *Physical Review Letters* 133, 230602 (2024).
- G. H. Low and I. L. Chuang, [Optimal Hamiltonian Simulation by Quantum Signal Processing](https://doi.org/10.1103/PhysRevLett.118.010501), *Physical Review Letters* 118, 010501 (2017).
- A. Gilyén, Y. Su, G. H. Low and N. Wiebe, [Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics](https://doi.org/10.1145/3313276.3316366), *Proceedings of STOC 2019*, 193--204.
- C. Sünderhauf *et al.*, [Matrix inversion polynomials for the quantum singular value transformation](https://arxiv.org/abs/2507.15537) (2025).
- J.-P. Liu *et al.*, [Efficient quantum algorithm for dissipative nonlinear differential equations](https://doi.org/10.1073/pnas.2026805118), *PNAS* 118, e2026805118 (2021); see the [2026 correction to the Supporting Information](https://doi.org/10.1073/pnas.2615307123).
- S. Sasaki, K. Endo and M. Muramatsu, [Hamiltonian simulation for nonlinear partial differential equation by Schrödingerization](https://doi.org/10.1038/s41598-026-44920-8), *Scientific Reports* 16, 11743 (2026).
