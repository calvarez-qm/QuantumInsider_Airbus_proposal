"""Classical numerical demonstration of heat-equation Schrödingerization.

The script constructs the enlarged Hermitian system, evolves two auxiliary
warped-phase profiles, recovers the physical solution and writes validation and
auxiliary-coordinate diagnostics.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def periodic_second_derivative(points: int, length: float) -> sparse.csr_matrix:
    """Return the second-order periodic finite-difference Laplacian."""
    spacing = length / points
    diagonals = [
        np.ones(points - 1),
        -2.0 * np.ones(points),
        np.ones(points - 1),
    ]
    matrix = sparse.diags(diagonals, offsets=[-1, 0, 1], format="lil")
    matrix[0, -1] = 1.0
    matrix[-1, 0] = 1.0
    return (matrix / spacing**2).tocsr()


def warped_phase_profile(p: np.ndarray, order: int = 1) -> np.ndarray:
    """Return a warped-phase profile used in the Schrödingerization scheme.

    ``order=1`` is the standard symmetric profile exp(-|p|). ``order=2``
    retains exp(-p) for p >= 0 and replaces the interval -1 < p < 0 by a
    cubic extension. The cubic matches the value and first derivative at both
    endpoints, removing the cusp at p=0 and improving the decay of its Fourier
    coefficients.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    profile = np.exp(-np.abs(p))
    if order == 2:
        bridge = (p > -1.0) & (p < 0.0)
        profile[bridge] = (
            (3.0 / np.e - 3.0) * p[bridge] ** 3
            + (4.0 / np.e - 5.0) * p[bridge] ** 2
            - p[bridge]
            + 1.0
        )
    return profile


def evolve_enlarged_state(
    physical_state: np.ndarray,
    auxiliary_profile: np.ndarray,
    hamiltonian: sparse.csr_matrix,
    spatial_points: int,
    auxiliary_points: int,
    final_time: float,
    diffusivity: float,
) -> np.ndarray:
    """Evolve one warped profile and return w(t, x, p)."""
    auxiliary_fourier_state = np.fft.ifft(auxiliary_profile)
    enlarged_initial_state = np.kron(physical_state, auxiliary_fourier_state)
    enlarged_final_state = expm_multiply(
        1j * hamiltonian * final_time * diffusivity,
        enlarged_initial_state,
    )
    return np.fft.fft(
        enlarged_final_state.reshape(spatial_points, auxiliary_points), axis=1
    )


def reconstruct_operator_slices(
    laplacian: sparse.csr_matrix,
    frequencies: np.ndarray,
    auxiliary_profile: np.ndarray,
    final_time: float,
    diffusivity: float,
) -> np.ndarray:
    """Return the warped spatial propagator associated with every p slice.

    The tensor-product Hamiltonian is separable in the auxiliary-frequency
    basis. Diagonalizing the small spatial Laplacian therefore reconstructs
    all operator blocks without forming the full enlarged unitary.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    phases = np.exp(
        1j * diffusivity * final_time * np.outer(frequencies, eigenvalues)
    )
    frequency_propagators = np.einsum(
        "ia,ka,ja->kij",
        eigenvectors,
        phases,
        eigenvectors.conj(),
        optimize=True,
    )
    auxiliary_fourier_state = np.fft.ifft(auxiliary_profile)
    return np.fft.fft(
        auxiliary_fourier_state[:, np.newaxis, np.newaxis]
        * frequency_propagators,
        axis=0,
    )


def relative_operator_errors(
    warped_operator_slices: np.ndarray,
    auxiliary_profile: np.ndarray,
    reference_operator: np.ndarray,
) -> np.ndarray:
    """Compute relative Frobenius reconstruction error for every p slice."""
    errors = np.full(auxiliary_profile.shape, np.nan, dtype=float)
    valid = np.abs(auxiliary_profile) > 1e-12 * np.max(np.abs(auxiliary_profile))
    reconstructed = (
        warped_operator_slices[valid]
        / auxiliary_profile[valid, np.newaxis, np.newaxis]
    )
    errors[valid] = np.linalg.norm(
        reconstructed - reference_operator[np.newaxis, :, :], axis=(1, 2)
    ) / np.linalg.norm(reference_operator, ord="fro")
    return errors


def auxiliary_measurement_probabilities(
    warped_operator_slices: np.ndarray,
    auxiliary_profile: np.ndarray,
    physical_state: np.ndarray,
) -> np.ndarray:
    """Return Born-rule probabilities for a normalized enlarged input state."""
    normalized_physical_state = physical_state / np.linalg.norm(physical_state)
    output_blocks = np.einsum(
        "kij,j->ki", warped_operator_slices, normalized_physical_state
    ) / np.linalg.norm(auxiliary_profile)
    return np.sum(np.abs(output_blocks) ** 2, axis=1).real


def run_demo(output_dir: Path) -> dict[str, float | int | str]:
    """Run the heat-equation Schrödingerization and save its artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    spatial_qubits = 5
    auxiliary_qubits = 9
    spatial_points = 2**spatial_qubits
    auxiliary_points = 2**auxiliary_qubits
    domain_length = 15.0
    auxiliary_scale = 10.0
    final_time = 5.0
    diffusivity = domain_length / (2.0 * np.pi) ** 2

    laplacian = periodic_second_derivative(spatial_points, domain_length)
    frequencies = np.arange(auxiliary_points, dtype=float)
    frequencies[auxiliary_points // 2 :] -= auxiliary_points
    frequencies /= auxiliary_scale
    frequency_operator = sparse.diags(frequencies, format="csr")
    hamiltonian = sparse.kron(laplacian, frequency_operator, format="csr")

    auxiliary_spacing = 2.0 * auxiliary_scale * np.pi / auxiliary_points
    auxiliary_grid = (
        np.arange(auxiliary_points) - auxiliary_points // 2
    ) * auxiliary_spacing
    warped_state = warped_phase_profile(auxiliary_grid, order=1)
    smooth_warped_state = warped_phase_profile(auxiliary_grid, order=2)

    x = np.linspace(0.0, domain_length, spatial_points, endpoint=False)
    initial_state = np.sin(2.0 * np.pi * x / domain_length)
    physical_auxiliary_grid = evolve_enlarged_state(
        initial_state,
        warped_state,
        hamiltonian,
        spatial_points,
        auxiliary_points,
        final_time,
        diffusivity,
    )
    smooth_physical_auxiliary_grid = evolve_enlarged_state(
        initial_state,
        smooth_warped_state,
        hamiltonian,
        spatial_points,
        auxiliary_points,
        final_time,
        diffusivity,
    )
    recovery_index = auxiliary_points // 2 + 1
    recovered = (
        physical_auxiliary_grid[:, recovery_index]
        / warped_state[recovery_index]
    ).real
    smooth_recovered = (
        smooth_physical_auxiliary_grid[:, recovery_index]
        / smooth_warped_state[recovery_index]
    ).real

    analytic = (
        np.exp(-final_time / domain_length)
        * np.sin(2.0 * np.pi * x / domain_length)
    )
    matrix_reference = expm_multiply(
        diffusivity * laplacian * final_time,
        initial_state,
    )

    error_vs_analytic = float(
        np.linalg.norm(recovered - analytic) / np.linalg.norm(analytic)
    )
    error_vs_matrix = float(
        np.linalg.norm(recovered - matrix_reference)
        / np.linalg.norm(matrix_reference)
    )
    spatial_error = float(
        np.linalg.norm(matrix_reference - analytic) / np.linalg.norm(analytic)
    )
    smooth_error_vs_analytic = float(
        np.linalg.norm(smooth_recovered - analytic) / np.linalg.norm(analytic)
    )
    smooth_error_vs_matrix = float(
        np.linalg.norm(smooth_recovered - matrix_reference)
        / np.linalg.norm(matrix_reference)
    )
    recovery_coordinate = float(auxiliary_grid[recovery_index])

    reference_operator = scipy.linalg.expm(
        diffusivity * laplacian.toarray() * final_time
    )
    standard_operator_slices = reconstruct_operator_slices(
        laplacian, frequencies, warped_state, final_time, diffusivity
    )
    smooth_operator_slices = reconstruct_operator_slices(
        laplacian, frequencies, smooth_warped_state, final_time, diffusivity
    )
    standard_operator_errors = relative_operator_errors(
        standard_operator_slices, warped_state, reference_operator
    )
    smooth_operator_errors = relative_operator_errors(
        smooth_operator_slices, smooth_warped_state, reference_operator
    )
    standard_probabilities = auxiliary_measurement_probabilities(
        standard_operator_slices, warped_state, initial_state
    )
    smooth_probabilities = auxiliary_measurement_probabilities(
        smooth_operator_slices, smooth_warped_state, initial_state
    )

    largest_generator_eigenvalue = float(
        np.max(np.linalg.eigvalsh(diffusivity * laplacian.toarray()))
    )
    if abs(largest_generator_eigenvalue) < 1e-12:
        largest_generator_eigenvalue = 0.0
    recovery_threshold = max(0.0, largest_generator_eigenvalue) * final_time
    recovery_set = auxiliary_grid >= recovery_threshold
    analysis_set = (
        (auxiliary_grid > recovery_threshold)
        & (auxiliary_grid <= 3.0)
        & np.isfinite(standard_operator_errors)
        & np.isfinite(smooth_operator_errors)
    )
    analysis_indices = np.flatnonzero(analysis_set)
    standard_best_index = int(
        analysis_indices[np.argmin(standard_operator_errors[analysis_indices])]
    )
    smooth_best_index = int(
        analysis_indices[np.argmin(smooth_operator_errors[analysis_indices])]
    )

    operator_state_consistency = float(
        np.linalg.norm(
            standard_operator_slices[recovery_index] @ initial_state
            / warped_state[recovery_index]
            - physical_auxiliary_grid[:, recovery_index]
            / warped_state[recovery_index]
        )
        / np.linalg.norm(matrix_reference)
    )
    if operator_state_consistency > 1e-10:
        raise RuntimeError(
            "Operator-slice and state-level reconstructions are inconsistent: "
            f"{operator_state_consistency:.3e}"
        )

    metrics: dict[str, float | int | str] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "spatial_points": spatial_points,
        "auxiliary_points": auxiliary_points,
        "enlarged_state_dimension": spatial_points * auxiliary_points,
        "auxiliary_domain_minimum": float(auxiliary_grid[0]),
        "auxiliary_domain_maximum": float(auxiliary_grid[-1]),
        "auxiliary_spacing": auxiliary_spacing,
        "recovery_auxiliary_coordinate": recovery_coordinate,
        "final_time": final_time,
        "diffusivity": diffusivity,
        "relative_l2_recovery_vs_analytic": error_vs_analytic,
        "relative_l2_recovery_vs_matrix_reference": error_vs_matrix,
        "relative_l2_spatial_discretization": spatial_error,
        "smooth_profile_relative_l2_recovery_vs_analytic": smooth_error_vs_analytic,
        "smooth_profile_relative_l2_recovery_vs_matrix_reference": smooth_error_vs_matrix,
        "operator_reconstruction_relative_frobenius_error": float(
            standard_operator_errors[recovery_index]
        ),
        "smooth_profile_operator_reconstruction_relative_frobenius_error": float(
            smooth_operator_errors[recovery_index]
        ),
        "standard_profile_best_index_in_displayed_window": standard_best_index,
        "standard_profile_best_coordinate_in_displayed_window": float(
            auxiliary_grid[standard_best_index]
        ),
        "standard_profile_best_operator_error_in_displayed_window": float(
            standard_operator_errors[standard_best_index]
        ),
        "smooth_profile_best_index_in_displayed_window": smooth_best_index,
        "smooth_profile_best_coordinate_in_displayed_window": float(
            auxiliary_grid[smooth_best_index]
        ),
        "smooth_profile_best_operator_error_in_displayed_window": float(
            smooth_operator_errors[smooth_best_index]
        ),
        "recovery_threshold": recovery_threshold,
        "standard_profile_positive_region_success_probability": float(
            np.sum(standard_probabilities[recovery_set])
        ),
        "smooth_profile_positive_region_success_probability": float(
            np.sum(smooth_probabilities[recovery_set])
        ),
        "standard_profile_probability_sum": float(
            np.sum(standard_probabilities)
        ),
        "smooth_profile_probability_sum": float(np.sum(smooth_probabilities)),
        "operator_state_consistency_error": operator_state_consistency,
        "standard_profile_selected_outcome_probability": float(
            standard_probabilities[recovery_index]
        ),
        "smooth_profile_selected_outcome_probability": float(
            smooth_probabilities[recovery_index]
        ),
        "standard_profile_displayed_window_success_probability": float(
            np.sum(standard_probabilities[analysis_set])
        ),
        "smooth_profile_displayed_window_success_probability": float(
            np.sum(smooth_probabilities[analysis_set])
        ),
    }
    (output_dir / "heat_equation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _plot(
        x,
        recovered,
        smooth_recovered,
        analytic,
        matrix_reference,
        output_dir / "heat_equation.png",
    )
    _plot_auxiliary_diagnostics(
        x=x,
        auxiliary_grid=auxiliary_grid,
        frequencies=frequencies,
        standard_profile=warped_state,
        smooth_profile=smooth_warped_state,
        standard_evolved=physical_auxiliary_grid,
        smooth_evolved=smooth_physical_auxiliary_grid,
        matrix_reference=matrix_reference,
        recovery_index=recovery_index,
        output_path=output_dir / "heat_equation_auxiliary.png",
    )
    _plot_reconstruction_diagnostics(
        auxiliary_grid=auxiliary_grid,
        standard_operator_errors=standard_operator_errors,
        smooth_operator_errors=smooth_operator_errors,
        standard_evolved=physical_auxiliary_grid,
        smooth_evolved=smooth_physical_auxiliary_grid,
        standard_profile=warped_state,
        smooth_profile=smooth_warped_state,
        standard_probabilities=standard_probabilities,
        smooth_probabilities=smooth_probabilities,
        matrix_reference=matrix_reference,
        recovery_threshold=recovery_threshold,
        recovery_index=recovery_index,
        output_path=output_dir / "heat_equation_reconstruction.png",
    )
    return metrics


def _relative_recovery_error_by_p(
    evolved_state: np.ndarray,
    profile: np.ndarray,
    reference: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Return the physical-state recovery error for selected p slices."""
    recovered_by_p = (
        evolved_state[:, indices] / profile[indices][np.newaxis, :]
    ).real
    return np.linalg.norm(
        recovered_by_p - reference[:, np.newaxis], axis=0
    ) / np.linalg.norm(reference)


def _plot_auxiliary_diagnostics(
    x: np.ndarray,
    auxiliary_grid: np.ndarray,
    frequencies: np.ndarray,
    standard_profile: np.ndarray,
    smooth_profile: np.ndarray,
    standard_evolved: np.ndarray,
    smooth_evolved: np.ndarray,
    matrix_reference: np.ndarray,
    recovery_index: int,
    output_path: Path,
) -> None:
    """Visualize the warped coordinate, Fourier representation and recovery."""
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    profile_window = (auxiliary_grid >= -2.5) & (auxiliary_grid <= 2.5)
    recovery_coordinate = auxiliary_grid[recovery_index]

    axes[0, 0].plot(
        auxiliary_grid[profile_window],
        standard_profile[profile_window],
        color="#8175D6",
        linewidth=2.2,
        linestyle="--",
        label=r"Standard $e^{-|p|}$",
    )
    axes[0, 0].plot(
        auxiliary_grid[profile_window],
        smooth_profile[profile_window],
        color="#D15C95",
        linewidth=2.2,
        label="Cubic smooth extension",
    )
    axes[0, 0].axvspan(-1.0, 0.0, color="#D15C95", alpha=0.08)
    axes[0, 0].axvline(
        recovery_coordinate,
        color="#191747",
        linewidth=1.2,
        linestyle=":",
        label=rf"Recovery $p_{{\mathrm{{rec}}}}={recovery_coordinate:.3f}$",
    )
    axes[0, 0].set_xlabel("Auxiliary coordinate p")
    axes[0, 0].set_ylabel("Warped profile")
    axes[0, 0].set_title("Warped-phase state preparation")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    standard_spectrum = np.abs(np.fft.fftshift(np.fft.ifft(standard_profile)))
    smooth_spectrum = np.abs(np.fft.fftshift(np.fft.ifft(smooth_profile)))
    shifted_frequencies = np.fft.fftshift(frequencies)
    axes[0, 1].semilogy(
        shifted_frequencies,
        np.maximum(standard_spectrum, 1e-18),
        color="#8175D6",
        linewidth=1.7,
        linestyle="--",
        label="Standard profile",
    )
    axes[0, 1].semilogy(
        shifted_frequencies,
        np.maximum(smooth_spectrum, 1e-18),
        color="#D15C95",
        linewidth=1.7,
        label="Smooth extension",
    )
    axes[0, 1].set_xlim(-8.0, 8.0)
    axes[0, 1].set_xlabel(r"Auxiliary frequency $\eta$")
    axes[0, 1].set_ylabel("Fourier magnitude")
    axes[0, 1].set_title("Auxiliary Fourier representation")
    axes[0, 1].grid(alpha=0.25, which="both")
    axes[0, 1].legend(frameon=False, fontsize=8)

    spatial_index = int(np.argmax(np.abs(matrix_reference)))
    axes[1, 0].plot(
        auxiliary_grid[profile_window],
        standard_evolved[spatial_index, profile_window].real,
        color="#8175D6",
        linewidth=2.0,
        linestyle="--",
        label="Evolved standard profile",
    )
    axes[1, 0].plot(
        auxiliary_grid[profile_window],
        smooth_evolved[spatial_index, profile_window].real,
        color="#D15C95",
        linewidth=2.0,
        label="Evolved smooth profile",
    )
    positive_window = profile_window & (auxiliary_grid >= 0.0)
    axes[1, 0].plot(
        auxiliary_grid[positive_window],
        matrix_reference[spatial_index] * standard_profile[positive_window],
        color="#191747",
        linewidth=1.5,
        label=r"Expected $e^{-p}u(t)$ branch",
    )
    axes[1, 0].axvline(
        recovery_coordinate, color="#191747", linewidth=1.2, linestyle=":"
    )
    axes[1, 0].set_xlabel("Auxiliary coordinate p")
    axes[1, 0].set_ylabel(r"$w(t,x,p)$")
    axes[1, 0].set_title(rf"Evolved slice at $x={x[spatial_index]:.3g}$")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)

    positive_indices = np.flatnonzero(
        (auxiliary_grid > 0.0) & (auxiliary_grid <= 3.0)
    )
    standard_errors = _relative_recovery_error_by_p(
        standard_evolved, standard_profile, matrix_reference, positive_indices
    )
    smooth_errors = _relative_recovery_error_by_p(
        smooth_evolved, smooth_profile, matrix_reference, positive_indices
    )
    axes[1, 1].semilogy(
        auxiliary_grid[positive_indices],
        standard_errors,
        color="#8175D6",
        linewidth=2.0,
        linestyle="--",
        label="Standard profile",
    )
    axes[1, 1].semilogy(
        auxiliary_grid[positive_indices],
        smooth_errors,
        color="#D15C95",
        linewidth=2.0,
        label="Smooth extension",
    )
    axes[1, 1].axvline(
        recovery_coordinate, color="#191747", linewidth=1.2, linestyle=":"
    )
    axes[1, 1].set_xlabel("Positive recovery coordinate p")
    axes[1, 1].set_ylabel("Relative recovery error")
    axes[1, 1].set_title("Recovery sensitivity to the p slice")
    axes[1, 1].grid(alpha=0.25, which="both")
    axes[1, 1].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Schrödingerization auxiliary-coordinate diagnostics",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_reconstruction_diagnostics(
    auxiliary_grid: np.ndarray,
    standard_operator_errors: np.ndarray,
    smooth_operator_errors: np.ndarray,
    standard_evolved: np.ndarray,
    smooth_evolved: np.ndarray,
    standard_profile: np.ndarray,
    smooth_profile: np.ndarray,
    standard_probabilities: np.ndarray,
    smooth_probabilities: np.ndarray,
    matrix_reference: np.ndarray,
    recovery_threshold: float,
    recovery_index: int,
    output_path: Path,
) -> None:
    """Plot operator/state reconstruction and postselection diagnostics."""
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    analysis_indices = np.flatnonzero(
        (auxiliary_grid > recovery_threshold)
        & (auxiliary_grid <= 3.0)
        & np.isfinite(standard_operator_errors)
        & np.isfinite(smooth_operator_errors)
    )
    k_values = np.arange(auxiliary_grid.size)

    axes[0, 0].semilogy(
        analysis_indices,
        standard_operator_errors[analysis_indices],
        color="#8175D6",
        linewidth=2.0,
        linestyle="--",
        label="Standard profile",
    )
    axes[0, 0].semilogy(
        analysis_indices,
        smooth_operator_errors[analysis_indices],
        color="#D15C95",
        linewidth=2.0,
        label="Smooth extension",
    )
    axes[0, 0].axvline(
        recovery_index,
        color="#191747",
        linewidth=1.2,
        linestyle=":",
        label=rf"Selected $k={recovery_index}$",
    )
    axes[0, 0].set_xlabel("Auxiliary index k")
    axes[0, 0].set_ylabel("Relative Frobenius error")
    axes[0, 0].set_title(r"Operator reconstruction $\widehat{\Phi}_k$")
    axes[0, 0].grid(alpha=0.25, which="both")
    axes[0, 0].legend(frameon=False, fontsize=8)

    standard_state_errors = _relative_recovery_error_by_p(
        standard_evolved,
        standard_profile,
        matrix_reference,
        analysis_indices,
    )
    smooth_state_errors = _relative_recovery_error_by_p(
        smooth_evolved,
        smooth_profile,
        matrix_reference,
        analysis_indices,
    )
    axes[0, 1].semilogy(
        analysis_indices,
        standard_state_errors,
        color="#8175D6",
        linewidth=2.0,
        linestyle="--",
        label="Standard profile",
    )
    axes[0, 1].semilogy(
        analysis_indices,
        smooth_state_errors,
        color="#D15C95",
        linewidth=2.0,
        label="Smooth extension",
    )
    axes[0, 1].axvline(
        recovery_index, color="#191747", linewidth=1.2, linestyle=":"
    )
    axes[0, 1].set_xlabel("Auxiliary index k")
    axes[0, 1].set_ylabel("Relative state error")
    axes[0, 1].set_title(r"State reconstruction $\widehat{u}_k(t)$")
    axes[0, 1].grid(alpha=0.25, which="both")
    axes[0, 1].legend(frameon=False, fontsize=8)

    probability_floor = 1e-18
    axes[1, 0].semilogy(
        k_values,
        np.maximum(standard_probabilities, probability_floor),
        color="#8175D6",
        linewidth=1.7,
        linestyle="--",
        label="Standard profile",
    )
    axes[1, 0].semilogy(
        k_values,
        np.maximum(smooth_probabilities, probability_floor),
        color="#D15C95",
        linewidth=1.7,
        label="Smooth extension",
    )
    recovery_indices = np.flatnonzero(auxiliary_grid >= recovery_threshold)
    if recovery_indices.size:
        axes[1, 0].axvspan(
            recovery_indices[0],
            recovery_indices[-1],
            color="#8175D6",
            alpha=0.08,
            label=r"Recovery set $I^\star$",
        )
    axes[1, 0].scatter(
        [recovery_index],
        [max(smooth_probabilities[recovery_index], probability_floor)],
        facecolors="none",
        edgecolors="#191747",
        linewidths=1.5,
        s=65,
        zorder=3,
    )
    axes[1, 0].set_xlabel("Auxiliary index k")
    axes[1, 0].set_ylabel(r"Postselection probability $P(k)$")
    axes[1, 0].set_title("Auxiliary measurement landscape")
    axes[1, 0].grid(alpha=0.25, which="both")
    axes[1, 0].legend(frameon=False, fontsize=8)

    standard_cumulative_probability = np.cumsum(
        standard_probabilities[analysis_indices]
    )
    smooth_cumulative_probability = np.cumsum(
        smooth_probabilities[analysis_indices]
    )
    standard_worst_error = np.maximum.accumulate(
        standard_operator_errors[analysis_indices]
    )
    smooth_worst_error = np.maximum.accumulate(
        smooth_operator_errors[analysis_indices]
    )
    axes[1, 1].plot(
        standard_worst_error,
        standard_cumulative_probability,
        color="#8175D6",
        linewidth=2.0,
        linestyle="--",
        label="Standard profile",
    )
    axes[1, 1].plot(
        smooth_worst_error,
        smooth_cumulative_probability,
        color="#D15C95",
        linewidth=2.0,
        label="Smooth extension",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Worst operator error in recovery window")
    axes[1, 1].set_ylabel("Cumulative success probability")
    axes[1, 1].set_title(r"Accuracy–postselection trade-off, $0<p\leq3$")
    axes[1, 1].grid(alpha=0.25, which="both")
    axes[1, 1].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Heat-equation Schrödingerization: reconstruction diagnostics",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot(
    x: np.ndarray,
    standard_recovered: np.ndarray,
    smooth_recovered: np.ndarray,
    analytic: np.ndarray,
    matrix_reference: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(x, analytic, color="#191747", linewidth=2.5, label="Analytic")
    axes[0].plot(
        x,
        matrix_reference,
        color="#8175D6",
        linewidth=2,
        linestyle="--",
        label="Matrix exponential",
    )
    axes[0].plot(
        x,
        standard_recovered,
        "o",
        markerfacecolor="none",
        markeredgecolor="#8175D6",
        markersize=4.5,
        label="Standard-profile recovery",
    )
    axes[0].plot(
        x,
        smooth_recovered,
        "x",
        color="#D15C95",
        markersize=4.5,
        label="Smooth-profile recovery",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].set_title("Recovered dissipative state")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].semilogy(
        x,
        np.maximum(np.abs(standard_recovered - matrix_reference), 1e-16),
        color="#8175D6",
        linewidth=2,
        linestyle="--",
        label="Standard profile",
    )
    axes[1].semilogy(
        x,
        np.maximum(np.abs(smooth_recovered - matrix_reference), 1e-16),
        color="#D15C95",
        linewidth=2,
        label="Smooth extension",
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("Recovery error vs matrix reference")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False)

    figure.suptitle(
        "Heat-equation Schrödingerization: physical-state recovery",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run_demo(arguments.output_dir)
    print(json.dumps(result, indent=2))
