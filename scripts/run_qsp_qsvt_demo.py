"""Integrated classical demonstration of a QSP--QSVT inversion pipeline.

The script constructs an odd polynomial approximation to the inverse function,
scales it to satisfy a conservative QSP feasibility condition, synthesizes a
QSP phase sequence with a contraction-mapping iteration, validates the scalar
response, applies it to a small matrix, and quantifies conditioning-driven
polynomial-degree growth.

This is a classical numerical demonstration. 
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import expm


def optimal_uniform_error(degree: int, kappa: float) -> float:
    """Return the raw uniform 1/x approximation error used in the study."""
    if degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    if kappa <= 1.0:
        raise ValueError("kappa must be greater than one")
    a = 1.0 / kappa
    order = (degree + 1) // 2
    log_error = (
        order * math.log1p(-a)
        - math.log(a)
        - (order - 1) * math.log1p(a)
    )
    return math.exp(log_error)


def minimum_degree(target_error: float, kappa: float) -> int:
    """Return the minimum odd degree from the analytic error expression."""
    if target_error <= 0.0:
        raise ValueError("target_error must be positive")
    if kappa <= 1.0:
        raise ValueError("kappa must be greater than one")
    a = 1.0 / kappa
    numerator = (
        math.log(1.0 / target_error)
        + math.log(1.0 / a)
        + math.log1p(a)
    )
    denominator = math.log((1.0 + a) / (1.0 - a))
    order = math.ceil(numerator / denominator)
    return 2 * order - 1


def _l_fraction(order: int, x: np.ndarray, a: float) -> np.ndarray:
    alpha = (1.0 + a) / (2.0 * (1.0 - a))
    previous = (x + (1.0 - a) / (1.0 + a)) / alpha
    current = (
        x**2 + (1.0 - a) / (1.0 + a) * x / 2.0 - 0.5
    ) / alpha**2
    if order == 1:
        return previous
    for _ in range(3, order + 1):
        previous, current = (
            current,
            x * current / alpha - previous / (4.0 * alpha**2),
        )
    return current


def _inverse_polynomial_values(
    x: np.ndarray, order: int, a: float
) -> np.ndarray:
    transformed = (2.0 * x**2 - (1.0 + a**2)) / (1.0 - a**2)
    return (
        1.0
        - (-1) ** order
        * (1.0 + a) ** 2
        / (4.0 * a)
        * _l_fraction(order, transformed, a)
    ) / x


def optimal_inverse_polynomial(
    degree: int, kappa: float
) -> np.polynomial.Chebyshev:
    """Construct the optimal odd Chebyshev interpolant for 1/x."""
    if degree < 1 or degree % 2 == 0:
        raise ValueError("degree must be a positive odd integer")
    if kappa <= 1.0:
        raise ValueError("kappa must be greater than one")
    a = 1.0 / kappa
    order = (degree + 1) // 2
    coefficients = np.polynomial.chebyshev.chebinterpolate(
        _inverse_polynomial_values,
        degree,
        args=(order, a),
    )
    coefficients[0::2] = 0.0
    return np.polynomial.Chebyshev(coefficients)


def _reduced_qsp_imaginary_response(
    reduced_phases: np.ndarray,
    x: np.ndarray,
    parity: int,
) -> np.ndarray:
    """Evaluate the reduced symmetric QSP response in a real representation."""
    response = np.zeros(x.size, dtype=float)
    for sample_index, value in enumerate(x):
        theta = np.arccos(np.clip(value, -1.0, 1.0))
        signal_rotation = np.array(
            [
                [np.cos(2.0 * theta), 0.0, -np.sin(2.0 * theta)],
                [0.0, 1.0, 0.0],
                [np.sin(2.0 * theta), 0.0, np.cos(2.0 * theta)],
            ]
        )
        if parity == 0:
            state = np.array([1.0, 0.0, 0.0])
        else:
            state = np.array([np.cos(theta), 0.0, np.sin(theta)])

        for phase in reduced_phases[:-1]:
            phase_rotation = np.array(
                [
                    [np.cos(2.0 * phase), -np.sin(2.0 * phase), 0.0],
                    [np.sin(2.0 * phase), np.cos(2.0 * phase), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            state = signal_rotation @ phase_rotation @ state

        final_phase = reduced_phases[-1]
        response[sample_index] = np.array(
            [np.sin(2.0 * final_phase), np.cos(2.0 * final_phase), 0.0]
        ) @ state
    return response


def _qsp_chebyshev_map(
    reduced_phases: np.ndarray,
    parity: int,
) -> np.ndarray:
    """Map reduced phases to their parity-compatible Chebyshev coefficients."""
    coefficient_count = reduced_phases.size
    half_grid_size = 2 * coefficient_count
    theta = np.arange(coefficient_count + 1) * np.pi / half_grid_size
    samples = np.zeros(2 * half_grid_size, dtype=float)
    samples[: coefficient_count + 1] = _reduced_qsp_imaginary_response(
        reduced_phases,
        np.cos(theta),
        parity,
    )
    samples[coefficient_count + 1 : half_grid_size + 1] = (
        (-1) ** parity * samples[coefficient_count - 1 :: -1]
    )
    samples[half_grid_size + 1 :] = samples[half_grid_size - 1 : 0 : -1]

    coefficients = np.fft.fft(samples).real / (2 * half_grid_size)
    coefficients[1:-1] *= 2.0
    return coefficients[parity::2][:coefficient_count]


def _expand_symmetric_qsp_phases(
    reduced_phases: np.ndarray,
    parity: int,
) -> np.ndarray:
    """Convert reduced real-target phases to the complete symmetric sequence."""
    right = reduced_phases.copy()
    right[-1] += np.pi / 4.0
    phase_count = 2 * right.size - (1 if parity == 0 else 0)
    full = np.zeros(phase_count, dtype=float)
    full[-right.size :] = right
    full[: right.size] += right[::-1]
    return full


def synthesize_qsp_phases(
    partial_chebyshev_coefficients: np.ndarray,
    parity: int,
    tolerance: float = 1e-12,
    maximum_iterations: int = 2000,
) -> tuple[np.ndarray, int, float]:
    """Synthesize symmetric QSP phases by contraction mapping.

    The input contains only coefficients of the selected parity: T_0, T_2, ...
    for parity zero or T_1, T_3, ... for parity one.
    """
    coefficients = np.asarray(partial_chebyshev_coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("partial Chebyshev coefficients must be a nonempty vector")
    if parity not in (0, 1):
        raise ValueError("parity must be zero or one")

    internal_target = -coefficients
    reduced_phases = internal_target / 2.0
    residual_norm = math.inf
    for iteration in range(1, maximum_iterations + 1):
        residual = _qsp_chebyshev_map(reduced_phases, parity) - internal_target
        residual_norm = float(np.linalg.norm(residual, ord=1))
        if residual_norm < tolerance:
            return (
                _expand_symmetric_qsp_phases(reduced_phases, parity),
                iteration,
                residual_norm,
            )
        reduced_phases -= residual / 2.0

    raise RuntimeError(
        "QSP phase synthesis did not converge: "
        f"residual={residual_norm:.3e} after {maximum_iterations} iterations"
    )


def qsp_response(x: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Return the real top-left entry of the scalar QSP sequence."""
    values = np.asarray(x, dtype=float)
    response = np.empty(values.size, dtype=float)
    phase_factors = np.exp(1j * phases)
    for sample_index, value in enumerate(values):
        complement = np.sqrt(max(0.0, 1.0 - value**2))
        signal = np.array(
            [[value, 1j * complement], [1j * complement, value]],
            dtype=complex,
        )
        unitary = np.diag([phase_factors[0], phase_factors[0].conjugate()])
        for phase_factor in phase_factors[1:]:
            phase_gate = np.diag([phase_factor, phase_factor.conjugate()])
            unitary = unitary @ signal @ phase_gate
        response[sample_index] = unitary[0, 0].real
    return response


def time_stacked_matrix() -> np.ndarray:
    """Return a structured 40x40 linear-system stress test."""
    mass = 1.0
    cart_mass = 5.0
    length = 2.0
    gravity = -10.0
    damping = 1.0
    dynamics = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -damping / cart_mass, mass * gravity / cart_mass, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                -damping / (cart_mass * length),
                -(cart_mass + mass) * gravity / (cart_mass * length),
                0.0,
            ],
        ]
    )
    control = np.array(
        [[0.0], [1.0 / cart_mass], [0.0], [1.0 / (cart_mass * length)]]
    )
    state_cost = np.eye(4)
    control_cost = np.array([[1e-2]])
    terminal_cost = np.diag([0.0, 10.0, 50.0, 10.0])
    control_term = control @ np.linalg.solve(control_cost, control.T)
    generator = np.block(
        [[dynamics, -control_term], [-state_cost, -dynamics.T]]
    )

    time_horizon = 3.0
    time_blocks = 5
    block_size = generator.shape[0]
    state_size = block_size // 2
    propagator = expm(generator * time_horizon / (time_blocks - 1))
    matrix = np.zeros((time_blocks * block_size, time_blocks * block_size))
    matrix[:state_size, :state_size] = np.eye(state_size)

    row = state_size
    for block in range(1, time_blocks):
        previous = (block - 1) * block_size
        current = block * block_size
        matrix[row : row + block_size, previous : previous + block_size] = (
            -propagator
        )
        matrix[row : row + block_size, current : current + block_size] = np.eye(
            block_size
        )
        row += block_size

    final_offset = (time_blocks - 1) * block_size
    matrix[row : row + state_size, final_offset : final_offset + state_size] = (
        -terminal_cost
    )
    matrix[
        row : row + state_size,
        final_offset + state_size : final_offset + block_size,
    ] = np.eye(state_size)
    return matrix


def matrix_level_inverse_test(
    phases: np.ndarray,
    kappa: float,
    scaling_factor: float,
) -> dict[str, np.ndarray | float | int]:
    """Apply the synthesized scalar response to a small SPD matrix."""
    eigenvectors = 0.5 * np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    eigenvalues = np.array([1.0, 0.7, 0.35, 1.0 / kappa])
    matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    transformed_values = qsp_response(eigenvalues, phases)
    transformed_matrix = (
        eigenvectors @ np.diag(transformed_values) @ eigenvectors.T
    )
    target_matrix = scaling_factor / kappa * np.linalg.inv(matrix)
    operator_error = float(
        np.linalg.norm(transformed_matrix - target_matrix, ord="fro")
        / np.linalg.norm(target_matrix, ord="fro")
    )

    right_hand_side = np.array([1.0, -0.5, 0.75, -1.25])
    normalized_right_hand_side = right_hand_side / np.linalg.norm(right_hand_side)
    exact_solution = np.linalg.solve(matrix, right_hand_side)
    recovered_solution = (
        kappa / scaling_factor * transformed_matrix @ right_hand_side
    )
    solution_error = float(
        np.linalg.norm(recovered_solution - exact_solution)
        / np.linalg.norm(exact_solution)
    )
    residual = float(
        np.linalg.norm(matrix @ recovered_solution - right_hand_side)
        / np.linalg.norm(right_hand_side)
    )
    postselection_probability = float(
        np.linalg.norm(transformed_matrix @ normalized_right_hand_side) ** 2
    )
    ideal_postselection_probability = float(
        np.linalg.norm(target_matrix @ normalized_right_hand_side) ** 2
    )
    return {
        "dimension": int(matrix.shape[0]),
        "condition_number": float(np.linalg.cond(matrix)),
        "operator_relative_frobenius_error": operator_error,
        "solution_relative_l2_error": solution_error,
        "linear_system_relative_residual": residual,
        "postselection_probability_for_normalized_rhs": (
            postselection_probability
        ),
        "ideal_scaled_inverse_probability_for_normalized_rhs": (
            ideal_postselection_probability
        ),
        "exact_solution": exact_solution,
        "recovered_solution": recovered_solution,
    }


def run_demo(output_dir: Path) -> dict[str, object]:
    """Run the integrated QSP--QSVT demonstration and save its artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    kappa = 6.0
    target_raw_error = 0.1
    qsp_scaling_factor = 0.5
    illustrative_degrees = (5, 11, 15, 25, 35)
    degree_sweep = np.arange(1, 200, 2, dtype=int)
    degree_sweep_errors = np.array(
        [optimal_uniform_error(int(value), kappa) for value in degree_sweep]
    )
    degree = minimum_degree(target_raw_error, kappa)
    inverse_polynomial = optimal_inverse_polynomial(degree, kappa)
    theoretical_raw_error = optimal_uniform_error(degree, kappa)

    gap = 1.0 / kappa
    valid_grid = np.concatenate(
        [np.linspace(-1.0, -gap, 3000), np.linspace(gap, 1.0, 3000)]
    )
    full_grid = np.linspace(-1.0, 1.0, 4001)
    raw_error = float(
        np.max(np.abs(inverse_polynomial(valid_grid) - 1.0 / valid_grid))
    )
    normalized_polynomial = inverse_polynomial / kappa
    unscaled_full_interval_maximum = float(
        np.max(np.abs(normalized_polynomial(full_grid)))
    )

    scaled_coefficients = qsp_scaling_factor * normalized_polynomial.coef
    odd_partial_coefficients = scaled_coefficients[1::2]
    phases, phase_iterations, phase_residual = synthesize_qsp_phases(
        odd_partial_coefficients,
        parity=1,
    )
    polynomial_target_full = np.polynomial.chebyshev.chebval(
        full_grid, scaled_coefficients
    )
    qsp_response_full = qsp_response(full_grid, phases)
    phase_synthesis_error = float(
        np.max(np.abs(qsp_response_full - polynomial_target_full))
    )
    qsp_response_valid = qsp_response(valid_grid, phases)
    scaled_inverse_target = qsp_scaling_factor / (kappa * valid_grid)
    end_to_end_uniform_error = float(
        np.max(np.abs(qsp_response_valid - scaled_inverse_target))
    )

    matrix_test = matrix_level_inverse_test(
        phases,
        kappa,
        qsp_scaling_factor,
    )

    stacked = time_stacked_matrix()
    singular_values = np.linalg.svd(stacked, compute_uv=False)
    stacked_condition_number = float(
        singular_values.max() / singular_values.min()
    )
    stacked_degree = minimum_degree(
        target_raw_error,
        stacked_condition_number,
    )

    metrics: dict[str, object] = {
        "scope": (
            "Classical QSP phase synthesis and QSVT-style matrix-inversion "
            "validation; not a heat-equation or quantum-hardware execution."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inverse_polynomial": {
            "condition_number": kappa,
            "excluded_singular_value_gap": gap,
            "target_raw_uniform_error": target_raw_error,
            "degree": degree,
            "theoretical_raw_uniform_error": theoretical_raw_error,
            "sampled_raw_uniform_error": raw_error,
            "sampled_normalized_uniform_error": raw_error / kappa,
            "unscaled_full_interval_maximum": unscaled_full_interval_maximum,
        },
        "degree_study": {
            "condition_number": kappa,
            "illustrative_degrees": list(illustrative_degrees),
            "illustrative_theoretical_raw_errors": [
                optimal_uniform_error(value, kappa)
                for value in illustrative_degrees
            ],
            "sweep_degrees": degree_sweep.tolist(),
            "sweep_theoretical_raw_errors": degree_sweep_errors.tolist(),
        },
        "qsp_phase_synthesis": {
            "scaling_factor_beta": qsp_scaling_factor,
            "scaled_partial_coefficient_l1_norm": float(
                np.sum(np.abs(odd_partial_coefficients))
            ),
            "scaled_polynomial_full_interval_maximum": float(
                np.max(np.abs(polynomial_target_full))
            ),
            "phase_count": int(phases.size),
            "iterations": phase_iterations,
            "coefficient_residual": phase_residual,
            "maximum_phase_response_error": phase_synthesis_error,
            "end_to_end_scaled_inverse_uniform_error": end_to_end_uniform_error,
            "phases": phases.tolist(),
        },
        "matrix_application": {
            key: value
            for key, value in matrix_test.items()
            if not isinstance(value, np.ndarray)
        },
        "conditioning_stress_test": {
            "matrix_dimension": int(stacked.shape[0]),
            "minimum_singular_value": float(singular_values.min()),
            "maximum_singular_value": float(singular_values.max()),
            "condition_number": stacked_condition_number,
            "required_degree_at_same_raw_error": stacked_degree,
        },
    }
    (output_dir / "qsp_qsvt_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_pipeline(
        inverse_polynomial=inverse_polynomial,
        kappa=kappa,
        scaling_factor=qsp_scaling_factor,
        full_grid=full_grid,
        valid_grid=valid_grid,
        polynomial_target_full=polynomial_target_full,
        qsp_response_full=qsp_response_full,
        exact_solution=np.asarray(matrix_test["exact_solution"]),
        recovered_solution=np.asarray(matrix_test["recovered_solution"]),
        solution_error=float(matrix_test["solution_relative_l2_error"]),
        output_path=output_dir / "qsp_qsvt_pipeline.png",
    )
    _plot_resources(
        phases=phases,
        demonstration_kappa=kappa,
        demonstration_degree=degree,
        target_raw_error=target_raw_error,
        stacked_kappa=stacked_condition_number,
        stacked_degree=stacked_degree,
        output_path=output_dir / "qsp_qsvt_resources.png",
    )
    _plot_inverse_polynomial_progression(
        kappa=kappa,
        degrees=illustrative_degrees,
        output_path=(
            output_dir / "qsp_qsvt_inverse_polynomial_progression.png"
        ),
    )
    _plot_degree_convergence(
        kappa=kappa,
        degrees=degree_sweep,
        errors=degree_sweep_errors,
        selected_degree=degree,
        target_error=target_raw_error,
        output_path=output_dir / "qsp_qsvt_degree_convergence.png",
    )
    return metrics


def _plot_pipeline(
    inverse_polynomial: np.polynomial.Chebyshev,
    kappa: float,
    scaling_factor: float,
    full_grid: np.ndarray,
    valid_grid: np.ndarray,
    polynomial_target_full: np.ndarray,
    qsp_response_full: np.ndarray,
    exact_solution: np.ndarray,
    recovered_solution: np.ndarray,
    solution_error: float,
    output_path: Path,
) -> None:
    """Plot scalar approximation, phase realization, and matrix application."""
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    gap = 1.0 / kappa

    axes[0, 0].plot(
        valid_grid,
        1.0 / (kappa * valid_grid),
        color="#191747",
        linewidth=2.5,
        label=r"Target $1/(\kappa x)$",
    )
    axes[0, 0].plot(
        valid_grid,
        inverse_polynomial(valid_grid) / kappa,
        color="#D15C95",
        linestyle="--",
        linewidth=2.0,
        label="Inverse polynomial",
    )
    axes[0, 0].axvspan(
        -gap,
        gap,
        color="#E8E6F6",
        alpha=0.85,
        label="Excluded gap",
    )
    axes[0, 0].set_xlabel("Scaled singular value x")
    axes[0, 0].set_ylabel("Normalized inverse response")
    axes[0, 0].set_title("1. QSVT inverse-polynomial design")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(
        full_grid,
        polynomial_target_full,
        color="#191747",
        linewidth=2.5,
        label=r"Target $\beta P_d(x)/\kappa$",
    )
    axes[0, 1].plot(
        full_grid,
        qsp_response_full,
        color="#D15C95",
        linestyle="--",
        linewidth=1.8,
        label="Synthesized QSP response",
    )
    axes[0, 1].axvspan(-gap, gap, color="#E8E6F6", alpha=0.55)
    axes[0, 1].set_xlabel("Signal value x")
    axes[0, 1].set_ylabel("QSP response")
    axes[0, 1].set_title(f"2. QSP realization with beta={scaling_factor:g}")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)

    phase_error = np.abs(qsp_response_full - polynomial_target_full)
    axes[1, 0].semilogy(
        full_grid,
        np.maximum(phase_error, 1e-16),
        color="#D15C95",
        linewidth=2.0,
    )
    axes[1, 0].axvspan(-gap, gap, color="#E8E6F6", alpha=0.55)
    axes[1, 0].set_xlabel("Signal value x")
    axes[1, 0].set_ylabel("Absolute synthesis error")
    axes[1, 0].set_title("3. QSP phase-synthesis validation")
    axes[1, 0].grid(alpha=0.25, which="both")

    components = np.arange(exact_solution.size)
    width = 0.36
    axes[1, 1].bar(
        components - width / 2,
        exact_solution,
        width,
        color="#191747",
        label="Direct solve",
    )
    axes[1, 1].bar(
        components + width / 2,
        recovered_solution,
        width,
        color="#D15C95",
        label="QSP--QSVT polynomial",
    )
    axes[1, 1].set_xticks(components)
    axes[1, 1].set_xlabel("Solution component")
    axes[1, 1].set_ylabel("Recovered value")
    axes[1, 1].set_title(
        f"4. Matrix inverse action; relative error={solution_error:.2e}"
    )
    axes[1, 1].grid(alpha=0.25, axis="y")
    axes[1, 1].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Integrated QSP--QSVT matrix-inversion demonstration",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_resources(
    phases: np.ndarray,
    demonstration_kappa: float,
    demonstration_degree: int,
    target_raw_error: float,
    stacked_kappa: float,
    stacked_degree: int,
    output_path: Path,
) -> None:
    """Plot the synthesized phase sequence and conditioning-driven degree."""
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    phase_indices = np.arange(phases.size)
    axes[0].plot(
        phase_indices,
        phases,
        marker="o",
        markersize=4,
        color="#D15C95",
        linewidth=1.8,
    )
    axes[0].axhline(0.0, color="#191747", linewidth=0.8)
    axes[0].set_xlabel("Phase index")
    axes[0].set_ylabel("Phase angle (rad)")
    axes[0].set_title(f"Synthesized symmetric sequence ({phases.size} phases)")
    axes[0].grid(alpha=0.25)

    kappas = np.geomspace(2.0, 2.0e4, 250)
    degrees = np.array(
        [minimum_degree(target_raw_error, value) for value in kappas]
    )
    axes[1].loglog(kappas, degrees, color="#52499E", linewidth=2.5)
    axes[1].scatter(
        [demonstration_kappa, stacked_kappa],
        [demonstration_degree, stacked_degree],
        color=["#D15C95", "#E08B4A"],
        s=55,
        zorder=3,
    )
    axes[1].annotate(
        f"integrated demo\n({demonstration_degree:,})",
        (demonstration_kappa, demonstration_degree),
        xytext=(12, -2),
        textcoords="offset points",
    )
    axes[1].annotate(
        f"structured stack\n({stacked_degree:,})",
        (stacked_kappa, stacked_degree),
        xytext=(-86, 6),
        textcoords="offset points",
    )
    axes[1].set_xlabel("Condition number kappa")
    axes[1].set_ylabel("Minimum odd degree")
    axes[1].set_title(
        f"Degree required for raw error <= {target_raw_error:g}"
    )
    axes[1].grid(alpha=0.25, which="both")

    figure.suptitle(
        "QSP phases and QSVT conditioning diagnostics",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_inverse_polynomial_progression(
    kappa: float,
    degrees: tuple[int, ...],
    output_path: Path,
) -> None:
    """Show how the inverse polynomial approaches 1/x as degree increases."""
    gap = 1.0 / kappa
    full_grid = np.linspace(-1.0, 1.0, 3001)
    negative_grid = np.linspace(-1.0, -gap, 900)
    positive_grid = np.linspace(gap, 1.0, 900)
    largest_error = max(optimal_uniform_error(value, kappa) for value in degrees)
    vertical_limit = 1.08 * (kappa + largest_error)

    figure, axes = plt.subplots(
        1,
        len(degrees),
        figsize=(16.0, 3.5),
        sharex=True,
        sharey=True,
    )
    for panel_index, (axis, degree) in enumerate(zip(axes, degrees)):
        polynomial = optimal_inverse_polynomial(degree, kappa)
        error = optimal_uniform_error(degree, kappa)

        axis.axvspan(-gap, gap, color="#E8E6F6", alpha=0.65)
        for branch in (negative_grid, positive_grid):
            inverse = 1.0 / branch
            axis.fill_between(
                branch,
                inverse - error,
                inverse + error,
                color="#B8B8C5",
                alpha=0.45,
                linewidth=0.0,
                label=(r"$1/x\pm E_d$" if panel_index == 0 and branch is negative_grid else None),
            )
            axis.plot(
                branch,
                inverse,
                color="#191747",
                linewidth=1.7,
                label=(r"Target $1/x$" if panel_index == 0 and branch is negative_grid else None),
            )

        axis.plot(
            full_grid,
            polynomial(full_grid),
            color="#D15C95",
            linewidth=1.8,
            label=(r"Polynomial $P_d(x)$" if panel_index == 0 else None),
        )
        axis.axvline(-gap, color="#666666", linestyle="--", linewidth=1.0)
        axis.axvline(gap, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_title(rf"$d={degree}$, $E_d={error:.3g}$")
        axis.set_xlabel(r"Signal value $x$")
        axis.set_ylim(-vertical_limit, vertical_limit)
        axis.grid(alpha=0.22)
        if panel_index == 0:
            axis.set_ylabel("Raw inverse response")
            axis.legend(frameon=False, fontsize=7, loc="upper left")

    figure.suptitle(
        rf"Inverse-polynomial progression at $\kappa={kappa:g}$",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_degree_convergence(
    kappa: float,
    degrees: np.ndarray,
    errors: np.ndarray,
    selected_degree: int,
    target_error: float,
    output_path: Path,
) -> None:
    """Plot inverse-approximation error against odd polynomial degree."""
    selected_error = optimal_uniform_error(selected_degree, kappa)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for axis in axes:
        axis.plot(degrees, errors, color="#52499E", linewidth=2.2)
        axis.axhline(
            target_error,
            color="#E08B4A",
            linestyle="--",
            linewidth=1.5,
            label=rf"Target $E_d\leq {target_error:g}$",
        )
        axis.scatter(
            [selected_degree],
            [selected_error],
            color="#D15C95",
            s=55,
            zorder=3,
            label=rf"Selected $d={selected_degree}$",
        )
        axis.set_xlabel("Odd polynomial degree d")
        axis.grid(alpha=0.25, which="both")

    axes[0].set_ylabel(r"Raw uniform error $E_d$")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_title("Linear scale: threshold crossing")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Raw uniform error $E_d$")
    axes[1].set_title("Log scale: convergence with degree")
    axes[1].legend(frameon=False, fontsize=8)

    figure.suptitle(
        rf"Inverse-approximation error versus degree at $\kappa={kappa:g}$",
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
