"""Two reproducible checks of Carleman linearization and Schrodingerization.

The first check is a scalar quadratic ODE with a closed-form solution. The
second is a coupled three-component quadratic ODE whose triangular structure
also provides a closed-form solution. Both checks separate Carleman truncation
error from auxiliary Schrodingerization and recovery error.

All calculations are classical numerical validations. No quantum circuit or
quantum-hardware execution is claimed.
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


NAVY = "#191747"
PURPLE = "#52499E"
LAVENDER = "#8175D6"
PINK = "#D15C95"
ORANGE = "#E08B4A"


def lift_state(state: np.ndarray, order: int) -> np.ndarray:
    """Return ``[x, x tensor x, ..., x**tensor order]``."""
    vector = np.asarray(state, dtype=complex).reshape(-1)
    if vector.size == 0 or order < 1:
        raise ValueError("state must be nonempty and order must be positive")
    blocks: list[np.ndarray] = []
    block = vector.copy()
    for degree in range(1, order + 1):
        if degree > 1:
            block = np.kron(block, vector)
        blocks.append(block)
    return np.concatenate(blocks)


def carleman_generator(
    linear: np.ndarray,
    quadratic: np.ndarray,
    order: int,
) -> np.ndarray:
    """Build the full order-truncated lift for x' = A x + B(x tensor x)."""
    matrix = np.asarray(linear, dtype=complex)
    tensor = np.asarray(quadratic, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("linear must be square")
    state_dimension = matrix.shape[0]
    if tensor.shape != (state_dimension, state_dimension**2):
        raise ValueError("quadratic must have shape (n, n**2)")
    if order < 1:
        raise ValueError("order must be positive")

    sizes = [state_dimension**degree for degree in range(1, order + 1)]
    offsets = np.cumsum([0, *sizes])
    generator = np.zeros((offsets[-1], offsets[-1]), dtype=complex)

    for degree in range(1, order + 1):
        row = slice(offsets[degree - 1], offsets[degree])
        linear_block = np.zeros((sizes[degree - 1],) * 2, dtype=complex)
        for position in range(degree):
            left = np.eye(state_dimension**position, dtype=complex)
            right = np.eye(
                state_dimension ** (degree - position - 1), dtype=complex
            )
            linear_block += np.kron(np.kron(left, matrix), right)
        generator[row, row] = linear_block

        if degree < order:
            column = slice(offsets[degree], offsets[degree + 1])
            quadratic_block = np.zeros(
                (state_dimension**degree, state_dimension ** (degree + 1)),
                dtype=complex,
            )
            for position in range(degree):
                left = np.eye(state_dimension**position, dtype=complex)
                right = np.eye(
                    state_dimension ** (degree - position - 1), dtype=complex
                )
                quadratic_block += np.kron(np.kron(left, tensor), right)
            generator[row, column] = quadratic_block
    return generator


def carleman_trajectory(
    linear: np.ndarray,
    quadratic: np.ndarray,
    initial_state: np.ndarray,
    times: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolve a truncated Carleman lift at evenly spaced times."""
    generator = carleman_generator(linear, quadratic, order)
    lifted_initial = lift_state(initial_state, order)
    lifted_trajectory = expm_multiply(
        sparse.csr_matrix(generator),
        lifted_initial,
        start=float(times[0]),
        stop=float(times[-1]),
        num=times.size,
        endpoint=True,
        traceA=np.trace(generator),
    )
    state_dimension = np.asarray(initial_state).size
    return lifted_trajectory[:, :state_dimension], lifted_trajectory, generator


def warped_profile(p_grid: np.ndarray) -> np.ndarray:
    """Return the C1 cubic extension of the exponential warped profile."""
    profile = np.exp(-np.abs(p_grid))
    bridge = (p_grid > -1.0) & (p_grid < 0.0)
    profile[bridge] = (
        (3.0 / np.e - 3.0) * p_grid[bridge] ** 3
        + (4.0 / np.e - 5.0) * p_grid[bridge] ** 2
        - p_grid[bridge]
        + 1.0
    )
    return profile


def schrodingerize_generator(
    generator: np.ndarray,
    initial_state: np.ndarray,
    final_time: float,
    *,
    auxiliary_points: int,
    auxiliary_half_width: float,
) -> dict[str, object]:
    """Recover a linear evolution through a sparse warped-phase Hamiltonian."""
    matrix = np.asarray(generator, dtype=complex)
    initial = np.asarray(initial_state, dtype=complex).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("generator must be square")
    if initial.shape != (matrix.shape[0],):
        raise ValueError("initial_state has incompatible dimension")
    if auxiliary_points < 16 or auxiliary_points & (auxiliary_points - 1):
        raise ValueError("auxiliary_points must be a power of two")
    if auxiliary_half_width <= 0.0:
        raise ValueError("auxiliary_half_width must be positive")

    hermitian_part = 0.5 * (matrix + matrix.conj().T)
    antihermitian_part = (matrix - matrix.conj().T) / (2.0j)
    decomposition_error = float(
        np.linalg.norm(
            matrix - (hermitian_part + 1.0j * antihermitian_part),
            ord=np.inf,
        )
    )
    component_hermiticity_error = float(
        max(
            np.linalg.norm(
                hermitian_part - hermitian_part.conj().T, ord=np.inf
            ),
            np.linalg.norm(
                antihermitian_part - antihermitian_part.conj().T,
                ord=np.inf,
            ),
        )
    )
    largest_hermitian_eigenvalue = float(
        np.max(np.linalg.eigvalsh(hermitian_part)).real
    )
    recovery_threshold = final_time * max(
        0.0, largest_hermitian_eigenvalue
    )

    spacing = 2.0 * auxiliary_half_width / auxiliary_points
    p_grid = (
        np.arange(auxiliary_points) - auxiliary_points // 2
    ) * spacing
    candidates = np.flatnonzero(
        p_grid > recovery_threshold + 10.0 * np.finfo(float).eps
    )
    if candidates.size == 0:
        raise ValueError("auxiliary domain does not contain the recovery region")
    recovery_index = int(candidates[0])

    frequencies = 2.0 * np.pi * np.fft.fftfreq(
        auxiliary_points, d=spacing
    )
    profile = warped_profile(p_grid)
    warped_initial = initial[:, np.newaxis] * profile[np.newaxis, :]
    frequency_initial = np.fft.fft(warped_initial, axis=1, norm="ortho")

    auxiliary_frequency = sparse.diags(frequencies, format="csr")
    auxiliary_identity = sparse.eye(auxiliary_points, format="csr")
    hamiltonian = sparse.kron(
        sparse.csr_matrix(hermitian_part),
        auxiliary_frequency,
        format="csr",
    ) - sparse.kron(
        sparse.csr_matrix(antihermitian_part),
        auxiliary_identity,
        format="csr",
    )
    hamiltonian_difference = hamiltonian - hamiltonian.getH()
    hamiltonian_error = (
        float(np.max(np.abs(hamiltonian_difference.data)))
        if hamiltonian_difference.nnz
        else 0.0
    )

    frequency_vector = frequency_initial.reshape(-1)
    evolution_generator = -1.0j * hamiltonian * final_time
    evolved_vector = expm_multiply(
        evolution_generator,
        frequency_vector,
        traceA=evolution_generator.diagonal().sum(),
    )
    frequency_final = evolved_vector.reshape(initial.size, auxiliary_points)
    warped_final = np.fft.ifft(frequency_final, axis=1, norm="ortho")
    recovered = warped_final[:, recovery_index] / profile[recovery_index]

    joint_norm_squared = (
        np.linalg.norm(initial) ** 2 * np.linalg.norm(profile) ** 2
    )
    probabilities = (
        np.sum(np.abs(warped_final) ** 2, axis=0) / joint_norm_squared
    ).real
    accepted = p_grid > recovery_threshold
    norm_drift = float(
        abs(np.linalg.norm(evolved_vector) - np.linalg.norm(frequency_vector))
        / np.linalg.norm(frequency_vector)
    )
    return {
        "recovered_state": recovered,
        "p_grid": p_grid,
        "profile": profile,
        "warped_state": warped_final,
        "probabilities": probabilities,
        "recovery_index": recovery_index,
        "recovery_coordinate": float(p_grid[recovery_index]),
        "recovery_threshold": recovery_threshold,
        "largest_hermitian_part_eigenvalue": largest_hermitian_eigenvalue,
        "decomposition_error": decomposition_error,
        "hamiltonian_hermiticity_error": max(
            hamiltonian_error, component_hermiticity_error
        ),
        "relative_norm_drift": norm_drift,
        "selected_outcome_probability": float(probabilities[recovery_index]),
        "accepted_region_probability": float(np.sum(probabilities[accepted])),
        "enlarged_dimension": int(hamiltonian.shape[0]),
    }


def scalar_exact_solution(
    initial_value: float,
    times: np.ndarray,
    decay: float,
    quadratic_growth: float,
) -> np.ndarray:
    """Return the exact solution of x' = -a x + b x^2."""
    exponential = np.exp(-decay * times)
    denominator = 1.0 - quadratic_growth * initial_value / decay * (
        1.0 - exponential
    )
    return initial_value * exponential / denominator


def run_scalar_validation() -> tuple[dict[str, object], dict[str, object]]:
    """Validate scalar Carleman convergence against a closed-form solution."""
    initial_value = 0.4
    final_time = 1.0
    times = np.linspace(0.0, final_time, 101)
    exact = scalar_exact_solution(initial_value, times, 1.0, 0.2)
    linear = np.array([[-1.0]])
    quadratic = np.array([[0.2]])
    orders = tuple(range(1, 7))
    records: list[dict[str, float | int]] = []
    trajectories: dict[int, np.ndarray] = {}
    representative: dict[str, object] | None = None

    for order in orders:
        states, _, generator = carleman_trajectory(
            linear, quadratic, np.array([initial_value]), times, order
        )
        trajectories[order] = states[:, 0].real
        transformed = schrodingerize_generator(
            generator,
            lift_state(np.array([initial_value]), order),
            final_time,
            auxiliary_points=512,
            auxiliary_half_width=8.0,
        )
        carleman_value = float(states[-1, 0].real)
        transformed_value = float(
            np.asarray(transformed["recovered_state"])[0].real
        )
        exact_value = float(exact[-1])
        records.append(
            {
                "order": order,
                "lifted_dimension": int(generator.shape[0]),
                "exact_value": exact_value,
                "carleman_value": carleman_value,
                "schrodingerized_value": transformed_value,
                "carleman_absolute_error": abs(carleman_value - exact_value),
                "schrodingerization_absolute_error": abs(
                    transformed_value - carleman_value
                ),
                "end_to_end_absolute_error": abs(
                    transformed_value - exact_value
                ),
                "recovery_coordinate": float(
                    transformed["recovery_coordinate"]
                ),
                "accepted_region_probability": float(
                    transformed["accepted_region_probability"]
                ),
                "relative_norm_drift": float(
                    transformed["relative_norm_drift"]
                ),
            }
        )
        if order == 4:
            representative = transformed

    metrics: dict[str, object] = {
        "equation": "x' = -x + 0.2 x^2",
        "analytical_solution": (
            "x0 exp(-t) / (1 - 0.2 x0 (1 - exp(-t)))"
        ),
        "initial_value": initial_value,
        "final_time": final_time,
        "auxiliary_points": 512,
        "auxiliary_half_width": 8.0,
        "orders": records,
    }
    plot_data: dict[str, object] = {
        "times": times,
        "exact": exact,
        "trajectories": trajectories,
        "records": records,
        "representative_transformation": representative,
        "representative_carleman_value": trajectories[4][-1],
    }
    return metrics, plot_data


def triangular_quadratic_model() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float, float
]:
    """Return a coupled vector system with a closed-form solution."""
    alpha = 0.5
    beta = 0.75
    linear = np.diag([-1.0, -2.0, -3.0])
    quadratic = np.zeros((3, 9))
    quadratic[1, 0] = alpha
    quadratic[2, 1] = beta / 2.0
    quadratic[2, 3] = beta / 2.0
    initial = np.array([0.8, 0.3, 0.1])
    return linear, quadratic, initial, alpha, beta


def triangular_exact_solution(
    times: np.ndarray,
    initial: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Return the exact solution of the triangular quadratic system."""
    x0, y0, z0 = initial
    exponential = np.exp(-times)
    return np.column_stack(
        (
            x0 * exponential,
            exponential**2 * (y0 + alpha * x0**2 * times),
            exponential**3
            * (
                z0
                + beta * x0 * y0 * times
                + 0.5 * alpha * beta * x0**3 * times**2
            ),
        )
    )


def run_vector_validation() -> tuple[dict[str, object], dict[str, object]]:
    """Validate vector Carleman convergence against a closed-form solution."""
    linear, quadratic, initial, alpha, beta = triangular_quadratic_model()
    final_time = 1.0
    times = np.linspace(0.0, final_time, 101)
    exact = triangular_exact_solution(times, initial, alpha, beta)
    orders = (1, 2, 3)
    trajectories: dict[int, np.ndarray] = {}
    lifted_trajectories: dict[int, np.ndarray] = {}
    generators: dict[int, np.ndarray] = {}
    records: list[dict[str, float | int]] = []

    for order in orders:
        states, lifted_states, generator = carleman_trajectory(
            linear, quadratic, initial, times, order
        )
        states = states.real
        trajectories[order] = states
        lifted_trajectories[order] = lifted_states
        generators[order] = generator
        errors = np.linalg.norm(states - exact, axis=1) / np.linalg.norm(
            exact, axis=1
        )
        records.append(
            {
                "order": order,
                "lifted_dimension": int(generator.shape[0]),
                "maximum_relative_state_error": float(np.max(errors)),
                "final_relative_state_error": float(errors[-1]),
            }
        )

    selected_order = 3
    transformed = schrodingerize_generator(
        generators[selected_order],
        lift_state(initial, selected_order),
        final_time,
        auxiliary_points=1024,
        auxiliary_half_width=12.0,
    )
    transformed_lifted = np.asarray(transformed["recovered_state"])
    transformed_state = transformed_lifted[: initial.size].real
    carleman_final = trajectories[selected_order][-1]
    exact_final = exact[-1]
    schrodingerization_error = float(
        np.linalg.norm(transformed_state - carleman_final)
        / np.linalg.norm(carleman_final)
    )
    lifted_error = float(
        np.linalg.norm(
            transformed_lifted - lifted_trajectories[selected_order][-1]
        )
        / np.linalg.norm(lifted_trajectories[selected_order][-1])
    )
    end_to_end_error = float(
        np.linalg.norm(transformed_state - exact_final)
        / np.linalg.norm(exact_final)
    )
    component_relative_errors = np.abs(
        transformed_state - exact_final
    ) / np.abs(exact_final)

    metrics: dict[str, object] = {
        "equation": {
            "x": "x' = -x",
            "y": "y' = -2 y + 0.5 x^2",
            "z": "z' = -3 z + 0.75 x y",
        },
        "analytical_solution": {
            "x": "x0 exp(-t)",
            "y": "exp(-2t) (y0 + 0.5 x0^2 t)",
            "z": "exp(-3t) (z0 + 0.75 x0 y0 t + 0.1875 x0^3 t^2)",
        },
        "initial_state": initial.tolist(),
        "final_time": final_time,
        "state_dimension": int(initial.size),
        "nonzero_quadratic_tensor_entries": int(np.count_nonzero(quadratic)),
        "initial_quadratic_rhs_norm": float(
            np.linalg.norm(quadratic @ np.kron(initial, initial))
        ),
        "exact_final_state": exact_final.tolist(),
        "carleman_orders": records,
        "schrodingerization": {
            "carleman_order": selected_order,
            "lifted_dimension": int(generators[selected_order].shape[0]),
            "auxiliary_points": 1024,
            "auxiliary_half_width": 12.0,
            "enlarged_dimension": int(transformed["enlarged_dimension"]),
            "profile": "C1 cubic extension of exp(-abs(p))",
            "largest_hermitian_part_eigenvalue": float(
                transformed["largest_hermitian_part_eigenvalue"]
            ),
            "recovery_threshold": float(transformed["recovery_threshold"]),
            "recovery_coordinate": float(transformed["recovery_coordinate"]),
            "physical_state_relative_error_vs_carleman": (
                schrodingerization_error
            ),
            "lifted_state_relative_error_vs_matrix_exponential": lifted_error,
            "end_to_end_relative_error_vs_analytical": end_to_end_error,
            "component_relative_error_vs_analytical": {
                label: float(error)
                for label, error in zip(
                    ("x", "y", "z"), component_relative_errors
                )
            },
            "selected_outcome_probability": float(
                transformed["selected_outcome_probability"]
            ),
            "accepted_region_probability": float(
                transformed["accepted_region_probability"]
            ),
            "generator_decomposition_error": float(
                transformed["decomposition_error"]
            ),
            "hamiltonian_hermiticity_error": float(
                transformed["hamiltonian_hermiticity_error"]
            ),
            "relative_enlarged_norm_drift": float(
                transformed["relative_norm_drift"]
            ),
        },
    }
    plot_data: dict[str, object] = {
        "times": times,
        "exact": exact,
        "trajectories": trajectories,
        "records": records,
        "transformed_state": transformed_state,
        "component_relative_errors": component_relative_errors,
        "schrodingerization_error": schrodingerization_error,
        "end_to_end_error": end_to_end_error,
    }
    return metrics, plot_data


def _plot_scalar_validation(data: dict[str, object], output_path: Path) -> None:
    """Plot the scalar trajectory, errors and auxiliary recovery."""
    times = np.asarray(data["times"])
    exact = np.asarray(data["exact"])
    trajectories = data["trajectories"]
    records = data["records"]
    transformed = data["representative_transformation"]
    if not isinstance(trajectories, dict) or not isinstance(records, list):
        raise TypeError("invalid scalar plot data")
    if not isinstance(transformed, dict):
        raise TypeError("invalid scalar diagnostic data")

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    axes[0].plot(times, exact, color=NAVY, linewidth=2.6, label="Exact")
    styles = {1: (ORANGE, "--"), 2: (LAVENDER, "--"), 4: (PURPLE, "-."), 6: (PINK, "-")}
    for order, (color, style) in styles.items():
        axes[0].plot(
            times,
            np.asarray(trajectories[order]),
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=f"Carleman K={order}",
        )
    axes[0].set_title("1. Exact trajectory and truncated lifts")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("x(t)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    orders = [int(row["order"]) for row in records]
    axes[1].semilogy(
        orders,
        [float(row["carleman_absolute_error"]) for row in records],
        "o-",
        color=NAVY,
        linewidth=2.0,
        label="Carleman truncation",
    )
    axes[1].semilogy(
        orders,
        [max(float(row["schrodingerization_absolute_error"]), 1e-16) for row in records],
        "s--",
        color=LAVENDER,
        linewidth=2.0,
        label="Schrödingerization",
    )
    axes[1].semilogy(
        orders,
        [float(row["end_to_end_absolute_error"]) for row in records],
        "^-.",
        color=PINK,
        linewidth=2.0,
        label="End to end",
    )
    axes[1].set_title("2. Final-time error separation")
    axes[1].set_xlabel("Carleman order K")
    axes[1].set_ylabel("Absolute error")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=8)

    p_grid = np.asarray(transformed["p_grid"])
    warped_state = np.asarray(transformed["warped_state"])[0].real
    profile = np.asarray(transformed["profile"])
    mask = (p_grid >= 0.0) & (p_grid <= 3.0)
    expected = float(data["representative_carleman_value"]) * profile
    recovery_coordinate = float(transformed["recovery_coordinate"])
    axes[2].plot(
        p_grid[mask], expected[mask], color=NAVY, linewidth=2.3, label="Expected"
    )
    axes[2].plot(
        p_grid[mask],
        warped_state[mask],
        color=PINK,
        linestyle="--",
        linewidth=2.0,
        label="Evolved enlarged state",
    )
    axes[2].axvline(
        recovery_coordinate,
        color=ORANGE,
        linestyle=":",
        linewidth=1.8,
        label=f"Recovery p={recovery_coordinate:.3f}",
    )
    axes[2].set_title("3. Auxiliary recovery of x(t)")
    axes[2].set_xlabel("Auxiliary coordinate p")
    axes[2].set_ylabel("Warped physical component x")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Check 1: scalar quadratic equation",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_vector_validation(data: dict[str, object], output_path: Path) -> None:
    """Plot the analytical and recovered vector-system trajectories."""
    times = np.asarray(data["times"])
    exact = np.asarray(data["exact"])
    trajectories = data["trajectories"]
    records = data["records"]
    if not isinstance(trajectories, dict) or not isinstance(records, list):
        raise TypeError("invalid vector plot data")

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    for index, (label, color) in enumerate(
        (("x(t)", NAVY), ("y(t)", PURPLE), ("z(t)", PINK))
    ):
        axes[0, 0].plot(
            times, exact[:, index], color=color, linewidth=2.2, label=label
        )
    axes[0, 0].set_title("1. Closed-form analytical solution")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Component value")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        times, exact[:, 2], color=NAVY, linewidth=2.6, label="Exact z(t)"
    )
    for order, color, style in (
        (1, ORANGE, "--"),
        (2, LAVENDER, "-."),
        (3, PINK, ":"),
    ):
        axes[0, 1].plot(
            times,
            np.asarray(trajectories[order])[:, 2],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=f"Carleman K={order}",
        )
    axes[0, 1].set_title("2. Component requiring order K=3")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("z(t)")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)

    for order, color in zip((1, 2, 3), (ORANGE, LAVENDER, PINK)):
        states = np.asarray(trajectories[order])
        errors = np.linalg.norm(states - exact, axis=1) / np.linalg.norm(
            exact, axis=1
        )
        axes[1, 0].semilogy(
            times,
            np.maximum(errors, 1e-16),
            color=color,
            linewidth=2.0,
            label=f"K={order}",
        )
    axes[1, 0].set_title("3. Carleman convergence")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Relative state error")
    axes[1, 0].grid(alpha=0.25, which="both")
    axes[1, 0].legend(frameon=False)

    labels = ["x", "y", "z"]
    values = np.asarray(data["component_relative_errors"])
    bars = axes[1, 1].bar(
        np.arange(len(labels)),
        values,
        color=[NAVY, PURPLE, PINK],
    )
    axes[1, 1].set_xticks(np.arange(len(labels)), labels)
    axes[1, 1].set_ylim(0.0, 1.22 * float(np.max(values)))
    axes[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 1].set_title("4. Component-wise Schr. recovery error")
    axes[1, 1].set_ylabel("Relative error vs analytical value")
    axes[1, 1].grid(alpha=0.25, axis="y")
    for bar, value in zip(bars, values):
        axes[1, 1].annotate(
            f"{value:.2e}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(
        "Check 2: coupled vector quadratic equation",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_demo(output_dir: Path) -> dict[str, object]:
    """Run both analytical validation checks and save metrics and figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_metrics, scalar_plot_data = run_scalar_validation()
    vector_metrics, vector_plot_data = run_vector_validation()
    metrics: dict[str, object] = {
        "scope": (
            "Classical validation of Carleman linearization followed by "
            "Schrodingerization on two ODEs with analytical solutions."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "scalar_quadratic_check": scalar_metrics,
        "vector_quadratic_check": vector_metrics,
    }
    (output_dir / "carleman_schrodingerization_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _plot_scalar_validation(
        scalar_plot_data, output_dir / "carleman_scalar_check.png"
    )
    _plot_vector_validation(
        vector_plot_data, output_dir / "carleman_vector_check.png"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Directory for metrics and figures (default: repository results/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run_demo(arguments.output_dir), indent=2))
