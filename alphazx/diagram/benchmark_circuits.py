"""
Benchmark circuit loader for structured quantum circuits.

Loads fixed benchmark circuits (arithmetic, QFT, Toffoli decompositions)
from Quipper/QASM/QC files.  These circuits have known structure — unlike
the randomly generated circuits used for training — and are the standard
benchmarks used across the ZX-calculus optimisation literature.

Circuit families
----------------
- **Arithmetic_and_Toffoli**: Toffoli decompositions (tof_N, barenco_tof_N),
  GF(2^n) multipliers (gf2^N_mult), modular arithmetic (mod_mult, mod_red,
  mod_adder), carry-lookahead adders (qcla_adder, qcla_com, qcla_mod),
  ripple-carry adders (rc_adder), VBE adder, CSLA mux, CSUM mux.
- **QFT_and_Adders**: QFT circuits (QFT8/16/32), Adder circuits
  (Adder8/16/32/64), QFT-based adders (QFTAdd8/16/32).

These circuits come from the PyZX benchmark suite, originally sourced from
Matthew Amy's T-count optimisation benchmarks.

Usage
-----
::

    from alphazx.diagram.benchmark_circuits import (
        load_benchmark_circuit,
        list_benchmark_circuits,
        BenchmarkCircuit,
        SMALL_BENCHMARKS,
        MEDIUM_BENCHMARKS,
    )

    # Load a single circuit
    bc = load_benchmark_circuit("tof_3_before", "arithmetic")
    state = GameState.from_diagram(bc.zx_diagram)

    # Load the standard small benchmark suite
    for bc in SMALL_BENCHMARKS:
        print(f"{bc.name}: {bc.qubits}q, T={bc.t_count}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyzx

from alphazx.diagram.diagram_generators import graph_to_nx_graph, post_process
from alphazx.diagram.zx_diagram import ZXDiagram
from alphazx.game.zx_game import num_non_clifford_gates

logger = logging.getLogger(__name__)

# Default location for benchmark circuits (relative to project root).
_DEFAULT_BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark_circuits"


@dataclass
class BenchmarkCircuit:
    """A loaded benchmark circuit with its metadata."""
    name: str
    family: str          # "arithmetic" or "qft_adders"
    qubits: int
    gate_count: int
    t_count: int
    zx_diagram: ZXDiagram
    pyzx_graph: object   # pyzx.Graph — kept for baseline comparison


def _load_quipper_file(path: str | Path) -> tuple[pyzx.Circuit, object]:
    """Load a Quipper-format circuit file via PyZX.

    Returns (circuit, pyzx_graph).
    """
    circuit = pyzx.Circuit.from_quipper_file(str(path))
    pyzx_graph = circuit.to_graph()
    return circuit, pyzx_graph


def _load_qasm_file(path: str | Path) -> tuple[pyzx.Circuit, object]:
    """Load a QASM-format circuit file via PyZX."""
    circuit = pyzx.Circuit.from_qasm_file(str(path))
    pyzx_graph = circuit.to_graph()
    return circuit, pyzx_graph


def _load_qc_file(path: str | Path) -> tuple[pyzx.Circuit, object]:
    """Load a .qc-format circuit file via PyZX."""
    circuit = pyzx.Circuit.from_qc_file(str(path))
    pyzx_graph = circuit.to_graph()
    return circuit, pyzx_graph


def _load_auto(path: str | Path) -> tuple[pyzx.Circuit, object]:
    """Auto-detect format and load.  Tries PyZX's Circuit.load first,
    then falls back to Quipper (since benchmark files often lack extensions).
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == '.qasm':
        return _load_qasm_file(path)
    elif ext == '.qc':
        return _load_qc_file(path)
    elif ext in ('.quip', '.quipper'):
        return _load_quipper_file(path)
    else:
        # Most PyZX benchmarks are extensionless Quipper files
        try:
            return _load_quipper_file(path)
        except Exception:
            # Fall back to PyZX's auto-detect
            circuit = pyzx.Circuit.load(path)
            pyzx_graph = circuit.to_graph()
            return circuit, pyzx_graph


def load_benchmark_circuit(
    name: str,
    family: str = "arithmetic",
    benchmark_dir: str | Path | None = None,
) -> BenchmarkCircuit:
    """Load a single benchmark circuit by name and family.

    Parameters
    ----------
    name : str
        Filename (with or without extension) in the family directory.
    family : str
        Subdirectory under benchmark_dir: ``"arithmetic"`` or ``"qft_adders"``.
    benchmark_dir : Path, optional
        Root directory containing family subdirectories.  Defaults to
        ``<project_root>/benchmark_circuits/``.

    Returns
    -------
    BenchmarkCircuit
        Loaded circuit with metadata and ZXDiagram.
    """
    if benchmark_dir is None:
        benchmark_dir = _DEFAULT_BENCHMARK_DIR
    benchmark_dir = Path(benchmark_dir)

    path = benchmark_dir / family / name
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark circuit not found: {path}\n"
            f"Available circuits in '{family}': "
            f"{sorted(os.listdir(benchmark_dir / family))}"
        )

    circuit, pyzx_graph = _load_auto(path)
    nx_graph = graph_to_nx_graph(pyzx_graph)
    post_process(nx_graph)

    # Determine phase denominator from the circuit's actual phases.
    # Clifford+T circuits need denominator 4 (π/4 = T gate).
    # QFT circuits may need 8 or higher (π/8, π/16, ... rotations).
    from alphazx.diagram.pyzx_nx_conv import PHASE
    from fractions import Fraction
    import math
    max_denom = 4  # minimum: Clifford+T
    for _node, ndata in nx_graph.nodes(data=True):
        phase = ndata.get(PHASE, 0.0)
        if phase == 0.0:
            continue
        # Find the smallest denominator d such that phase * d is (close to) integer.
        # Use limit_denominator high enough for QFT32 (finest = 1/2^32, but
        # practical circuits rarely exceed 1/2^16 after ZX conversion).
        frac = Fraction(phase).limit_denominator(65536)
        if frac.denominator > max_denom:
            max_denom = frac.denominator
    # Round up to a power of 2 for clean division
    phase_denom = 2 ** math.ceil(math.log2(max(max_denom, 1)))

    zx_diag = ZXDiagram(phase_denom, nx_graph)
    t_count = num_non_clifford_gates(zx_diag)

    return BenchmarkCircuit(
        name=name,
        family=family,
        qubits=circuit.qubits,
        gate_count=len(circuit.gates),
        t_count=t_count,
        zx_diagram=zx_diag,
        pyzx_graph=pyzx_graph,
    )


def list_benchmark_circuits(
    family: str = "arithmetic",
    benchmark_dir: str | Path | None = None,
    max_qubits: Optional[int] = None,
    max_t_count: Optional[int] = None,
) -> list[BenchmarkCircuit]:
    """Load all benchmark circuits in a family, optionally filtered by size.

    Parameters
    ----------
    family : str
        ``"arithmetic"`` or ``"qft_adders"``.
    benchmark_dir : Path, optional
        Root benchmark directory.
    max_qubits : int, optional
        Skip circuits with more qubits than this.
    max_t_count : int, optional
        Skip circuits with more T-gates than this.

    Returns
    -------
    list[BenchmarkCircuit]
        Sorted by T-count (ascending).
    """
    if benchmark_dir is None:
        benchmark_dir = _DEFAULT_BENCHMARK_DIR
    benchmark_dir = Path(benchmark_dir)

    family_dir = benchmark_dir / family
    if not family_dir.exists():
        logger.warning("Benchmark family directory not found: %s", family_dir)
        return []

    circuits = []
    for name in sorted(os.listdir(family_dir)):
        path = family_dir / name
        if path.is_dir():
            continue
        try:
            bc = load_benchmark_circuit(name, family, benchmark_dir)
        except Exception as e:
            logger.warning("Failed to load benchmark %s/%s: %s", family, name, e)
            continue

        if max_qubits is not None and bc.qubits > max_qubits:
            continue
        if max_t_count is not None and bc.t_count > max_t_count:
            continue
        circuits.append(bc)

    circuits.sort(key=lambda c: c.t_count)
    return circuits


def load_all_benchmarks(
    benchmark_dir: str | Path | None = None,
    max_qubits: Optional[int] = None,
    max_t_count: Optional[int] = None,
) -> list[BenchmarkCircuit]:
    """Load benchmark circuits from all families."""
    results = []
    for family in ("arithmetic", "qft_adders"):
        results.extend(list_benchmark_circuits(
            family, benchmark_dir, max_qubits, max_t_count,
        ))
    results.sort(key=lambda c: c.t_count)
    return results


# -----------------------------------------------------------------------
# Pre-defined benchmark suites (by size tier)
# -----------------------------------------------------------------------

# Small: circuits with ≤15 qubits and ≤120 T-gates.
# These are feasible for the agent at training scale.
_SMALL_SPECS = [
    ("tof_3_before", "arithmetic"),           #  5q,  T=21
    ("mod5_4_before", "arithmetic"),           #  5q,  T=28
    ("barenco_tof_3_before", "arithmetic"),    #  5q,  T=28
    ("tof_4_before", "arithmetic"),            #  7q,  T=35
    ("mod_mult_55_before", "arithmetic"),      #  9q,  T=49
    ("tof_5_before", "arithmetic"),            #  9q,  T=49
    ("vbe_adder_3_before", "arithmetic"),      # 10q,  T=70
    ("csla_mux_3_original_before", "arithmetic"),  # 15q, T=70
    ("rc_adder_6_before", "arithmetic"),       # 14q,  T=77
    ("barenco_tof_4_before", "arithmetic"),    #  7q,  T=56
    ("barenco_tof_5_before", "arithmetic"),    #  9q,  T=84
    ("QFT8_before", "qft_adders"),            #  8q,  T=84
    ("gf2^4_mult_before", "arithmetic"),      # 12q,  T=112
    ("tof_10_before", "arithmetic"),           # 19q,  T=119
    ("mod_red_21_before", "arithmetic"),       # 11q,  T=119
]

# Medium: circuits with ≤30 qubits and ≤500 T-gates.
# These test cross-scale generalization.
_MEDIUM_SPECS = [
    ("gf2^5_mult_before", "arithmetic"),      # 15q,  T=175
    ("csum_mux_9_corrected_before", "arithmetic"),  # 30q, T=196
    ("qcla_com_7_before", "arithmetic"),      # 24q,  T=203
    ("barenco_tof_10_before", "arithmetic"),   # 19q,  T=224
    ("qcla_adder_10_before", "arithmetic"),   # 36q,  T=238 (36q — slightly over)
    ("gf2^6_mult_before", "arithmetic"),      # 18q,  T=252
    ("QFTAdd8_before", "qft_adders"),         # 16q,  T=252
    ("Adder8_before", "qft_adders"),          # 23q,  T=266
    ("gf2^7_mult_before", "arithmetic"),      # 21q,  T=343
    ("QFT16_before", "qft_adders"),           # 16q,  T=342
    ("adder_8_before", "arithmetic"),          # 24q,  T=399
    ("qcla_mod_7_before", "arithmetic"),      # 26q,  T=413
    ("gf2^8_mult_before", "arithmetic"),      # 24q,  T=448
]

# Large: circuits with >30 qubits or >500 T-gates.
# Stress tests for the agent.
_LARGE_SPECS = [
    ("gf2^9_mult_before", "arithmetic"),      # 27q,  T=567
    ("Adder16_before", "qft_adders"),         # 47q,  T=602
    ("gf2^10_mult_before", "arithmetic"),     # 30q,  T=700
    ("QFT32_before", "qft_adders"),           # 32q,  T=918
    ("QFTAdd16_before", "qft_adders"),        # 32q,  T=1026
    ("Adder32_before", "qft_adders"),         # 95q,  T=1274
    ("gf2^16_mult_before", "arithmetic"),     # 48q,  T=1792
    ("mod_adder_1024_before", "arithmetic"),  # 28q,  T=1995
]


def _load_specs(specs: list[tuple[str, str]]) -> list[BenchmarkCircuit]:
    """Load circuits from a spec list, skipping any that fail."""
    results = []
    for name, family in specs:
        try:
            results.append(load_benchmark_circuit(name, family))
        except Exception as e:
            logger.warning("Failed to load %s/%s: %s", family, name, e)
    return results


def get_small_benchmarks() -> list[BenchmarkCircuit]:
    """Load the small benchmark suite (≤15q, ≤120 T-gates)."""
    return _load_specs(_SMALL_SPECS)


def get_medium_benchmarks() -> list[BenchmarkCircuit]:
    """Load the medium benchmark suite (≤30q, ≤500 T-gates)."""
    return _load_specs(_MEDIUM_SPECS)


def get_large_benchmarks() -> list[BenchmarkCircuit]:
    """Load the large benchmark suite (>500 T-gates)."""
    return _load_specs(_LARGE_SPECS)
