import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit import apply_random_circuit
from simulations import (
    build_mutual_information_graph,
    get_bloch_data_from_statevector,
)


def test_rqc_is_prefix_consistent():
    one_layer = apply_random_circuit(4, 1, seed=7)
    two_layers = apply_random_circuit(4, 2, seed=7)
    assert two_layers.data[: len(one_layer.data)] == one_layer.data


def test_bloch_for_zero_and_plus_states():
    zero = get_bloch_data_from_statevector(np.array([1.0, 0.0], complex), 1)
    assert np.allclose([zero["x"][0], zero["y"][0], zero["z"][0]], [0, 0, 1])
    assert np.isclose(zero["r"][0], 1.0)
    assert np.isclose(zero["linear_entropy"][0], 0.0)

    plus = get_bloch_data_from_statevector(np.array([1.0, 1.0], complex) / np.sqrt(2), 1)
    assert np.allclose([plus["x"][0], plus["y"][0], plus["z"][0]], [1, 0, 0])
    assert np.isclose(plus["theta"][0], np.pi / 2)


def test_bell_state_has_mixed_local_bloch_and_two_bits_of_mi():
    bell = np.array([1.0, 0.0, 0.0, 1.0], complex) / np.sqrt(2)
    bloch = get_bloch_data_from_statevector(bell, 2)
    assert np.allclose(bloch["r"], [0.0, 0.0], atol=1e-10)
    assert np.allclose(bloch["linear_entropy"], [1.0, 1.0], atol=1e-10)

    graph, matrix = build_mutual_information_graph(bell, 2)
    assert np.isclose(matrix[0, 1], 2.0, atol=1e-10)
    assert np.isclose(graph[0][1]["weight"], 2.0, atol=1e-10)


def test_product_state_has_zero_mutual_information():
    product = np.array([1.0, 0.0, 0.0, 0.0], complex)
    graph, matrix = build_mutual_information_graph(product, 2)
    assert np.allclose(matrix, 0.0)
    assert graph.number_of_edges() == 0
