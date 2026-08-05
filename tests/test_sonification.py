import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sonification import render_layer_audio


def test_render_is_deterministic_and_does_not_clip():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    graph.add_edge(0, 1, weight=1.0)
    bloch = {
        "theta": np.array([0.0, np.pi]),
        "phi": np.array([0.0, np.pi]),
        "r": np.array([1.0, 0.5]),
        "linear_entropy": np.array([0.0, 0.75]),
    }

    first = render_layer_audio(graph, bloch, preferred_root=0)
    second = render_layer_audio(graph, bloch, preferred_root=0)

    for mode in first:
        assert np.array_equal(first[mode], second[mode])
        assert np.max(np.abs(first[mode])) <= 1.0
