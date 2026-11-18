import numpy as np
import pytest
from roseate import context


@pytest.fixture
def saved(tmp_path):
    def _save(rows):
        path = tmp_path / 'contexts.npy'
        np.save(path, np.array(rows))
        return path
    return _save


def test_load_contexts_accepts_matching_indices(saved):
    ctx = context.load_contexts(saved([[1, 2], [0, 2], [0, 1]]), n_queries=3, n_context_seqs=3, exclude_self=True)
    assert ctx.shape == (3, 2)

@pytest.mark.parametrize('rows, n_queries, n_context, exclude_self, message', [
    ([[1, 2], [0, 2], [0, 1]], 4, 3, False, 'must match n_queries'),
    ([[1, 2], [0, 2], [0, 1]], 3, 2, False, 'indices must be between'),
    ([[0, 2], [0, 2], [0, 1]], 3, 3, True, 'in its own context'),
    ([[0.5, 1.0]], 1, 3, False, 'integer array'),
])
def test_load_contexts_rejects_mismatches(saved, rows, n_queries, n_context, exclude_self, message):
    with pytest.raises(AssertionError, match=message):
        context.load_contexts(saved(rows), n_queries, n_context, exclude_self)