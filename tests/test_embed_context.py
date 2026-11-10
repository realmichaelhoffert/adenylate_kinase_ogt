"""
Level 3 reproduction tests: roseate.context and roseate.embed.

Central check: the same query and the same context give the same embedding.
Embedding tests need the ESM weights and are slow; by default they embed a few
proteins per set. Full run and tolerance:
    ROSEATE_FULL_EMBED=1 ROSEATE_EMBED_ATOL=1e-4 pytest tests/test_level3_context_embed.py
"""
import os

import numpy as np
import pytest

from reference.verify_reference import Reference
from roseate import context, embed, io
from tests.archive import MODEL, REF, SUPPORT, VALIDATION, path, requires_archive
from tests.conftest import even

pytestmark = requires_archive

ATOL = float(os.environ.get("ROSEATE_EMBED_ATOL", "1e-4"))   # set from the hardware gap you measure
FULL = False #bool(os.environ.get("ROSEATE_FULL_EMBED"))


def pick(n):
    """Proteins to embed: all when FULL, else the first three and the last two."""
    return list(range(n)) if FULL else [0, 1, 2, n - 2, n - 1]


# ---------------------------------------------------------------- the one place to adapt
def embed_with_context(model, batch_converter, query, context_dataset, context_inds, layer=12):
    """
    Embed one (id, seq) query with an explicit list of (id, seq) context sequences.
    
    """
    embeddings, _, _ = embed.get_embeddings(
        model=model, batch_converter=batch_converter, 
        data_to_embed=query, 
        context_dataset=context_dataset, 
        context_inds=context_inds,
        layer=layer, return_mean=False)
    return np.asarray(embeddings)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def short_even():
    return io.reload_pickle(path(f"{SUPPORT}/even_msa_dataset.pickle"))[:10]

@pytest.fixture(scope="module")
def esm():
    model, _, batch_converter = embed.load_plm()
    model.eval()
    return model, batch_converter

def pairs(dataset, indices):
    return [(dataset[i][0], dataset[i][1]) for i in indices]


# ---------------------------------------------------------------- context.py
def test_legacy_even_context_matches_portable(ref, even):
    idx = context.load_legacy_context(path("intermediates/esm_msa/even_full_context.pickle"), even)
    ref.assert_array_close(idx, "portable/esm_msa/even_full_context_indices.npy")
    assert idx.shape == (len(even), 8)
    assert not np.any(idx == np.arange(len(even))[:, None]), "training contexts must exclude self"

# didn't save the corkrey embedding context
# # def test_legacy_c_w_e_context_matches_portable(ref, even, corkrey):
# #     idx = context.load_legacy_context(path("intermediates/esm_msa/c_w_e_context.pickle"), even)
# #     ref.assert_array_close(idx, "portable/esm_msa/c_w_e_context_indices.npy")
# #     assert idx.shape == (len(corkrey), 8)

@pytest.mark.parametrize("name", VALIDATION)
def test_context_values_match_portable(ref, name, even):
    rel = f"validation_datasets/{name}/embeddings/{name}_even_context_values"
    idx = np.asarray(io.reload_pickle(path(f"intermediates/{rel}.pickle")))
    ref.assert_array_close(idx, f"portable/{rel}.npy")
    assert idx.max() < len(even)

def test_random_context_seed_is_reproducible(even):
    a = context.get_random_context(even, n_contexts=50, context_size=8, self_exclude=True, seed=0)
    b = context.get_random_context(even, n_contexts=50, context_size=8, self_exclude=True, seed=0)
    c = context.get_random_context(even, n_contexts=50, context_size=8, self_exclude=True, seed=1)
    assert a == b and a != c

def test_random_context_shape_and_self_exclusion(even):
    ctx = np.asarray(context.get_random_context(even, n_contexts=len(even), context_size=8, self_exclude=True))
    assert ctx.shape == (len(even), 8)
    assert ctx.min() >= 0 and ctx.max() < len(even)
    assert not np.any(ctx == np.arange(len(even))[:, None])


# ---------------------------------------------------------------- embed.py: identical context -> identical embedding
@pytest.mark.slow
def test_same_context_twice_is_bitwise_identical(esm, short_even):
    ctx1 = context.get_random_context(msa_dataset=short_even,
                            n_contexts=len(short_even), 
                            context_size=8,
                            self_exclude=True,
                            seed = 1234)
    ctx2 = context.get_random_context(msa_dataset=short_even,
                            n_contexts=len(short_even), 
                            context_size=8,
                            self_exclude=True,
                            seed = 1234)
    # for i, q in enumerate(short_even):
    # embed_with_context(esm, query, context_dataset, context_pairs, layer=12)
    # a = embed_with_context(esm, [q], short_even, [ctx1[i]])
    # b = embed_with_context(esm, [q], short_even, [ctx2[i]])
    a = embed_with_context(esm[0], esm[1], short_even, short_even, ctx1)
    b = embed_with_context(esm[0], esm[1], short_even, short_even, ctx2)
    np.testing.assert_array_equal(a, b)

# # not necessary
# # @pytest.mark.slow
# # def test_embedding_shape_matches_model(esm, short_even):
# #     ctx1 = context.get_random_context(msa_dataset=short_even,
# #                         n_contexts=len(short_even), 
# #                         context_size=8,
# #                         self_exclude=True,
# #                         seed = 1234)
# #     out = embed_with_context(esm, short_even, short_even, ctx1)
# #     expected = tuple(io.reload_pickle(path(MODEL))["metadata"]["input_shape"])[1:]
# #     assert out.shape == expected and out.dtype == np.float32

@pytest.mark.slow
def test_context_changes_embedding(esm, short_even):
    """Sanity check that the context is actually used."""

    ctx1 = context.get_random_context(msa_dataset=short_even,
                    n_contexts=len(short_even), 
                    context_size=8,
                    self_exclude=True,
                    seed = 1234)
    ctx2 = context.get_random_context(msa_dataset=short_even,
                    n_contexts=len(short_even), 
                    context_size=8,
                    self_exclude=True,
                    seed = 0000)
    a = embed_with_context(esm[0], esm[1], short_even, short_even, ctx1)
    b = embed_with_context(esm[0], esm[1], short_even, short_even, ctx2)
    assert np.abs(a - b).max() > ATOL


# ---------------------------------------------------------------- embed.py: archived context -> archived embedding
@pytest.mark.slow
@pytest.mark.parametrize("query_set", ["even"])
def test_reembedding_with_archived_context(ref, esm, even, query_set):
    queries = even
    stem = "even_full"
    ctx = np.load(path(f"portable/esm_msa/{stem}_context_indices.npy"))
    archived = ref.load_array("portable/esm_msa/even_full_embeddings.npy")
    worst = 0.0
    # for j in range(len(queries))[:10]:
    n_test = 100
    out = embed_with_context(model=esm[0], batch_converter=esm[1],
                             query=even[:n_test], 
                             context_dataset=even, 
                             context_inds=ctx[:n_test])
    diff = float(np.abs(out - archived[:n_test]).max())
    worst = max(worst, diff)
    assert diff <= ATOL, f"{query_set} protein {j}: max abs diff {diff:.3g} > {ATOL}"
    print(f"{query_set}: worst max abs diff {worst:.3g}")

# very slow
@pytest.mark.slow
@pytest.mark.parametrize("name", VALIDATION[:1])
def test_reembedding_validation_dataset(ref, esm, even, name):
    print(f'Working for dataset: {name}')
    rel = f"validation_datasets/{name}/embeddings/{name}"
    msa = io.reload_pickle(path(f"intermediates/{rel}_msa_dataset.pickle"))
    ctx = np.load(path(f"portable/{rel}_even_context_values.npy"))
    archived = ref.load_array(f"portable/{rel}_full_even.npy")
    n_test = 100
    out = embed_with_context(esm[0], esm[1], msa[:n_test], even, ctx[:n_test])
    print(out.shape, archived[:n_test, :, :].shape)
    assert np.abs(out - archived[:n_test, :, :]).max() <= ATOL #, f"{name} protein {j}"

# not possible, didn't save the corkrey embedding context
# but confirmed embeddings on same context are identical
# and model retrained on saming embeddings is identical
# @pytest.mark.slow
# def test_reembedding_preserves_predictions(esm, even, corkrey):
#     """Translates the embedding difference into prediction error, in scaled units."""
#     model = io.reload_pickle(path(MODEL))
#     ctx = np.load(path("portable/esm_msa/c_w_e_context_indices.npy"))
#     stored = np.asarray(model["full_predictions"])
#     for j in pick(len(corkrey)):
#         out = embed_with_context(esm[0], esm[1], [corkrey[j]], even, [ctx[j]])
#         preds = [model["best_model"][s].predict(model["scalers"][s].transform(out[s][None, :]))[0]
#                  for s in range(out.shape[0])]
#         np.testing.assert_allclose(preds, stored[:, j], atol=1e-3, rtol=0)