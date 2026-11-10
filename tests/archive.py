# Functions for working with archived run

import os
import glob
import pytest
from pathlib import Path

REF = Path(os.environ.get("ROSEATE_REFERENCE", "reference/v1_preprint"))
pytestmark = pytest.mark.skipif(not (REF / "MANIFEST.tsv").exists(),
                                reason=f"reference archive not found at {REF}")

# archive-relative locations
SUPPORT = "inputs/predictive_models/supporting_data"
MODEL = "outputs/predictive_models/even_enetcv_ind.pickle"
SPEARMANS = "outputs/predictive_models/supporting_data/even_enetcv_ind_spearmans.tsv*"
BACKBONE = "inputs/high_qual_alignments/backbone.filtered_only.afa"
BACKBONE_TEMPS = 'inputs/high_qual_alignments/temp_data.tsv'
C_W_E = "intermediates/esm_msa/c_w_e_full_embeddings.pickle"
VALIDATION = sorted(Path(p).parts[-3] for p in
                    glob.glob(str(REF / "intermediates/validation_datasets/*/embeddings/*_full_even.pickle")))

requires_archive = pytest.mark.skipif(not (REF / "MANIFEST.tsv").exists(),
                                     reason=f"reference archive not found at {REF}")

def path(rel):
    """Archive file for a relative path or glob; skip the test if it is not archived."""
    hits = sorted(glob.glob(str(REF / rel)))
    if not hits:
        pytest.skip(f"not in archive: {rel}")
    return Path(hits[0])

