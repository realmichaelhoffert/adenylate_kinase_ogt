import pytest

from tests.archive import *
from roseate import io, aggregate
from reference.verify_reference import Reference

# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def ref():
    return Reference(REF)

@pytest.fixture(scope="session")
def model():
    return io.reload_pickle(path(MODEL))

@pytest.fixture(scope="session")
def scaler():
    return io.reload_pickle(path(f"{SUPPORT}/ogt_scaler.pickle"))

@pytest.fixture(scope="session")
def archived_scaler():
    legacy_path = '/data/mhoffert/fiererlab/adenylate_kinase_ogt/data/'
    return io.reload_pickle(path(f"{legacy_path}predictive_models/supporting_data/ogt_scaler.pickle"))
    
@pytest.fixture(scope="session")
def even():
    return io.reload_pickle(path(f"{SUPPORT}/even_msa_dataset.pickle"))

@pytest.fixture(scope="session")
def corkrey():
    return io.reload_pickle(path(f"{SUPPORT}/corkrey_msa_dataset.pickle"))

@pytest.fixture(scope="session")
def full():
    return io.construct_msa_dataset(path(BACKBONE), 
                                    path(BACKBONE_TEMPS))

@pytest.fixture(scope="session")
def full_scaled_cell15(full):
    """05 cell 15: the full dataset scaled with a scaler fit on itself."""
    return io.scale_traits(full)

@pytest.fixture(scope="session")
def c_w_e():
    return io.reload_pickle(path(C_W_E))

@pytest.fixture(scope="session")
def spearmans():
    return aggregate.load_site_spearmans(path(SPEARMANS))