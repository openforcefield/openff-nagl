from openff.nagl import GNNModel, __version__
from openff.nagl_models import list_available_nagl_models

from openff.toolkit import Molecule, ForceField
from packaging.version import Version

def test_versions_sane():
    # partially safeguard against a malformed version number;
    # 0.minor.patch and major.0.patch should be okay but not 0.0.patch
    found_version = Version(__version__)
    assert not (found_version.minor == 0 and found_version.major == 0), found_version

    print(f"found version {found_version}")

def test_find_and_load_models():

    assert len(list_available_nagl_models()) > 0

    [GNNModel.load(model) for model in list_available_nagl_models()]

def test_assign_ashgc():
    molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    
    molecule.assign_partial_charges(
        partial_charge_method="openff-gnn-am1bcc-1.0.0.pt",
    )
    
def test_build_interchange():
    molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    
    ForceField("openff-2.3.0.offxml").create_interchange(molecule.to_topology())
