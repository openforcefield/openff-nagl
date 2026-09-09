
def rdkit_molecule_to_smiles(rdkit_molecule):
    from openff.toolkit.topology.molecule import Molecule

    smiles = Molecule.from_rdkit(
        rdkit_molecule,
        allow_undefined_stereo=True,
    ).to_smiles()
    return clean_smiles(smiles)


def clean_smiles(smiles, mapped=False):
    from openff.toolkit.topology.molecule import Molecule

    if mapped:
        func = Molecule.from_mapped_smiles
    else:
        func = Molecule.from_smiles
    return func(
        smiles,
        allow_undefined_stereo=True,
    ).to_smiles(mapped=mapped)


def assert_smiles_equal(smiles1: str, smiles2: str, mapped=False):
    smiles1 = clean_smiles(smiles1, mapped=mapped)
    smiles2 = clean_smiles(smiles2, mapped=mapped)
    assert smiles1 == smiles2
