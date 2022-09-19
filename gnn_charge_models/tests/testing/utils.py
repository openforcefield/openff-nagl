from openff.toolkit.topology.molecule import Molecule as OFFMolecule


def rdkit_molecule_to_smiles(rdkit_molecule):
    smiles = OFFMolecule.from_rdkit(
        rdkit_molecule,
        allow_undefined_stereo=True,
    ).to_smiles()
    return clean_smiles(smiles)


def clean_smiles(smiles, mapped=False):
    if mapped:
        func = OFFMolecule.from_mapped_smiles
    else:
        func = OFFMolecule.from_smiles
    return func(
        smiles,
        allow_undefined_stereo=True,
    ).to_smiles(mapped=mapped)
