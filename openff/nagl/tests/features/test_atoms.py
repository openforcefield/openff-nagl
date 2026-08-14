from openff.nagl.features import atoms

def test_atom_hybridization_categories_validation():
    assert atoms.AtomHybridization(categories=['OTHER', 'SP', 'SP2', 'SP3']).categories == [
        atoms.HybridizationType.OTHER,
        atoms.HybridizationType.SP,
        atoms.HybridizationType.SP2,
        atoms.HybridizationType.SP3,
    ]