from openff.nagl.app.trainer import Trainer
from openff.nagl.data.files import EXAMPLE_MODEL_CONFIG

def test_trainer(tmpdir):

    with tmpdir.as_cwd():
        trainer = Trainer.from_yaml_file(EXAMPLE_MODEL_CONFIG)
        assert len(trainer.atom_features) == 8
        assert len(trainer.bond_features) == 4
        
        assert type(trainer.atom_features[0]).__name__ == "AtomicElement"
        assert trainer.atom_features[0].categories == ["C", "O", "H", "N", "S", "F", "Br", "Cl", "I", "P"]

        trainer.to_yaml_file("output.yaml")
        trainer2 = Trainer.from_yaml_file("output.yaml")

        assert trainer.to_simple_dict() == trainer2.to_simple_dict()
        assert len(trainer.to_simple_hash()) == 64

