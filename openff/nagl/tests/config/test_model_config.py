import textwrap

from openff.nagl.config.model import ModelConfig


class TestModelConfig:
    def test_from_yaml(self, tmpdir):
        YAML = textwrap.dedent(
            """\
            version: '0.1'
            convolution:
              architecture: SAGEConv
              layers:
                - hidden_feature_size: 512
                  activation_function: ReLU
                  dropout: 0
                  aggregator_type: mean
                - hidden_feature_size: 512
                  activation_function: ReLU
                  dropout: 0
                  aggregator_type: mean
            readouts:
              charges:
                pooling: atoms
                postprocess: compute_partial_charges
                layers:
                  - hidden_feature_size: 128
                    activation_function: Sigmoid
                    dropout: 0
            atom_features:
                - name: atomic_element
                  categories: ["C", "H", "O", "N", "P", "S"]
                - name: atom_connectivity
                  categories: [1, 2, 3, 4, 5, 6]
                - name: atom_hybridization
                - name: atom_in_ring_of_size
                  ring_size: 3
                - name: atom_in_ring_of_size
                  ring_size: 4
                - name: atom_in_ring_of_size
                  ring_size: 5
                - name: atom_in_ring_of_size
                  ring_size: 6
            bond_features:
                - name: bond_is_in_ring
            """
        )
        with tmpdir.as_cwd():
            with open("model.yaml", "w") as file:
                file.write(YAML)
            model = ModelConfig.from_yaml("model.yaml")
            assert model.version == "0.1"
            assert model.convolution.architecture == "SAGEConv"
            assert len(model.convolution.layers) == 2
            assert model.readouts["charges"].pooling == "atoms"
            assert model.readouts["charges"].postprocess == "compute_partial_charges"
            assert len(model.readouts["charges"].layers) == 1
