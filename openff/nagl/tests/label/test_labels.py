import pytest

import numpy as np

from openff.nagl.label.dataset import LabelledDataset
from openff.nagl.label.labels import (
    LabelConformers,
    LabelCharges,
    LabelMultipleDipoles,
    LabelMultipleESPs,
)
from openff.toolkit.utils import OPENEYE_AVAILABLE

pytest.importorskip("pyarrow")

@pytest.fixture
def dataset_with_conformers_and_charges(tmp_path):
    smiles = ["NCN", "[NH4+]", "[Cl-]"]
    dataset = LabelledDataset.from_smiles(
        tmp_path,
        smiles,
        mapped=False
    )
    conformers = [
        [
            [
                [ 0.74329, -1.68674,  0.91965],
                [ 0.5887 , -0.26471,  0.88179],
                [ 1.84291,  0.42191,  0.81687],
                [ 0.97947, -2.1323 ,  1.79494],
                [ 0.91666, -2.18114,  0.05611],
                [ 0.06335,  0.05231,  1.78599],
                [ 0.     ,  0.     ,  0.     ],
                [ 2.3784 ,  0.54315,  1.66482],
                [ 2.31718,  0.49023, -0.07242],
            ],
            [
                [ 0.74329, -1.68674,  0.91965],
                [ 0.5887 , -0.26471,  0.88179],
                [ 1.84291,  0.42191,  0.81687],
                [ 0.61331, -2.21802,  0.07036],
                [ 0.6752 , -2.16072,  1.80896],
                [ 0.06335,  0.05231,  1.78599],
                [ 0.     ,  0.     ,  0.     ],
                [ 2.3784 ,  0.54315,  1.66482],
                [ 2.31718,  0.49023, -0.07242],
            ],
            [
                [ 0.74329, -1.68674,  0.91965],
                [ 0.5887 , -0.26471,  0.88179],
                [ 1.84291,  0.42191,  0.81687],
                [ 0.61331, -2.21802,  0.07036],
                [ 0.6752 , -2.16072,  1.80896],
                [ 0.06335,  0.05231,  1.78599],
                [ 0.     ,  0.     ,  0.     ],
                [ 2.16427,  0.76974, -0.07544],
                [ 2.22509,  0.82245,  1.66182],
            ]
        ],
        [
            [
                [ 2.91824e-04,  6.14226e-05, -1.24454e-04],
                [ 4.29114e-02,  1.01246e+00, -1.66842e-01],
                [ 6.17639e-01, -4.83254e-01, -6.63131e-01],
                [-9.64223e-01, -3.27161e-01, -1.29666e-01],
                [ 3.03381e-01, -2.02105e-01,  9.59764e-01],
            ]
        ],
        [
            [
                [0., 0., 0.]
            ]
        ]
    ]

    flat_conformers = [np.concatenate(c).flatten() for c in conformers]
    n_conformers = [3, 1, 1]
    charges = [
        [
            -0.95369,  0.38078, -0.95369, 
            0.34833,  0.34833,  0.06664,
            0.06664,  0.34833,  0.34833,
        ],
        [-0.9132,  0.4783,  0.4783,  0.4783,  0.4783],
        [-1.0]
    ]

    dataset._append_columns({
        "conformers": flat_conformers,
        "n_conformers": n_conformers,
        "charges": charges,
    })
    return dataset

class TestLabelCharges:

    def test_label_with_conformers_on_fly(self, small_dataset):
        labellers = [
            LabelConformers(),
            LabelCharges(use_existing_conformers=True),
        ]
        small_dataset.apply_labellers(labellers)
        columns = ["mapped_smiles", "conformers", "n_conformers", "charges"]
        assert small_dataset.dataset.schema.names == columns

    def test_label_alkane_dataset(self):
        # test conformer generation and labelling
        # as in examples

        training_alkanes = [
            'C',
            'CC',
            'CCC',
            'CCCC',
            'CC(C)C',
            'CCCCC',
            'CC(C)CC',
            'CCCCCC',
            'CC(C)CCC',
            'CC(CC)CC',
        ]

        training_dataset = LabelledDataset.from_smiles(
            "training_data",
            training_alkanes,
            mapped=False,
            overwrite_existing=True,
        )
        training_df = training_dataset.to_pandas()
        assert training_df.mapped_smiles[0] in (
            "[H:2][C:1]([H:3])([H:4])[H:5]",
            "[C:1]([H:2])([H:3])([H:4])[H:5]"
        )

        label_conformers = LabelConformers(
            # create a new 'conformers' with output conformers
            conformer_column="conformers",
            # create a new 'n_conformers' with number of conformers
            n_conformer_column="n_conformers",
            n_conformer_pool=500, # initially generate 500 conformers
            n_conformers=10, # prune to max 10 conformers
            rms_cutoff=0.05,
        )

        label_am1_charges = LabelCharges(
            charge_method="am1-mulliken", # AM1
            # use previously generate conformers instead of new ones
            use_existing_conformers=True,
            # use the 'conformers' column as input for charge assignment
            conformer_column="conformers",
            # write generated charges to 'target-am1-charges' column
            charge_column="target-am1-charges",
        )

        labellers = [
            label_conformers, # generate initial conformers,
            label_am1_charges,
        ]

        training_dataset.apply_labellers(labellers)



class TestLabelMultipleDipoles:
    
    def test_apply_label(self, dataset_with_conformers_and_charges):
        columns = [
            "mapped_smiles", "conformers", "n_conformers", "charges",
        ]
        assert dataset_with_conformers_and_charges.dataset.schema.names == columns

        labellers = [
            LabelMultipleDipoles(),
        ]
        dataset_with_conformers_and_charges.apply_labellers(labellers)
        columns = [
            "mapped_smiles", "conformers", "n_conformers", "charges",
            "dipoles",
        ]
        assert dataset_with_conformers_and_charges.dataset.schema.names == columns

        expected_dipoles = [
            [
                [ 0.05804, -0.03359, -0.00185],
                [-0.15361, -0.05634,  0.00799],
                [-0.26028,  0.13831,  0.0059 ],
            ],
            [
                [-4.06074e-04, -8.54553e-05,  1.73207e-04]
            ],
            [
                [0., 0., 0.]
            ]
        ]

        flat_dipoles = [np.concatenate(d).flatten() for d in expected_dipoles]
        table = dataset_with_conformers_and_charges.dataset.to_table()
        calculated = table.to_pydict()["dipoles"]
        assert len(calculated) == len(flat_dipoles)
        for calc, exp in zip(calculated, flat_dipoles):
            np.testing.assert_allclose(calc, exp, rtol=1e-4, atol=1e-4)


class TestLabelMultipleESPs:
    
    def test_apply_label(self, dataset_with_conformers_and_charges):
        pytest.importorskip("openff.recharge")

        columns = [
            "mapped_smiles", "conformers", "n_conformers", "charges",
        ]
        assert dataset_with_conformers_and_charges.dataset.schema.names == columns

        labellers = [
            LabelMultipleESPs(),
        ]
        dataset_with_conformers_and_charges.apply_labellers(labellers)
        columns = [
            "mapped_smiles", "conformers", "n_conformers", "charges",
            "esp_lengths", "grid_inverse_distances", "esps",
        ]
        assert dataset_with_conformers_and_charges.dataset.schema.names == columns

        table = dataset_with_conformers_and_charges.dataset.to_table()
        pydict = table.to_pydict()
        expected_esp_lengths = [
            [542, 538, 541],
            [340],
            [384],
        ]
        calculated_esp_lengths = pydict["esp_lengths"]
        for calc, exp in zip(calculated_esp_lengths, expected_esp_lengths):
            assert calc == exp

        calculated_esps = pydict["esps"]
        for esps, lengths in zip(calculated_esps, calculated_esp_lengths):
            assert len(esps) == sum(lengths)
