import numpy as np
import pytest

from openff.nagl.storage.db import DBConformerRecord, match_conformers
from openff.nagl.storage.record import ConformerRecord


def test_match_conformers():

    matches = match_conformers(
        "[Cl:1][H:2]",
        db_conformers=[
            DBConformerRecord(
                coordinates=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            ),
            DBConformerRecord(
                coordinates=np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
            ),
        ],
        query_conformers=[
            ConformerRecord(
                coordinates=np.array([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]]),
                partial_charges=[],
                bond_orders=[],
            ),
            ConformerRecord(
                coordinates=np.array([[0.0, -2.0, 0.0], [0.0, 3.0, 0.0]]),
                partial_charges=[],
                bond_orders=[],
            ),
            ConformerRecord(
                coordinates=np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
                partial_charges=[],
                bond_orders=[],
            ),
        ],
    )

    assert matches == {0: 1, 2: 0}
