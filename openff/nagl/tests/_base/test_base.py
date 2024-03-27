from openff.nagl._base.base import MutableModel
from openff.units import unit
import numpy as np
import json


class TestMutableModel:

    class Model(MutableModel):
        int_type: int
        float_type: float
        list_type: list
        np_array_type: np.ndarray
        tuple_type: tuple
        unit_type: unit.Quantity

    def test_init(self):
        model = self.Model(int_type=1, float_type=1.0, list_type=[1, 2, 3], np_array_type=np.array([1, 2, 3]), tuple_type=(1, 2, 3), unit_type=unit.Quantity(1.0, "angstrom"))
        assert model.int_type == 1
        assert model.float_type == 1.0
        assert model.list_type == [1, 2, 3]
        assert np.array_equal(model.np_array_type, np.array([1, 2, 3]))
        assert model.tuple_type == (1, 2, 3)
        assert model.unit_type == unit.Quantity(1.0, "angstrom")
    
    def test_to_json(self):
        arr = np.arange(10).reshape(2, 5)
        model = self.Model(int_type=1, float_type=1.0, list_type=[1, 2, 3], np_array_type=arr, tuple_type=(1, 2, 3), unit_type=unit.Quantity(1.0, "angstrom"))
        json_dict = json.loads(model.to_json())
        expected = {
            "int_type": 1,
            "float_type": 1.0,
            "list_type": [1, 2, 3],
            "np_array_type": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9] ],
            "tuple_type": [1, 2, 3],
            "unit_type": [1.0, [["angstrom", 1]]]
        }
        assert json_dict == expected

    