from openff.nagl._base.base import MutableModel
from openff.units import unit
import numpy as np
import json
import textwrap

from pydantic import field_validator

class TestMutableModel:

    class Model(MutableModel):
        int_type: int
        float_type: float
        list_type: list
        np_array_type: np.ndarray
        tuple_type: tuple
        unit_type: unit.Quantity

        @field_validator("np_array_type", mode="before")
        @classmethod
        def _validate_np_array_type(cls, v):
            return np.asarray(v)


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
        json_dict = json.loads(model.model_dump_json())
        expected = {
            "int_type": 1,
            "float_type": 1.0,
            "list_type": [1, 2, 3],
            "np_array_type": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9] ],
            "tuple_type": [1, 2, 3],
            "unit_type": {"magnitude": 1.0, "units": "angstrom"}
        }
        assert json_dict == expected

    def test_from_json_string(self):
        input_text = """
        {
            "int_type": 4,
            "float_type": 10.0,
            "list_type": [1, 2, 3],
            "np_array_type": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9] ],
            "tuple_type": [1, 2, 3],
            "unit_type": {"magnitude": 1.0, "units": "angstrom"}
        }
        """
        model = self.Model.from_json(input_text)
        assert model.int_type == 4
        assert model.float_type == 10.0
        assert model.list_type == [1, 2, 3]
        arr = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        assert np.array_equal(model.np_array_type, arr)
        assert model.tuple_type == (1, 2, 3)
        assert model.unit_type == unit.Quantity(1.0, "angstrom")

    def test_from_json_file(self, tmp_path):
        input_text = """
        {
            "int_type": 4,
            "float_type": 10.0,
            "list_type": [1, 2, 3],
            "np_array_type": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9] ],
            "tuple_type": [1, 2, 3],
            "unit_type": {"magnitude": 1.0, "units": "angstrom"}
        }
        """
        file_path = tmp_path / "test.json"
        with open(file_path, "w") as f:
            f.write(input_text)
        model = self.Model.from_json(file_path)
        assert model.int_type == 4
        assert model.float_type == 10.0
        assert model.list_type == [1, 2, 3]
        arr = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        assert np.array_equal(model.np_array_type, arr)
        assert model.tuple_type == (1, 2, 3)
        assert model.unit_type == unit.Quantity(1.0, "angstrom")

    def test_to_yaml(self, tmp_path):
        model = self.Model(int_type=1, float_type=1.0, list_type=[1, 2, 3], np_array_type=np.array([1, 2, 3]), tuple_type=(1, 2, 3), unit_type=unit.Quantity(1.0, "angstrom"))
        file_path = tmp_path / "test.yaml"
        model.to_yaml(file_path)
        with open(file_path, "r") as f:
            yaml_text = f.read()
        expected = textwrap.dedent("""
            float_type: 1.0
            int_type: 1
            list_type:
            - 1
            - 2
            - 3
            np_array_type:
            - 1
            - 2
            - 3
            tuple_type:
            - 1
            - 2
            - 3
            unit_type:
              magnitude: 1.0
              units: angstrom
            """)
        assert yaml_text.strip() == expected.strip()
    
    def test_from_yaml(self, tmp_path):
        input_text = textwrap.dedent("""
            float_type: 1.0
            int_type: 1
            list_type:
            - 1
            - 2
            - 3
            np_array_type:
            - 1
            - 2
            - 3
            tuple_type:
            - 1
            - 2
            - 3
            unit_type:
              magnitude: 1.0
              units: angstrom
            """)
        file_path = tmp_path / "test.yaml"
        with open(file_path, "w") as f:
            f.write(input_text)
        model = self.Model.from_yaml(file_path)
        assert model.int_type == 1
        assert model.float_type == 1.0
        assert model.list_type == [1, 2, 3]
        assert np.array_equal(model.np_array_type, np.array([1, 2, 3]))
        assert model.tuple_type == (1, 2, 3)
        assert model.unit_type == unit.Quantity(1.0, "angstrom")
