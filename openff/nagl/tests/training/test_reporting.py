import pathlib
import pytest

import torch
from openff.toolkit import Molecule

from openff.nagl.molecule._dgl.molecule import DGLMolecule
from openff.nagl.training.reporting import (
    _draw_molecule,
    _draw_molecule_with_atom_labels,
    _generate_jinja_dicts_per_atom,
    _generate_jinja_dicts_per_molecule,
    create_atom_label_report,
    create_molecule_label_report,
)

pytest.importorskip("dgl")
pytest.importorskip("rdkit")

@pytest.fixture()
def hbr():
    return Molecule.from_smiles("Br")

@pytest.fixture()
def dgl_hbr(hbr):
    return DGLMolecule.from_openff(
        hbr,
        atom_features=[],
        bond_features=[],
    )

def test_draw_molecule_with_atom_labels():
    mol = Molecule.from_smiles("[Cl-]")
    svg = _draw_molecule_with_atom_labels(
        mol, torch.tensor([1.0]), torch.tensor([0.0])
    )

    assert "svg" in svg


def test_generate_jinja_dicts_per_atom(hbr, dgl_hbr):
    tensor = torch.tensor([1.0, 0.0])

    jinja_dicts = _generate_jinja_dicts_per_atom(
        molecules=[hbr,  dgl_hbr],
        predicted_labels=[tensor, tensor],
        reference_labels=[tensor, tensor],
        metrics=["rmse"],
        highlight_outliers=True,
        outlier_threshold=0.5,
    )

    assert len(jinja_dicts) == 2
    for item in jinja_dicts:
        assert "img" in item
        assert "metrics" in item
        assert "RMSE" in item["metrics"]


def test_create_atom_label_report(tmpdir, hbr, dgl_hbr):
    tensor = torch.tensor([1.0, 0.0])

    with tmpdir.as_cwd():
        create_atom_label_report(
            molecules=[hbr, dgl_hbr],
            predicted_labels=[tensor, tensor],
            reference_labels=[tensor, tensor],
            metrics=["rmse"],
            rank_by="rmse",
            output_path="report.html",
            highlight_outliers=True,
            outlier_threshold=0.5,
            top_n_entries=2,
            bottom_n_entries=1,
        )

        output = pathlib.Path("report.html")
        assert output.exists()

        contents = output.read_text()
        assert "Top 2 Structures" in contents
        assert "Bottom 1 Structures" in contents


def test_create_molecule_label_report(tmpdir, hbr, dgl_hbr):
    with tmpdir.as_cwd():
        create_molecule_label_report(
            molecules=[hbr, dgl_hbr],
            losses=[torch.tensor([1.0]), torch.tensor([0.0])],
            metric_name="rmse",
            output_path="report.html",
            top_n_entries=1,
            bottom_n_entries=2,
        )

        output = pathlib.Path("report.html")
        assert output.exists()

        contents = output.read_text()
        assert "Top 1 Structures" in contents
        assert "Bottom 2 Structures" in contents