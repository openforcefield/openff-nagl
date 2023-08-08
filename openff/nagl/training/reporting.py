import base64
import pathlib
import typing

import jinja2
import torch
import numpy as np
from openff.toolkit import Molecule

from openff.utilities import requires_package
import openff.nagl.training

if typing.TYPE_CHECKING:
    from openff.nagl.molecule._dgl import DGLMolecule
    from openff.nagl.training.metrics import MetricType


def _encode_image(image):
    image_encoded = base64.b64encode(image.encode()).decode()
    image_src = f"data:image/svg+xml;base64,{image_encoded}"
    return image_src

@requires_package("rdkit")
def _draw_molecule_with_atom_labels(
    molecule: Molecule,
    predicted_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    highlight_outliers: bool = False,
    outlier_threshold: float = 1.0,
) -> str:
    """
    Draw a molecule with predicted and reference atom labels.

    Parameters
    ----------
    molecule : Molecule
        The OpenFF molecule to draw.
    predicted_labels : torch.Tensor
        The predicted atom labels.
    reference_labels : torch.Tensor
        The reference atom labels.
    highlight_outliers : bool, optional
        Whether to highlight atoms with predicted labels that are more than
        ``outlier_threshold`` away from the reference labels.
    outlier_threshold : float, optional
        The threshold for highlighting outliers.
    
    Returns
    -------
    str
        The SVG image of the molecule, as text
    """
    from openff.nagl.molecule._dgl import DGLMolecule
    from rdkit.Chem import Draw

    if isinstance(molecule, DGLMolecule):
        molecule = molecule.to_openff()

    predicted_labels = predicted_labels.detach().numpy().flatten()
    reference_labels = reference_labels.detach().numpy().flatten()
    highlight_atoms = None
    if highlight_outliers:
        diff = np.abs(predicted_labels - reference_labels)
        highlight_atoms = list(np.where(diff > outlier_threshold)[0])
    
    predicted_molecule = molecule.to_rdkit()
    for atom, label in zip(predicted_molecule.GetAtoms(), predicted_labels):
        atom.SetProp("atomNote", f"{float(label):.3f}")
    
    reference_molecule = molecule.to_rdkit()
    for atom, label in zip(reference_molecule.GetAtoms(), reference_labels):
        atom.SetProp("atomNote", f"{float(label):.3f}")
    
    Draw.PrepareMolForDrawing(predicted_molecule)
    Draw.PrepareMolForDrawing(reference_molecule)

    draw_options = Draw.MolDrawOptions()
    draw_options.legendFontSize = 25

    image = Draw.MolsToGridImage(
        [predicted_molecule, reference_molecule],
        legends=["prediction", "reference"],
        molsPerRow=2,
        subImgSize=(400, 400),
        useSVG=True,
        drawOptions=draw_options,
        highlightAtomLists=[highlight_atoms, highlight_atoms],
    )
    return image


@requires_package("rdkit")
def _draw_molecule(
    molecule: typing.Union[Molecule, "DGLMolecule"],
) -> str:
    """
    Draw a molecule without labels.

    Parameters
    ----------
    molecule : typing.Union[Molecule, "DGLMolecule"]
        The molecule to draw.

    Returns
    -------
    str
        The SVG image of the molecule, as text
    """
    from rdkit.Chem import Draw

    from openff.nagl.molecule._dgl import DGLMolecule

    if isinstance(molecule, DGLMolecule):
        molecule = molecule.to_openff()
    
    rdmol = molecule.to_rdkit()
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400, 400)
    Draw.PrepareAndDrawMolecule(drawer, rdmol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _generate_jinja_dicts_per_atom(
    molecules: typing.List[Molecule],
    predicted_labels: typing.List[torch.Tensor],
    reference_labels: typing.List[torch.Tensor],
    metrics: typing.List["MetricType"],
    highlight_outliers: bool = False,
    outlier_threshold: float = 1.0,
) -> typing.List[typing.Dict[str, str]]:
    from openff.nagl.training.metrics import get_metric_type

    metrics = [get_metric_type(metric) for metric in metrics]
    jinja_dicts = []

    n_molecules = len(molecules)
    if n_molecules != len(predicted_labels):
        raise ValueError(
            "The number of molecules and predicted labels must match."
        )
    if n_molecules != len(reference_labels):
        raise ValueError(
            "The number of molecules and reference labels must match."
        )
    
    for molecule, predicted, reference in zip(
        molecules, predicted_labels, reference_labels
    ):
        entry_metrics = {
            metric.name.upper(): f"{metric.compute(predicted, reference):.4f}"
            for metric in metrics
        }
        image = _draw_molecule_with_atom_labels(
            molecule,
            predicted,
            reference,
            highlight_outliers=highlight_outliers,
            outlier_threshold=outlier_threshold,
        )
        jinja_dicts.append(
            {
                "img": _encode_image(image),
                "metrics": entry_metrics,
            }
        )
    return jinja_dicts


def _generate_jinja_dicts_per_molecule(
    molecules: typing.List[Molecule],
    metrics: typing.List[torch.Tensor],
    metric_name: str
) -> typing.List[typing.Dict[str, str]]:
    assert len(metrics) == len(molecules)

    jinja_dicts = []
    for molecule, metric in zip(molecules, metrics):
        image = _draw_molecule(molecule)
        data = {
            "img": _encode_image(image),
            "metrics": {
                metric_name.upper(): f"{float(metric):.4f}"
            }
        }
        jinja_dicts.append(data)
    return jinja_dicts
    

def _write_jinja_report(
    output_path: pathlib.Path,
    top_n_structures: typing.List[typing.Dict[str, str]],
    bottom_n_structures: typing.List[typing.Dict[str, str]],
):
    output_path = pathlib.Path(output_path)
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("openff.nagl.training"),
    )
    template = env.get_template("jinja_report.html")
    rendered = template.render(
        top_n_structures=top_n_structures,
        bottom_n_structures=bottom_n_structures,
    )
    output_path = pathlib.Path(output_path)
    output_path.write_text(rendered)



def create_atom_label_report(
    molecules: typing.List[Molecule],
    predicted_labels: typing.List[torch.Tensor],
    reference_labels: typing.List[torch.Tensor],
    metrics: typing.List["MetricType"],
    rank_by: "MetricType",
    output_path: pathlib.Path,
    top_n_entries: int = 100,
    bottom_n_entries: int = 100,
    highlight_outliers: bool = False,
    outlier_threshold: float = 1.0,
):
    from openff.nagl.training.metrics import get_metric_type

    ranker = get_metric_type(rank_by)
    metrics = [get_metric_type(metric) for metric in metrics]
    
    n_molecules = len(molecules)
    if n_molecules != len(predicted_labels):
        raise ValueError(
            "The number of molecules and predicted labels must match."
        )
    if n_molecules != len(reference_labels):
        raise ValueError(
            "The number of molecules and reference labels must match."
        )

    entries_and_ranks = []
    for molecule, predicted, reference in zip(
        molecules, predicted_labels, reference_labels
    ):
        diff = ranker.compute(predicted, reference)
        entries_and_ranks.append((molecule, predicted, reference, diff))
    
    entries_and_ranks.sort(key=lambda x: x[-1], reverse=True)
    top_molecules, top_predicted, top_reference, _ = zip(
        *entries_and_ranks
    )


    top_jinja_dicts = _generate_jinja_dicts_per_atom(
        top_molecules[:top_n_entries],
        top_predicted[:top_n_entries],
        top_reference[:top_n_entries],
        metrics,
        highlight_outliers=highlight_outliers,
        outlier_threshold=outlier_threshold,
    )

    bottom_jinja_dicts = _generate_jinja_dicts_per_atom(
        top_molecules[-bottom_n_entries:],
        top_predicted[-bottom_n_entries:],
        top_reference[-bottom_n_entries:],
        metrics,
        highlight_outliers=highlight_outliers,
        outlier_threshold=outlier_threshold,
    )

    _write_jinja_report(
        output_path,
        top_jinja_dicts,
        bottom_jinja_dicts,
    )

    


def create_molecule_label_report(
    molecules: typing.List[Molecule],
    losses: typing.List[torch.Tensor],
    metric_name: str,
    output_path: pathlib.Path,
    top_n_entries: int = 100,
    bottom_n_entries: int = 100,
):
    assert len(molecules) == len(losses)

    entries = sorted(zip(molecules, losses), key=lambda x: x[-1])
    molecules_, losses_ = zip(*entries)
    top_n_entries = _generate_jinja_dicts_per_molecule(
        molecules_[:top_n_entries],
        losses_[:top_n_entries],
        metric_name,
    )
    bottom_n_entries = _generate_jinja_dicts_per_molecule(
        molecules_[-bottom_n_entries:],
        losses_[-bottom_n_entries:],
        metric_name,
    )
    _write_jinja_report(
        output_path,
        top_n_entries,
        bottom_n_entries,
    )