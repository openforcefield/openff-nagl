import copy
from typing import TYPE_CHECKING, Tuple, Dict, Union, Callable, Literal, Optional
import warnings

import torch
import pytorch_lightning as pl

from openff.utilities.exceptions import MissingOptionalDependencyError
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule
from openff.nagl.config.model import ModelConfig
from openff.nagl.domains import ChemicalDomain

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.nagl.molecule._dgl import DGLMoleculeOrBatch


class BaseGNNModel(pl.LightningModule):
    def __init__(
        self,
        convolution_module: ConvolutionModule,
        readout_modules: ReadoutModule,
    ):
        super().__init__()
        self.convolution_module = convolution_module
        self.readout_modules = torch.nn.ModuleDict(readout_modules)

    def forward(
        self, molecule: "DGLMoleculeOrBatch"
    ) -> Dict[str, torch.Tensor]:
        self.convolution_module(molecule)

        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }
        return readouts

    def _forward_unpostprocessed(self, molecule: "DGLMoleculeOrBatch"):
        """
        Forward pass without postprocessing the readout modules.
        This is quality-of-life method for debugging and testing.
        It is *not* intended for public use.
        """
        self.convolution_module(molecule)
        readouts: Dict[str, torch.Tensor] = {
            readout_type: readout_module._forward_unpostprocessed(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }
        return readouts


class GNNModel(BaseGNNModel):
    def __init__(
        self,
        config: ModelConfig,
        chemical_domain: Optional[ChemicalDomain] = None,
    ):
        if not isinstance(config, ModelConfig):
            config = ModelConfig(**config)

        if chemical_domain is None:
            chemical_domain = ChemicalDomain(
                allowed_elements=tuple(),
                forbidden_patterns=tuple(),
            )
        elif not isinstance(chemical_domain, ChemicalDomain):
            chemical_domain = ChemicalDomain(**chemical_domain)

        convolution_module = ConvolutionModule.from_config(
            config.convolution,
            n_input_features=config.n_atom_features,
        )

        readout_modules = {}
        for readout_name, readout_config in config.readouts.items():
            readout_modules[readout_name] = ReadoutModule.from_config(
                readout_config,
                n_input_features=config.convolution.layers[-1].hidden_feature_size,
            )

        super().__init__(
            convolution_module=convolution_module,
            readout_modules=readout_modules,
        )

        self.save_hyperparameters({
            "config": config.dict(),
            "chemical_domain": chemical_domain.dict(),
        })
        self.config = config
        self.chemical_domain = chemical_domain
        

    @classmethod
    def from_yaml(cls, filename):
        config = ModelConfig.from_yaml(filename)
        return cls(config)
    
    @property
    def _is_dgl(self):
        return self.convolution_module._is_dgl
    
    def _as_nagl(self):
        copied = type(self)(self.config)
        copied.convolution_module = self.convolution_module._as_nagl(copy_weights=True)
        copied.load_state_dict(self.state_dict())
        return copied
    
    def compute_properties(
        self,
        molecule: "Molecule",
        as_numpy: bool = True,
        check_domains: bool = False,
        error_if_unsupported: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the trained property for a molecule.

        Parameters
        ----------
        molecule: :class:`~openff.toolkit.topology.Molecule`
            The molecule to compute the property for.
        as_numpy: bool
            Whether to return the result as a numpy array.
            If ``False``, the result will be a ``torch.Tensor``.
        check_domains: bool
            Whether to check if the molecule is similar
            to the training data.
        error_if_unsupported: bool
            Whether to raise an error if the molecule
            is not represented in the training data.
            This is only used if ``check_domains`` is ``True``.
            If ``False``, a warning will be raised instead.

        Returns
        -------
        result: Dict[str, torch.Tensor] or Dict[str, numpy.ndarray]
        """
        if check_domains:
            is_supported, error = self.chemical_domain.check_molecule(
                molecule, return_error_message=True
            )
            if not is_supported:
                if error_if_unsupported:
                    raise ValueError(error)
                else:
                    warnings.warn(error)

        try:
            values = self._compute_properties_dgl(molecule)
        except (MissingOptionalDependencyError, TypeError):
            values = self._compute_properties_nagl(molecule)
        if as_numpy:
            values = {k: v.detach().numpy().flatten() for k, v in values.items()}
        return values
        
    def compute_property(
        self,
        molecule: "Molecule",
        readout_name: Optional[str] = None,
        as_numpy: bool = True,
        check_domains: bool = False,
        error_if_unsupported: bool = True,
    ):
        """
        Compute the trained property for a molecule.

        Parameters
        ----------
        molecule: :class:`~openff.toolkit.topology.Molecule`
            The molecule to compute the property for.
        readout_name: str
            The name of the readout property to return.
            If this is not given and there is only one readout,
            the result of that readout will be returned.
        as_numpy: bool
            Whether to return the result as a numpy array.
            If ``False``, the result will be a ``torch.Tensor``.
        check_domains: bool
            Whether to check if the molecule is similar
            to the training data.
        error_if_unsupported: bool
            Whether to raise an error if the molecule
            is not represented in the training data.
            This is only used if ``check_domains`` is ``True``.
            If ``False``, a warning will be raised instead.

        Returns
        -------
        result: torch.Tensor or numpy.ndarray
        """
        properties = self.compute_properties(
            molecule=molecule,
            as_numpy=as_numpy,
            check_domains=check_domains,
            error_if_unsupported=error_if_unsupported,
        )
        if readout_name is None:
            if len(properties) == 1:
                return next(iter(properties.values()))
            raise ValueError(
                "The readout name must be specified if the model has multiple readouts"
            )
        return properties[readout_name]

    def _compute_properties_nagl(self, molecule: "Molecule") -> "torch.Tensor":
        from openff.nagl.molecule._graph.molecule import GraphMolecule

        nxmol = GraphMolecule.from_openff(
            molecule,
            atom_features=self.config.atom_features,
            bond_features=self.config.bond_features,
        )
        model = self
        if self._is_dgl:
            model = self._as_nagl()
        return model.forward(nxmol)

    def _compute_properties_dgl(self, molecule: "Molecule") -> "torch.Tensor":
        from openff.nagl.molecule._dgl.molecule import DGLMolecule

        if not self._is_dgl:
            raise TypeError(
                "This model is not a DGL-based model "
                 "and cannot be used to compute properties with the DGL backend"
            )

        dglmol = DGLMolecule.from_openff(
            molecule,
            atom_features=self.config.atom_features,
            bond_features=self.config.bond_features,
        )
        return self.forward(dglmol)
    
    def _convert_to_nagl_molecule(self, molecule: "Molecule"):
        from openff.nagl.molecule._graph.molecule import GraphMolecule
        if self._is_dgl:
            from openff.nagl.molecule._dgl.molecule import DGLMolecule

            return DGLMolecule.from_openff(
                molecule,
                atom_features=self.config.atom_features,
                bond_features=self.config.bond_features,
            )

        return GraphMolecule.from_openff(
            molecule,
            atom_features=self.config.atom_features,
            bond_features=self.config.bond_features,
        )
    
    @classmethod
    def load(cls, model: str, eval_mode: bool = True, **kwargs):
        """
        Load a model from a file.

        Parameters
        ----------
        model: str
            The path to the model to load.
            This should be a file containing a dictionary of
            hyperparameters and a state dictionary,
            with the keys "hyperparameters" and "state_dict".
            This can be created using the `save` method.
        eval_mode: bool
            Whether to set the model to evaluation mode.
        **kwargs
            Additional keyword arguments to pass to `torch.load`.

        Returns
        -------
        model: GNNModel

        Examples
        --------

        >>> model.save("model.pt")
        >>> new_model = GNNModel.load("model.pt")

        Notes
        -----
        This method is not compatible with normal Pytorch
        models saved with ``torch.save``, as it expects
        a dictionary of hyperparameters and a state dictionary.
        """
        model_kwargs = torch.load(str(model), **kwargs)
        if isinstance(model_kwargs, dict):
            model = cls(**model_kwargs["hyperparameters"])
            model.load_state_dict(model_kwargs["state_dict"])
        elif isinstance(model_kwargs, cls):
            model = model_kwargs
        else:
            raise ValueError(f"Unknown model type {type(model_kwargs)}")
        if eval_mode:
            model.eval()

        return model

    def save(self, path: str):
        """
        Save this model to a file.

        Parameters
        ----------
        path: str
            The path to save this file to.

        Examples
        --------

        >>> model.save("model.pt")
        >>> new_model = GNNModel.load("model.pt")

        Notes
        -----
        This method writes a dictionary of the hyperparameters and the state dictionary,
        with the keys "hyperparameters" and "state_dict".
        """
        torch.save(
            {
                "hyperparameters": self.hparams,
                "state_dict": self.state_dict(),
            },
            str(path),
        )


