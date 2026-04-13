import collections
import logging
import types
from typing import TYPE_CHECKING, Tuple, Dict, Union, Callable, Literal, Optional
import warnings

import torch
import pytorch_lightning as pl

from openff.utilities.exceptions import MissingOptionalDependencyError
from openff.nagl.nn._containers import ConvolutionModule, ReadoutModule
from openff.nagl.config.model import ModelConfig
from openff.nagl.domains import ChemicalDomain
from openff.nagl.lookups import LookupTableType, _as_lookup_table
from openff.nagl.utils._utils import potential_dict_to_list

logger = logging.getLogger(__name__)

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

        readouts: Dict[str, torch.Tensor] = {}
        for readout_type, readout_module in self.readout_modules.items():
            readouts[readout_type] = readout_module(
                molecule,
                model=self,
                readouts=readouts,
            )
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
    """
    A GNN model for predicting properties of molecules.

    Parameters
    ----------
    config: ModelConfig or dict
        The configuration for the model.

    chemical_domain: ChemicalDomain or dict
        The applicable chemical domain for the model.

    lookup_tables: dict
        A dictionary of lookup tables for properties.
        The keys should be the property names, and the values
        should be instances of :class:`~openff.nagl.lookups.BaseLookupTable`.
    """
    def __init__(
        self,
        config: ModelConfig,
        chemical_domain: Optional[ChemicalDomain] = None,
        lookup_tables: dict[str, LookupTableType] = None,
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

        valid_lookup_tables = {}
        if not lookup_tables:
            lookup_tables = {}
        # allow an iterable of lookup tables
        lookup_tables = potential_dict_to_list(lookup_tables)
        for lookup_table in lookup_tables:
            lookup_table = _as_lookup_table(lookup_table)
            if not lookup_table.property_name in readout_modules:
                raise ValueError(
                    f"The lookup table property name {lookup_table.property_name} "
                    f"is not in the readout modules."
                )
            valid_lookup_tables[lookup_table.property_name] = lookup_table
            

        super().__init__(
            convolution_module=convolution_module,
            readout_modules=readout_modules,
        )

        lookup_tables_dict = {}
        for k, v in valid_lookup_tables.items():
            v_ = v.dict()
            v_["properties"] = dict(v_["properties"])
            lookup_tables_dict[k] = v_

        self.save_hyperparameters({
            "config": config.dict(),
            "chemical_domain": chemical_domain.dict(),
            "lookup_tables": lookup_tables_dict,
        })
        self.config = config
        self.chemical_domain = chemical_domain
        self.lookup_tables = types.MappingProxyType(valid_lookup_tables)
        

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
        check_lookup_table: bool = True,
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
        check_lookup_table: bool
            Whether to check a lookup table for the property values.
            If ``False`` or if the molecule is not in the lookup
            table, the property will be computed using the model.

        Returns
        -------
        result: Dict[str, torch.Tensor] or Dict[str, numpy.ndarray]
        """
        import numpy as np

        # split up molecule in case it's fragments
        from openff.nagl.toolkits.openff import split_up_molecule
        
        fragments, all_indices = split_up_molecule(molecule)
        # TODO: this assumes atom-wise properties
        # we should add support for bond-wise/more general properties

        results = [
            self._compute_properties(
                fragment,
                as_numpy=as_numpy,
                check_domains=check_domains,
                error_if_unsupported=error_if_unsupported,
                check_lookup_table=check_lookup_table,
            )
            for fragment in fragments
        ]

        # combine the results
        combined_results = {}

        if as_numpy:
            tensor = np.empty
        else:
            tensor = torch.empty
        
        for property_name in results[0].keys():
            n_values = sum(len(result[property_name]) for result in results)
            combined_results[property_name] = tensor(n_values)

        seen_indices = collections.defaultdict(set)
        
        for i, (result, indices) in enumerate(zip(results, all_indices)):
            for property_name, value in result.items():
                j = 0
                if self.readout_modules[property_name].pooling_layer.name == "atom":
                    combined_results[property_name][indices] = value
                    if seen_indices[property_name] & set(indices):
                        raise ValueError(
                            "Overlapping indices in the fragments"
                        )
                    seen_indices[property_name].update(indices)
                else:
                    warnings.warn(
                        "TODO: currently non-atom-wise properties "
                        "are not properly handled!!! "
                        "We just assume they are **strictly sequential**. "
                        "In general we don't recommend using multiple molecules!!"
                    )
                    combined_results[property_name][j : j+ len(value)] = value
                    j += len(value)
                
                

        expected_indices = list(range(molecule.n_atoms))
        for property_name, seen_indices in seen_indices.items():
            if self.readout_modules[property_name].pooling_layer.name == "atom":
                assert sorted(seen_indices) == expected_indices, (
                    f"Missing indices for property {property_name}: "
                    f"{set(expected_indices) - seen_indices}"
                )
        return combined_results


    
    def _compute_properties(
        self,
        molecule: "Molecule",
        as_numpy: bool = True,
        check_domains: bool = False,
        error_if_unsupported: bool = True,
        check_lookup_table: bool = True,
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
        check_lookup_table: bool
            Whether to check a lookup table for the property values.
            If ``False`` or if the molecule is not in the lookup
            table, the property will be computed using the model.

        Returns
        -------
        result: Dict[str, torch.Tensor] or Dict[str, numpy.ndarray]
        """

        values = {}

        expected_value_keys = list(self.readout_modules.keys())

        if check_lookup_table and self.lookup_tables:
            for property_name in expected_value_keys:
                try:
                    value = self._check_property_lookup_table(
                        molecule=molecule,
                        readout_name=property_name,
                    )
                except KeyError as e:
                    logger.info(
                        f"Could not find property in lookup table: {e}"
                    )
                    continue
                else:
                    logger.info(
                        f"Using lookup table for property {property_name}"
                    )
                    values[property_name] = value

        
        computed_value_keys = set(values.keys())
        if computed_value_keys == set(expected_value_keys):
            if as_numpy:
                values = {k: v.detach().numpy().flatten() for k, v in values.items()}
            return values
        
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
        except (MissingOptionalDependencyError, TypeError) as e:
            raise e
            values = self._compute_properties_nagl(molecule)
        

        if as_numpy:
            values = {k: v.detach().numpy().flatten() for k, v in values.items()}
        return values
    
    def _check_property_lookup_table(
        self,
        molecule: "Molecule",
        readout_name: str,
    ):
        """
        Check if the molecule is in the property lookup table.

        Parameters
        ----------
        molecule: :class:`~openff.toolkit.topology.Molecule`
            The molecule to check.
        readout_name: str
            The name of the readout to check.
        
        Returns
        -------
        torch.Tensor
        

        Raises
        ------
        KeyError
            If there is no table for this property, or
            if the molecule is not in the property lookup table
        """

        table = self.lookup_tables[readout_name]
        return table.lookup(molecule)

        
    def compute_property(
        self,
        molecule: "Molecule",
        readout_name: Optional[str] = None,
        as_numpy: bool = True,
        check_domains: bool = False,
        error_if_unsupported: bool = True,
        check_lookup_table: bool = True
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
        check_lookup_table: bool
            Whether to check a lookup table for the property values.
            If ``False`` or if the molecule is not in the lookup
            table, the property will be computed using the model.

        Returns
        -------
        result: torch.Tensor or numpy.ndarray
        """
        properties = self.compute_properties(
            molecule=molecule,
            as_numpy=as_numpy,
            check_domains=check_domains,
            error_if_unsupported=error_if_unsupported,
            check_lookup_table=check_lookup_table
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

        nxmol = GraphMolecule.from_openff_config(
            molecule,
            self.config,
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

        dglmol = DGLMolecule.from_openff_config(
            molecule,
            self.config,
            model=self,
        )
        return self.forward(dglmol)
    
    def _convert_to_nagl_molecule(self, molecule: "Molecule"):
        from openff.nagl.molecule._graph.molecule import GraphMolecule
        if self._is_dgl:
            from openff.nagl.molecule._dgl.molecule import DGLMolecule

            return DGLMolecule.from_openff_config(
                molecule,
                self.config,
                model=self
            )

        return GraphMolecule.from_openff_config(
            molecule,
            self.config,
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
        model_kwargs = torch.load(str(model), weights_only=False, **kwargs)
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


