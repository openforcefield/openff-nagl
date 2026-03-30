"""Defining the NAGLToolkitRegistry class."""

from typing import List, Optional

from openff.toolkit.utils.toolkit_registry import (
    ToolkitRegistry as _ToolkitRegistry,
    ToolkitUnavailableException,
)

from openff.nagl.toolkits._base import (
    NAGLToolkitWrapperMeta,
    ToolkitWrapperType,
)


class NAGLToolkitRegistry(_ToolkitRegistry):
    def __init__(
        self,
        toolkit_precedence: Optional[List[ToolkitWrapperType]] = None,
        exception_if_unavailable: bool = True,
        _register_imported_toolkit_wrappers: bool = False,
    ):
        self._toolkits = []
        toolkits_to_register = []

        if _register_imported_toolkit_wrappers:
            if toolkit_precedence is None:
                toolkit_precedence = ["openeye", "rdkit"]
            for toolkit in toolkit_precedence:
                try:
                    toolkit_class = NAGLToolkitWrapperMeta._get_class(toolkit)
                except (KeyError, ValueError):
                    pass
                else:
                    toolkits_to_register.append(toolkit_class)

        else:
            if toolkit_precedence is not None:
                toolkits_to_register = toolkit_precedence

        if toolkits_to_register:
            for toolkit in toolkits_to_register:
                self.register_toolkit(
                    toolkit, exception_if_unavailable=exception_if_unavailable
                )

    @classmethod
    def _resolve_registry(cls, toolkit_registry: _ToolkitRegistry | None) -> "NAGLToolkitRegistry":
        if toolkit_registry is None:
            from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY
            toolkit_registry = GLOBAL_TOOLKIT_REGISTRY
        if isinstance(toolkit_registry, NAGLToolkitRegistry):
            return toolkit_registry
        elif isinstance(toolkit_registry, _ToolkitRegistry):
            return cls.from_openff_toolkit_registry(toolkit_registry)
        else:
            raise ValueError(
                f"toolkit_registry must be an instance of NAGLToolkitRegistry, ToolkitRegistry, or None. Got {type(toolkit_registry)}"
            )
        
    @classmethod
    def from_openff_toolkit_registry(cls, toolkit_registry: _ToolkitRegistry) -> "NAGLToolkitRegistry":
        """
        Convert an openff.toolkit.utils.ToolkitRegistry to a NAGLToolkitRegistry

        Parameters
        ----------
        toolkit_registry : openff.toolkit.utils.ToolkitRegistry
            The ToolkitRegistry to convert

        Returns
        -------
        NAGLToolkitRegistry
            A NAGLToolkitRegistry with the same toolkits as the input registry
        """
        from openff.toolkit.utils import OpenEyeToolkitWrapper, RDKitToolkitWrapper
        from openff.nagl.toolkits.openeye import NAGLOpenEyeToolkitWrapper
        from openff.nagl.toolkits.rdkit import NAGLRDKitToolkitWrapper

        _COUNTERPARTS = {
            NAGLRDKitToolkitWrapper: NAGLRDKitToolkitWrapper,
            RDKitToolkitWrapper: NAGLRDKitToolkitWrapper,
            NAGLOpenEyeToolkitWrapper: NAGLOpenEyeToolkitWrapper,
            OpenEyeToolkitWrapper: NAGLOpenEyeToolkitWrapper,
        }

        # build new registry from scratch
        new_nagl_registry = NAGLToolkitRegistry(exception_if_unavailable=False)
        for toolkit_wrapper in toolkit_registry.registered_toolkits:
            if type(toolkit_wrapper) in _COUNTERPARTS:
                nagl_toolkit_wrapper_class = _COUNTERPARTS[type(toolkit_wrapper)]
                new_nagl_registry.register_toolkit(
                    nagl_toolkit_wrapper_class,
                    exception_if_unavailable=False,
                )
        return new_nagl_registry

    def deregister_toolkit(self, toolkit_wrapper: ToolkitWrapperType):
        """
        Remove a ToolkitWrapper from the list of toolkits in this ToolkitRegistry

        .. warning :: This API is experimental and subject to change.

        Parameters
        ----------
        toolkit_wrapper : instance or subclass of ToolkitWrapper
            The toolkit wrapper to remove from the registry

        Raises
        ------
        InvalidToolkitError
            If toolkit_wrapper is not a ToolkitWrapper or subclass
        ToolkitUnavailableException
            If toolkit_wrapper is not found in the registry
        """
        toolkit_wrapper = NAGLToolkitWrapperMeta._get_class(toolkit_wrapper)
        toolkits_to_remove = []

        for toolkit in self._toolkits:
            if type(toolkit) == toolkit_wrapper:
                toolkits_to_remove.append(toolkit)

        if not toolkits_to_remove:
            msg = (
                f"Did not find {toolkit_wrapper.name} in registry. "
                f"Currently registered toolkits are {self._toolkits}"
            )
            raise ToolkitUnavailableException(msg)

        for toolkit_to_remove in toolkits_to_remove:
            self._toolkits.remove(toolkit_to_remove)

    def add_toolkit(self, toolkit_wrapper: ToolkitWrapperType):
        """
        Append a ToolkitWrapper onto the list of toolkits in this ToolkitRegistry

        .. warning :: This API is experimental and subject to change.

        Parameters
        ----------
        toolkit_wrapper : openff.toolkit.utils.ToolkitWrapper
            The ToolkitWrapper object to add to the list of registered toolkits

        Raises
        ------
        InvalidToolkitError
            If toolkit_wrapper is not a ToolkitWrapper or subclass
        """
        toolkit_wrapper = NAGLToolkitWrapperMeta._get_class(toolkit_wrapper)
        self._toolkits.append(toolkit_wrapper)
