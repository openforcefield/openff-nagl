from typing import List, Optional

from openff.toolkit.utils.toolkit_registry import (
    ToolkitRegistry as _ToolkitRegistry,
    ToolkitUnavailableException
)

from openff.nagl.utils.toolkits.base import NAGLToolkitWrapperMeta, NAGLToolkitWrapperBase, ToolkitWrapperType

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

