from typing import List

from openff.toolkit.utils.toolkit_registry import (
    ToolkitRegistry as _ToolkitRegistry,
    ToolkitUnavailableException
)

from openff.nagl.utils.toolkits.base import ToolkitWrapperMeta, ToolkitWrapperType

class ToolkitRegistry(_ToolkitRegistry):

    def __init__(
        self,
        toolkit_wrappers: List[ToolkitWrapperType]

    ):
        self._toolkits = []

        toolkit_wrappers = [ToolkitWrapperMeta._get_object(wrapper) for wrapper in toolkit_wrappers]
        for toolkit in toolkit_wrappers:
            self.register_toolkit(toolkit)

    
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
        toolkit_wrapper = ToolkitWrapperMeta._get_class(toolkit_wrapper)
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
        toolkit_wrapper = ToolkitWrapperMeta._get_class(toolkit_wrapper)
        self._toolkits.append(toolkit_wrapper)