from collections import UserDict

from typing import Dict


import torch

class FrameDict(UserDict):

    def subframe(self, indices):
        return type(self)((key, value[indices]) for key, value in self.items())



class EdgeBatch:
    def __init__(self, graph, eid, etype, src_data, edge_data, dst_data):
        self._graph = graph
        self._eid = eid
        self._etype = etype
        self._src_data = src_data
        self._edge_data = edge_data
        self._dst_data = dst_data

    @classmethod
    def _all_from_networkx_molecule(cls, nxmolecule):
        eid = "__ALL__"
        etype = ('_N', '_E', '_N')
        source, destination, edge_ids = nxmolecule._all_edges()
        src_data = nxmolecule._node_data(source)
        dst_data = nxmolecule._node_data(destination)
        edge_data = nxmolecule._edge_data(edge_ids)
        return cls(nxmolecule, eid, etype, src_data, edge_data, dst_data)


class NodeBatch:
    def __init__(self, graph, nodes: torch.Tensor, ntype: str, data: Dict[str, torch.Tensor], msgs=None):
        self._graph = graph
        self._nodes = nodes
        self._ntype = ntype
        self._data = data
        self._msgs = msgs

    @property
    def data(self):
        """Return a view of the node features for the nodes in the batch.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received and
        >>> # the original feature for each node.
        >>> def node_udf(nodes):
        >>>     # nodes.data['h'] is a tensor of shape (N, 1),
        >>>     # nodes.mailbox['m'] is a tensor of shape (N, D, 1),
        >>>     # where N is the number of nodes in the batch, D is the number
        >>>     # of messages received per node for this node batch.
        >>>     return {'h': nodes.data['h'] + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[2.],
                [3.]])
        """
        return self._data

    @property
    def mailbox(self):
        """Return a view of the messages received.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received and
        >>> # the original feature for each node.
        >>> def node_udf(nodes):
        >>>     # nodes.data['h'] is a tensor of shape (N, 1),
        >>>     # nodes.mailbox['m'] is a tensor of shape (N, D, 1),
        >>>     # where N is the number of nodes in the batch, D is the number
        >>>     # of messages received per node for this node batch.
        >>>     return {'h': nodes.data['h'] + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[2.],
                [3.]])
        """
        return self._msgs

    def nodes(self):
        """Return the nodes in the batch.

        Returns
        -------
        NID : Tensor
            The IDs of the nodes in the batch. :math:`NID[i]` gives the ID of
            the i-th node.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph and set a feature 'h'.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received and
        >>> # the original ID for each node.
        >>> def node_udf(nodes):
        >>>     # nodes.nodes() is a tensor of shape (N),
        >>>     # nodes.mailbox['m'] is a tensor of shape (N, D, 1),
        >>>     # where N is the number of nodes in the batch, D is the number
        >>>     # of messages received per node for this node batch.
        >>>     return {'h': nodes.nodes().unsqueeze(-1).float()
        >>>         + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[1.],
                [3.]])
        """
        return self._nodes

    def batch_size(self):
        """Return the number of nodes in the batch.

        Returns
        -------
        int

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> # Instantiate a graph.
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
        >>> g.ndata['h'] = torch.ones(2, 1)

        >>> # Define a UDF that computes the sum of the messages received for
        >>> # each node and increments the result by 1.
        >>> def node_udf(nodes):
        >>>     return {'h': torch.ones(nodes.batch_size(), 1)
        >>>         + nodes.mailbox['m'].sum(1)}

        >>> # Use node UDF in message passing.
        >>> import dgl.function as fn
        >>> g.update_all(fn.copy_u('h', 'm'), node_udf)
        >>> g.ndata['h']
        tensor([[2.],
                [3.]])
        """
        return len(self._nodes)

    def __len__(self):
        """Return the number of nodes in this node batch.

        Returns
        -------
        int
        """
        return self.batch_size()

    @property
    def ntype(self):
        """Return the node type of this node batch, if available."""
        return self._ntype