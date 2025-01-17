import torch
from torch_geometric.nn import GCNConv


class TGCN(torch.nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is True.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, improved: bool=False,
                 cached: bool=False, add_self_loops: bool=True):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()


    def _create_graph_conv_parameters_and_layers(self):

        self.conv_layer_1 = GCNConv(in_channels=self.in_channels,
                              out_channels=self.hidden_channels,
                              improved=self.improved,
                              cached=self.cached,
                              add_self_loops=self.add_self_loops)

        self.conv_layer_2 = GCNConv(in_channels=self.hidden_channels,
                                    out_channels=self.out_channels,
                                    improved=self.improved,
                                    cached=self.cached,
                                    add_self_loops=self.add_self_loops)


    def _create_update_gate_parameters_and_layers(self):

        self.linear_z = torch.nn.Linear(2*self.out_channels,
                                        self.out_channels)


    def _create_reset_gate_parameters_and_layers(self):

        self.linear_r = torch.nn.Linear(2*self.out_channels,
                                        self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.linear_h = torch.nn.Linear(2*self.out_channels,
                                        self.out_channels)


    def _create_parameters_and_layers(self):
        self._create_graph_conv_parameters_and_layers()
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()


    def _calculate_graph_conv(self, X, edge_index, edge_weight):
        GC = self.conv_layer_1(X, edge_index, edge_weight)
        GC = torch.relu(GC)
        GC = self.conv_layer_2(GC, edge_index, edge_weight)
        GC = torch.sigmoid(GC)
        return GC


    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H


    def _calculate_update_gate(self, X, H):
        # implementation of eq.4 (u is z)
        # u_t = \theta·(W_u[f(A, X_t), h_(t-1)] + bias_u)
        Z = torch.cat([X, H],axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z


    def _calculate_reset_gate(self, X, H):
        R = torch.cat([X, H],axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R


    def _calculate_candidate_state(self, X, H, R):
        H_tilde = torch.cat([X, H*R],axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde


    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z*H + (1-Z)*H_tilde
        return H


    def forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor=None, H: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        GC = self._calculate_graph_conv(X, edge_index, edge_weight)
        H = self._set_hidden_state(GC, H)
        Z = self._calculate_update_gate(GC, H)
        R = self._calculate_reset_gate(GC, H)
        H_tilde = self._calculate_candidate_state(GC, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
