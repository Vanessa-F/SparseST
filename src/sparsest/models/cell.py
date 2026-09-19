"""Delta-thresholded sparse ConvLSTM cell.

The cell keeps a memory of the previous input and hidden state and only feeds the
*change* since that memory into a sparse convolution. Changes whose magnitude falls
below a learnable threshold are zeroed, which is what makes the convolution input
sparse. Those thresholds are the decision variables the multi-objective search
optimises: raising them buys sparsity at the cost of reconstruction error.

One threshold pair is learned per timestep, so ``threshold_x`` and ``threshold_h``
are vectors of length ``window``.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv


def signed_shrink(delta, threshold):
    """Zero out sub-threshold changes and shrink the rest toward zero.

    ``relu(d - t) - relu(-d - t)`` leaves ``0`` wherever ``|d| <= t`` and otherwise
    returns ``d`` reduced in magnitude by ``t``, keeping its sign. Unlike a ``where``
    mask it is differentiable in ``threshold``, which is what lets the search learn it.
    """
    return torch.relu(delta - threshold) - torch.relu(-delta - threshold)


class SparseST(nn.Module):
    """A ConvLSTM cell whose gate convolutions run on thresholded deltas.

    Args:
        window: number of timesteps unrolled; also the number of learned thresholds.
        shape: spatial size ``(H, W)`` of the feature maps.
        batch_size: fixed batch size. State buffers are allocated once at this size,
            so loaders must use ``drop_last=True``.
        input_channels: channels of the cell input.
        filter_size: square kernel size of both gate convolutions.
        num_features: hidden channels produced by the cell.
        debug_unit: when true, print per-timestep sparsity and threshold values.
    """

    def __init__(
        self,
        window,
        shape,
        batch_size,
        input_channels,
        filter_size,
        num_features,
        debug_unit=False,
    ):
        super().__init__()

        self.threshold_x = nn.Parameter(torch.zeros(window), requires_grad=True)
        self.threshold_h = nn.Parameter(torch.zeros(window), requires_grad=True)

        self.window = window
        self.shape = shape
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.debug_unit = debug_unit

        self.spconv_x = spconv.SparseSequential(
            spconv.SparseConv2d(
                in_channels=input_channels,
                out_channels=4 * num_features,
                kernel_size=filter_size,
                stride=1,
                padding=self.padding,
            ),
            spconv.ToDense(),
        )

        self.spconv_h = spconv.SparseSequential(
            spconv.SparseConv2d(
                in_channels=num_features,
                out_channels=4 * num_features,
                kernel_size=filter_size,
                stride=1,
                padding=self.padding,
            ),
            spconv.ToDense(),
        )

    def _zeros(self, channels, device):
        return torch.zeros(
            self.batch_size, channels, self.shape[0], self.shape[1], device=device
        )

    def clamp_thresholds(self):
        """Clamp thresholds to be non-negative.

        A negative threshold would *widen* the delta rather than shrink it, inverting
        the sparsity objective. This runs outside the autograd graph so it is safe on
        leaf parameters.
        """
        with torch.no_grad():
            self.threshold_x.clamp_(min=0.0)
            self.threshold_h.clamp_(min=0.0)

    def calculate_unit_sparsity(self, delta_input, delta_hidden):
        """Return ``(l0_sparsity_pair, l1_occupancy)`` for one timestep.

        ``l1_occupancy`` is the mean absolute magnitude that survives thresholding.
        ``l0_sparsity`` is the fraction of spatial positions that are zero after
        summing over channels.
        """
        l1_occupancy = (
            torch.mean(torch.abs(delta_input)) + torch.mean(torch.abs(delta_hidden))
        ) / 2

        input_projection = torch.sum(delta_input, dim=1)
        hidden_projection = torch.sum(delta_hidden, dim=1)

        l0_sparsity_input = (
            input_projection.numel() - torch.count_nonzero(input_projection)
        ).float() / input_projection.numel()
        l0_sparsity_hidden = (
            hidden_projection.numel() - torch.count_nonzero(hidden_projection)
        ).float() / hidden_projection.numel()

        return [l0_sparsity_input, l0_sparsity_hidden], l1_occupancy

    def _sparse_conv(self, branch, tensor, device):
        """Run a sparse convolution, short-circuiting when the delta is all zeros.

        ``SparseConvTensor.from_dense`` cannot represent an empty tensor, so a fully
        thresholded delta is handled directly.
        """
        if torch.count_nonzero(tensor) == 0:
            return self._zeros(4 * self.num_features, device)
        sparse = spconv.SparseConvTensor.from_dense(tensor.permute(0, 2, 3, 1))
        return branch(sparse)

    def forward(self, sum_l0, sum_l1, inputs=None, hidden_state=None, delta_m=None):
        """Unroll the cell over ``window`` timesteps.

        Args:
            sum_l0, sum_l1: running sparsity accumulators, threaded through the whole
                encoder/decoder stack and normalised once in :class:`~sparsest.models.ed.ED`.
            inputs: ``(S, B, C, H, W)`` sequence, or ``None`` for decoder layers that
                are driven purely by the hidden state.
            hidden_state: ``(h, c)`` handed over from the encoder, or ``None`` to start
                from zeros.
            delta_m: per-gate delta memory, or ``None`` to start from zeros.

        Returns:
            ``(sum_l0, sum_l1, outputs, (h, c))`` where ``outputs`` is ``(S, B, F, H, W)``.
        """
        device = self.threshold_x.device

        if hidden_state is None:
            h = self._zeros(self.num_features, device)
            c = self._zeros(self.num_features, device)
        else:
            h, c = hidden_state

        h_prev = self._zeros(self.num_features, device)
        x_prev = self._zeros(self.input_channels, device)

        output_inner = []

        for index in range(self.window):

            if inputs is None:
                x = self._zeros(self.input_channels, device)
            else:
                x = inputs[index, ...]

            delta_x_before = x - x_prev
            delta_h_before = h - h_prev

            threshold_x = self.threshold_x[index]
            threshold_h = self.threshold_h[index]

            delta_x_after = signed_shrink(delta_x_before, threshold_x)
            delta_h_after = signed_shrink(delta_h_before, threshold_h)

            # Refresh the memory only where the change cleared the threshold.
            x_prev = torch.where(torch.abs(delta_x_before) > threshold_x, x, x_prev)
            h_prev = torch.where(torch.abs(delta_h_before) > threshold_h, h, h_prev)

            res_x = self._sparse_conv(self.spconv_x, delta_x_after, device)
            res_h = self._sparse_conv(self.spconv_h, delta_h_after, device)

            pre_ingate, pre_forgetgate, pre_cellgate, pre_outgate = torch.split(
                res_x + res_h, self.num_features, dim=1
            )

            if delta_m is None:
                zeros = self._zeros(self.num_features, device)
                input_m = forget_m = cell_m = out_m = zeros
            else:
                input_m, forget_m, cell_m, out_m = delta_m

            ingate = torch.sigmoid(pre_ingate + input_m)
            forgetgate = torch.sigmoid(pre_forgetgate + forget_m)
            cellgate = torch.tanh(pre_cellgate + cell_m)
            outgate = torch.sigmoid(pre_outgate + out_m)

            ct = (forgetgate * c) + (ingate * cellgate)
            ht = outgate * torch.tanh(ct)

            # Because the convolutions see deltas rather than absolute values, each
            # gate carries its pre-activation forward as the baseline for the next step.
            delta_m = (pre_ingate, pre_forgetgate, pre_cellgate, pre_outgate)

            output_inner.append(ht)
            h, c = ht, ct

            l0_sparsity, l1_occupancy = self.calculate_unit_sparsity(
                delta_x_after, delta_h_after
            )
            sum_l0 = sum_l0 + torch.mean(torch.stack(l0_sparsity))
            sum_l1 = sum_l1 + l1_occupancy

            if self.debug_unit:
                print(
                    f"{index} {l0_sparsity[0].item():.6f} {l0_sparsity[1].item():.6f} "
                    f"{threshold_x.item():.6f} {threshold_h.item():.6f}"
                )

        return sum_l0, sum_l1, torch.stack(output_inner), (ht, ct)
