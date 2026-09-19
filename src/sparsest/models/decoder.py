#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Decoder stack of :class:`~sparsest.models.cell.SparseST` cells."""

from torch import nn


class Decoder(nn.Module):
    """Mirror of the encoder: cells run top-down, layer N first and layer 1 last.

    ``lstms`` is given in execution order, so the first entry is registered as
    ``lstmN`` and the last as ``lstm1``, matching the encoder's naming.
    """

    def __init__(self, lstms):
        super().__init__()
        self.blocks = len(lstms)
        for index, lstm in enumerate(lstms):
            setattr(self, "lstm" + str(self.blocks - index), lstm)

    def forward_by_layer(self, sum_l0, sum_l1, inputs, state, lstm):
        return lstm(sum_l0, sum_l1, inputs, state, delta_m=None)

    def forward(self, sum_l0, sum_l1, hidden_states):
        # Deepest layer is driven purely by the encoder's final state, with no input
        # sequence of its own.
        sum_l0, sum_l1, inputs, _ = self.forward_by_layer(
            sum_l0, sum_l1, None, hidden_states[-1], getattr(self, f"lstm{self.blocks}")
        )

        # Remaining layers each take the layer above's output plus their counterpart
        # encoder state.
        for i in list(range(2, self.blocks))[::-1]:
            sum_l0, sum_l1, inputs, _ = self.forward_by_layer(
                sum_l0, sum_l1, inputs, hidden_states[i - 1], getattr(self, "lstm" + str(i))
            )

        # The output layer starts from a fresh state.
        sum_l0, sum_l1, outputs, _ = self.forward_by_layer(
            sum_l0, sum_l1, inputs, None, getattr(self, "lstm1")
        )
        outputs = outputs.transpose(0, 1)  # S,B,C,H,W -> B,S,C,H,W
        return sum_l0, sum_l1, outputs
