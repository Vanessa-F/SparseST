#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Encoder stack of :class:`~sparsest.models.cell.SparseST` cells."""

from torch import nn


class Encoder(nn.Module):
    """Run the cells bottom-up, collecting each layer's final ``(h, c)``.

    Layers are registered as ``lstm1 .. lstmN`` so that state-dict keys match the
    checkpoints produced by earlier revisions of this model.
    """

    def __init__(self, lstms):
        super().__init__()
        self.blocks = len(lstms)
        for index, lstm in enumerate(lstms, 1):
            setattr(self, "lstm" + str(index), lstm)

    def forward_by_layer(self, sum_l0, sum_l1, inputs, lstm):
        return lstm(sum_l0, sum_l1, inputs, None, None)

    def forward(self, sum_l0, sum_l1, inputs):
        inputs = inputs.transpose(0, 1)  # B,S,C,H,W -> S,B,C,H,W

        # Final state of each layer; the decoder picks these up in reverse.
        last_hiddens = []
        for i in range(1, self.blocks + 1):
            sum_l0, sum_l1, inputs, last_hidden = self.forward_by_layer(
                sum_l0, sum_l1, inputs, getattr(self, "lstm" + str(i))
            )
            last_hiddens.append(last_hidden)

        return sum_l0, sum_l1, last_hiddens
