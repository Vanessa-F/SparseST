"""Encoder-decoder wrapper that normalises the accumulated sparsity metrics."""

from torch import nn


class ED(nn.Module):
    """Join an encoder and decoder and average their sparsity accumulators.

    Every cell adds one term to ``sum_l0``/``sum_l1`` per timestep, so the totals grow
    with both the number of cells and the unroll length. Dividing by ``n_cells * window``
    turns them back into per-cell-per-timestep means, which is what the multi-objective
    loss and the reported occupancy expect.
    """

    def __init__(self, encoder, decoder, window):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.window = window
        self.n_cells = encoder.blocks + decoder.blocks
        self.normalizer = self.n_cells * window

    def forward(self, sum_l0, sum_l1, input):
        sum_l0, sum_l1, last_hiddens = self.encoder(sum_l0, sum_l1, input)
        sum_l0, sum_l1, output = self.decoder(sum_l0, sum_l1, last_hiddens)
        return sum_l0 / self.normalizer, sum_l1 / self.normalizer, output
