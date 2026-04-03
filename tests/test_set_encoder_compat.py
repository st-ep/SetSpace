from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from set_encoders import SetEncoderOperator, WeightedSetEncoder


class SetEncoderCompatTests(unittest.TestCase):
    def test_weighted_set_encoder_default_value_coords_matches_old_path(self):
        torch.manual_seed(0)
        encoder = WeightedSetEncoder(
            n_tokens=4,
            coord_dim=3,
            value_input_dim=2,
            output_dim=5,
            key_dim=6,
            value_dim=7,
            hidden_dim=8,
            activation_fn=nn.ReLU,
            value_mode="mlp_xu",
        )
        coords = torch.randn(2, 6, 3)
        values = torch.randn(2, 6, 2)

        out_default = encoder(coords, values)
        out_explicit = encoder(coords, values, value_coords=coords)

        self.assertTrue(torch.allclose(out_default, out_explicit, atol=1e-6, rtol=1e-6))

    def test_set_encoder_operator_default_value_xs_matches_old_path(self):
        torch.manual_seed(1)
        operator = SetEncoderOperator(
            input_size_src=3,
            output_size_src=1,
            input_size_tgt=3,
            output_size_tgt=1,
            p=4,
            rho_hidden_size=16,
            trunk_hidden_size=16,
            n_trunk_layers=3,
            activation_fn=nn.ReLU,
            value_mode="mlp_xu",
            use_positional_encoding=True,
            pos_encoding_dim=12,
        )
        xs = torch.randn(2, 5, 3)
        us = torch.randn(2, 5, 1)
        ys = torch.randn(2, 7, 3)

        encoded_xs = operator._sinusoidal_encoding(xs)
        out_default = operator(xs, us, ys)
        out_explicit = operator(xs, us, ys, value_xs=encoded_xs)

        self.assertTrue(torch.allclose(out_default, out_explicit, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
