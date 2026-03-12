from set_encoders.encoders import WeightedSetEncoder


class QuadratureHead(WeightedSetEncoder):
    def __init__(
        self,
        *,
        p: int,
        dx_enc: int,
        du: int,
        dout: int,
        dk: int,
        dv: int,
        hidden: int,
        activation_fn,
        key_hidden: int | None = None,
        key_layers: int = 3,
        phi_activation: str = "tanh",
        value_mode: str = "linear_u",
        eps: float = 1e-8,
        normalize: str = "total",
        learn_temperature: bool = False,
    ) -> None:
        super().__init__(
            n_tokens=p,
            coord_dim=dx_enc,
            value_input_dim=du,
            output_dim=dout,
            key_dim=dk,
            value_dim=dv,
            hidden_dim=hidden,
            activation_fn=activation_fn,
            key_hidden_dim=key_hidden,
            key_layers=key_layers,
            basis_activation=phi_activation,
            value_mode=value_mode,
            eps=eps,
            normalize=normalize,
            learn_temperature=learn_temperature,
        )

__all__ = ["QuadratureHead"]
