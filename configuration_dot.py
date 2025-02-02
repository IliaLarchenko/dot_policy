from dataclasses import dataclass


@dataclass
class DOTConfig:
    # Input / output structure.
    n_obs_steps: int = 2
    train_horizon: int = 100
    inference_horizon: int = 100
    lookback_obs_steps: int = 30
    lookback_aug: int = 5

    input_shapes: dict[str, list[int]] = None
    output_shapes: dict[str, list[int]] = None

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = None
    output_normalization_modes: dict[str, str] = None

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pre_norm: bool = False
    lora_rank: int = 20
    merge_lora: bool = True

    dim_model: int = 128
    n_heads: int = 8
    dim_feedforward: int = 512
    n_decoder_layers: int = 8
    rescale_shape: tuple[int, int] = (96, 96)

    # Augmentation.
    crop_scale: float = 1.0
    state_noise: float = 0.01
    noise_decay: float = 0.999995

    # Training and loss computation.
    dropout: float = 0.1

    # Weighting and inference.
    alpha: float = 0.7
    train_alpha: float = 0.7
    predict_every_n: int = 1
    return_every_n: int = 1

    def __post_init__(self):
        if self.predict_every_n > self.inference_horizon:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to horizon ({self.inference_horizon})."
            )
        if self.return_every_n > self.inference_horizon:
            raise ValueError(
                f"return_every_n ({self.return_every_n}) must be less than or equal to horizon ({self.inference_horizon})."
            )
        if self.predict_every_n > self.inference_horizon // self.return_every_n:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be less than or equal to horizon //  return_every_n({self.inference_horizon // self.return_every_n})."
            )
        if self.train_horizon < self.inference_horizon:
            raise ValueError(
                f"train_horizon ({self.train_horizon}) must be greater than or equal to horizon ({self.inference_horizon})."
            )
        if (
            not any(k.startswith("observation.image") for k in self.input_shapes)
            and "observation.environment_state" not in self.input_shapes
        ):
            raise ValueError(
                "You must provide at least one image or environment state among the inputs."
            )
        if self.input_shapes is None:
            raise ValueError("You must provide input_shapes.")
        if self.output_shapes is None:
            raise ValueError("You must provide output_shapes.")
        if self.input_normalization_modes is None:
            raise ValueError("You must provide input_normalization_modes.")
        if self.output_normalization_modes is None:
            raise ValueError("You must provide output_normalization_modes.")
        if set(self.input_shapes.keys()) != set(self.input_normalization_modes.keys()):
            raise ValueError(
                "input_shapes and input_normalization_modes must have the same keys."
            )
        if set(self.output_shapes.keys()) != set(
            self.output_normalization_modes.keys()
        ):
            raise ValueError(
                "output_shapes and output_normalization_modes must have the same keys."
            )
