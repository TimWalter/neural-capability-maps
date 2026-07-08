import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class Model(nn.Module):
    """
    RAM model that predicts reachability from a modified Denavit-Hartenberg morphology parametrisation and a pose.
    """

    @classmethod
    def from_id(cls, model_id: int):
        """
        Instantiate a model from its wandb ID.

        Args:
            model_id: Wandb ID of the model.

        Returns:
            model: Instantiated model.
        """

        model_dir = Path(__file__).parent.parent / "data" / "trained_models"
        pattern = rf"{model_id}-[a-z]+-[a-z]+"
        folder = next((f for f in model_dir.iterdir() if re.match(pattern, f.name)), None)
        metadata_path = model_dir / folder / 'metadata.json'
        metadata = json.load(open(metadata_path, 'r'))

        model = cls(**metadata["hyperparameter"])
        model_folder = Path(str(model_dir / folder))
        prime = model_folder / "model.pth"
        if prime.exists():
            model.load_state_dict(torch.load(prime))
        else:
            model.load_state_dict(torch.load(model_folder / "checkpoint.pth"))
        return model

    def __init__(self,
                 dim_encoding: int = 128,
                 num_encoder_layers: int = 1,
                 drop_prob: float = 0.0,
                 dim_decoder: int = 1792,
                 num_decoder_layer: int = 8):
        """
        Initialise the model.

        Args:
            dim_encoding: The dimension of the latent morphology encoding.
            num_encoder_layers: Number of LSTM layers.
            drop_prob: Dropout probability of the LSTM.
            dim_decoder: Hidden dimension of the MLP.
            num_decoder_layer: Number of layers of the MLP.
        """

        super().__init__()
        self.encoder = nn.LSTM(3, dim_encoding, num_encoder_layers, dropout=drop_prob, batch_first=True, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(9 + dim_encoding, dim_decoder),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(dim_decoder, dim_decoder), nn.ReLU())
              for _ in range(num_decoder_layer)],
            nn.Linear(dim_decoder, 1)
        )

    def forward(self, morph: Float[Tensor, "batch seq 3"], pose: Float[Tensor, "batch 9"]) -> Float[Tensor, "batch"]:
        """
        Predict reachability.

        Args:
            morph: Morphology description.
            pose: Pose as vector encoded.
        Returns:
            Reachability logit.
        """
        lengths = (morph != 0).any(dim=-1).sum(dim=-1).cpu()
        morph = torch.nn.utils.rnn.pack_padded_sequence(morph, lengths=lengths, batch_first=True, enforce_sorted=False)
        latent = self.encoder(morph)[1][0][-1]
        logit = self.decoder(torch.cat([pose, latent], dim=-1)).squeeze(-1)
        return logit

    @torch.inference_mode()
    def predict(self, *args, **kwargs):
        """
        forward with inference mode
        """
        return self.forward(*args, **kwargs)
