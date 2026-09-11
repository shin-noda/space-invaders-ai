import torch
import torch.nn as nn


class AtariCNN(nn.Module):
    """
    Convolutional neural network for Atari image observations.

    Input:
        (batch, channels, height, width)

    Output:
        (batch, num_actions)

    The output can represent:
        - policy logits for REINFORCE
        - Q-values for DQN
    """

    def __init__(
        self,
        in_channels=4,
        num_actions=2,
        input_size=(84, 84),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )

        # Determine flattened feature size automatically.
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                in_channels,
                input_size[1],
                input_size[0],
            )

            feature_size = self.conv(
                dummy
            ).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(
                feature_size,
                512,
            ),
            nn.ReLU(),

            nn.Linear(
                512,
                num_actions,
            ),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization.
        """

        for layer in self.modules():

            if isinstance(
                layer,
                (nn.Conv2d, nn.Linear),
            ):
                nn.init.orthogonal_(
                    layer.weight,
                    gain=nn.init.calculate_gain(
                        "relu"
                    ),
                )

                if layer.bias is not None:
                    nn.init.constant_(
                        layer.bias,
                        0.0,
                    )

    def forward(self, x):
        """
        Forward pass.
        """

        x = x.float()

        features = self.conv(x)

        features = features.view(
            features.size(0),
            -1,
        )

        return self.fc(features)