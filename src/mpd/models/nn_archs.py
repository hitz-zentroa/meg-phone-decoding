"""Neural-network architectures used by the training loops.

The file contains three *lightweight* baselines that work with very little
training data plus a Transformer encoder inspired by **DyslexNet**:

* :class:`ANNModel`           – fully-connected MLP with an arbitrary list of
  hidden layers.
* :class:`CNNModel`           – 1-D temporal CNN that first downsamples the
  waveform then applies one or more dense layers.
* :class:`CNNModel1C`         – variant that treats each sensor as an
  independent “image” channel before concatenating features.
* :class:`DyslexNetTransformer` – reduced BERT encoder with parameter sharing,
  adapted for MEG time-series (*no* token IDs are used).

All models end in a linear layer whose output dimension equals
`num_classes`; a `sigmoid` activation is appended automatically when
`num_classes == 2` so that the network can be trained with
:class:`torch.nn.BCEWithLogitsLoss` *or* :class:`torch.nn.CrossEntropyLoss`
depending on the task.
"""

import torch
from torch import nn
from transformers import BertConfig, BertModel


class ANNModel(nn.Module):
    """Simple fully-connected network.

    Parameters
    ----------
    num_features : int
        Size of the flattened input vector.
    classes : int
        Number of output classes.
    hidden_layers : list[int] or None
        Sequence of hidden-layer dimensions.  If `None` the model reduces to
        a single linear classifier.

    Notes
    -----
    * Activation is ReLU between hidden layers.
    * For binary tasks a `sigmoid` is applied in :pymeth:`forward`.
    """

    def __init__(self, num_features, classes, hidden_layers=None):
        super().__init__()
        self.classes = classes

        # List to hold all layers
        layers = []

        # Create hidden layers
        if hidden_layers is not None:
            # First hidden layer
            layers.append(nn.Linear(num_features, hidden_layers[0]))
            layers.append(nn.ReLU())  # Using ReLU for non-linearity
            print(f"Added layer: Linear({num_features}, {hidden_layers[0]})")

            # Additional hidden layers
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(nn.ReLU())
                print(
                    (
                        f"Added layer: Linear({hidden_layers[i-1]}, "
                        f"{hidden_layers[i]})"
                    )
                )

            # Final layer to output
            layers.append(nn.Linear(hidden_layers[-1], classes))
            print(f"Added layer: Linear({hidden_layers[-1]}, {classes})")
        else:
            # If no hidden layers, just add a single linear layer
            layers.append(nn.Linear(num_features, classes))
            print(f"Added layer: Linear({num_features}, {classes})")

        # Register all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Return raw logits; `sigmoid` is added for *binary* tasks."""
        # Process input through all layers
        x = self.layers(x)
        if self.classes == 2:
            # Use sigmoid for binary classification output
            x = torch.sigmoid(x)
        return x


class CNNModel(nn.Module):
    """
    One-dimensional CNN that downsamples the *time* axis before the dense head.

    Parameters
    ----------
    num_features : int
        Original length of the time axis.
    classes : int
        Number of output classes.
    hidden_layers : list[int] or None
        Fully-connected layers after the convolution.
    input_channels : int, default `1`
        Channels dimension fed into :class:`torch.nn.Conv1d`; usually the number
        of sensors.
    downsampling_factor : int, default `10`
        Stride / kernel size of the initial convolution (acts like average
        pooling + learned filter).

    Shape
    -----
    *Input*  – `(batch, input_channels, num_features)`
    *Output* – `(batch, classes)`
    """

    def __init__(
        self,
        num_features,
        classes,
        hidden_layers=None,
        input_channels=1,
        downsampling_factor=10,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        self.classes = classes

        # Calculate the output length after convolution
        output_length = num_features // downsampling_factor

        # Convolutional layer for downsampling
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=1,
            kernel_size=downsampling_factor,
            stride=downsampling_factor,
        )

        print(
            (
                f"Added layer: Conv1d(in_channels={input_channels}, "
                f"out_channels=1, kernel_size={downsampling_factor}, "
                f"stride={downsampling_factor})\n"
                f"  num_features: {num_features}"
                f"  output_length: {output_length}"
            )
        )

        # Fully connected layers
        layers = []
        current_input = output_length

        if hidden_layers is not None:
            # First hidden layer
            layers.append(nn.Linear(current_input, hidden_layers[0]))
            layers.append(nn.ReLU())
            print(f"Added layer: Linear({current_input}, {hidden_layers[0]})")

            # Additional hidden layers
            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(nn.ReLU())
                print(
                    (
                        f"Added layer: Linear({hidden_layers[i-1]}, "
                        f"{hidden_layers[i]})"
                    )
                )

            # Final layer to output
            layers.append(nn.Linear(hidden_layers[-1], classes))
            print(f"Added layer: Linear({hidden_layers[-1]}, {classes})")
        else:
            # If no hidden layers, just add a single linear layer
            layers.append(nn.Linear(current_input, classes))
            print(f"Added layer: Linear({current_input}, {classes})")

        # Register fully connected layers
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        """Return raw logits (sigmoid added if *binary*)."""
        # Assuming x is of shape (batch_size, channels, num_features)
        x = self.conv(x)  # Apply convolutional layer for downsampling
        # Flatten the outputs for the fully connected layers
        x = x.view(x.size(0), -1)
        # Process through all fully connected layers
        x = self.fc_layers(x)
        if self.classes == 2:
            # Use sigmoid for binary classification output
            x = torch.sigmoid(x)
        return x


class CNNModel1C(nn.Module):
    """
    Alternate CNN that first extracts features **per sensor**.

    The convolution is applied with `num_channels=1` so that each sensor is
    processed independently; the resulting feature maps are flattened and
    passed to a dense head identical to :class:`CNNModel`.
    """

    def __init__(
        self,
        num_features,
        classes,
        hidden_layers=None,
        num_channels=1,
        out_channels=16,
        kernel_size=5,
        stride=2,
        padding=2,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        self.classes = classes

        # Define a simple CNN layer to downsample/feature extract
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        print(
            (
                f"Added layer: Conv1d(in_channels={num_channels}, "
                f"out_channels={out_channels}, kernel_size={kernel_size}, "
                f"stride={stride}, padding={padding})\n"
                "  ReLU()\n"
                "  MaxPool1d(kernel_size=2, stride=2)"
            )
        )

        # Calculate the output size after the CNN layers
        self.output_size = self._get_conv_output_size(
            num_features, out_channels, kernel_size, stride, padding
        )

        # List to hold all FC layers
        layers = []

        # Create FC hidden layers
        input_dim = self.output_size * out_channels
        if hidden_layers is not None:
            layers.append(nn.Linear(input_dim, hidden_layers[0]))
            layers.append(nn.ReLU())
            print(f"Added layer: Linear({input_dim}, {hidden_layers[0]})")

            for i in range(1, len(hidden_layers)):
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(nn.ReLU())
                print(
                    (
                        f"Added layer: Linear({hidden_layers[i-1]}, "
                        f"{hidden_layers[i]})"
                    )
                )

            layers.append(nn.Linear(hidden_layers[-1], classes))
        else:
            layers.append(nn.Linear(input_dim, classes))
        print(
            (
                "Added layer: "
                f"Linear({input_dim if hidden_layers else num_features}, "
                f"{classes})"
            )
        )

        self.fc = nn.Sequential(*layers)

    def _get_conv_output_size(
        self, size, out_channels, kernel_size, stride, padding
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments, unused-argument  # noqa: E501
        # `out_channles` is not used here

        # Apply convolution parameters
        size = (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        # Apply max pooling
        size = (size - 2) // 2 + 1  # Assuming kernel_size=2 for MaxPool1d
        return size

    def forward(self, x):
        """Return raw logits (sigmoid added if *binary*)."""
        # Ensure input x is of shape (batch_size, channels, length)
        if x.dim() == 2:
            # Unsqueeze to add channel dimension if necessary
            x = x.unsqueeze(1)

        # Apply CNN
        x = self.cnn(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, start_dim=1)

        # Apply the fully connected layers
        x = self.fc(x)
        if self.classes == 2:
            x = torch.sigmoid(x)
        return x


class DyslexNetTransformer(nn.Module):
    """Model based in DyslexNet, implementation simplified using BERT from HF.

    Each time-step is projected to `emb_size` and then to the *factorised*
    dimension `factor_size` before reaching the full Transformer hidden size.
    All encoder layers **share parameters** (like ALBERT) to keep the model
    small enough for the dataset.

    - Paper: https://sciencedirect.com/science/article/pii/S1053811923002185

    Parameters
    ----------
    input_dim : int
        Number of sensors (features per time-step).
    seq_len : int
        Number of time-steps in the input epoch.
    num_classes : int, default `2`
    hidden_size : int, default `3072`
    emb_size : int, default `768`
    factor_size : int, default `128`
    num_heads : int, default `12`
    num_layers : int, default `4`
        Logical number of layers **before sharing**.
    """

    def __init__(
        self,
        input_dim,
        seq_len,
        num_classes=2,
        hidden_size=3072,
        emb_size=768,
        factor_size=128,
        num_heads=12,
        num_layers=4,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_size)
        self.factorization = nn.Linear(emb_size, factor_size)
        self.projection = nn.Linear(factor_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, hidden_size))

        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=seq_len,
            vocab_size=1,  # dummy
            pad_token_id=0,
        )
        self.encoder = BertModel(config)

        # Cross-layer parameter sharing
        shared_layer = self.encoder.encoder.layer[0]
        for i in range(1, len(self.encoder.encoder.layer)):
            self.encoder.encoder.layer[i] = shared_layer

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Define the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor, shape `(batch, channels, time)`
            Raw MEG epoch.

        Returns
        -------
        torch.Tensor
            Logits of shape `(batch, num_classes)`.
        """
        # from (batch, channels, timesteps) to (batch, timesteps, channels)
        x = x.transpose(1, 2)

        x = self.embedding(x)
        x = self.factorization(x)
        x = self.projection(x)
        x = x + self.pos_encoding
        outputs = self.encoder(inputs_embeds=x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled_output)
