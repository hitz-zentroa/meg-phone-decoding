"""Utility functions shared by the different **PyTorch training loops**.

They take NumPy arrays produced by the feature-extraction stage and wrap them
into :class:`torch.utils.data.DataLoader` objects, handling label encoding for
multi-class problems.

Nothing in here is model-specific; the helpers are imported by both
`train_ann` (single split) and `train_ann_cv` (cross-validation).
"""

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split


def encode_labels(y):
    """Convert an arbitrary label vector to consecutive integer indices.

    Parameters
    ----------
    y : array-like
        Labels (strings, integers...).

    Returns
    -------
    y_encoded : ndarray[int]
        Zero-based class indices suitable for :class:`torch.nn.CrossEntropyLoss`.
    classes : ndarray
        The original class names in the order chosen by
        :class:`sklearn.preprocessing.LabelEncoder`.

    Notes
    -----
    The function is a paper-thin wrapper around *scikit-learn*’s
    :class:`LabelEncoder`, returned here to avoid re-importing it everywhere.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder.classes_


def prepare_data(xdata, y, batch_size=64, split_ratio=0.2, classes=2):
    """Create **train / validation** loaders from the full dataset.

    Parameters
    ----------
    xdata : ndarray, shape `(n_samples, ...)`
        Input features.
    y : ndarray
        Labels.  If `classes > 2` they will be encoded to integers.
    batch_size : int, default `64`
        Mini-batch size for both loaders.
    split_ratio : float, default `0.2`
        Fraction of samples reserved for the *validation* loader.
    classes : int, default `2`
        Number of classes in the task.  Determines label dtype
        (`float` for BCE vs. `long` for CE).

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
        Iterators that yield `(inputs, labels)` tensors on the correct dtype
        and device.
    """
    if classes > 2:
        y, _ = encode_labels(y)
        y = torch.tensor(y, dtype=torch.long)
    else:
        # Assuming y is already [0, 1] for binary tasks
        y = torch.tensor(y, dtype=torch.float32)

    xdata = torch.tensor(xdata, dtype=torch.float32)
    dataset = TensorDataset(xdata, y)

    train_size = int((1 - split_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def prepare_data_fold(
    x_train, x_val, y_train, y_val, batch_size=64, classes=2
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Create **train / validation** loaders when using cross-validation.

    Same as :pyfunc:`prepare_data`, but for **pre-split folds** inside a
    cross-validation loop.

    Parameters
    ----------
    X_train, X_val : ndarray
        Feature matrices for the current fold.
    y_train, y_val : ndarray
        Corresponding label vectors.
    batch_size : int, default `64`
    classes : int, default `2`

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    # Encode labels if multi-class classification
    if classes > 2:
        y_train, _ = encode_labels(y_train)
        y_val, _ = encode_labels(y_val)

    # Convert arrays to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create datasets and data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
