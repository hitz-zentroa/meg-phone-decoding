"""High-level **PyTorch training loops** for neural models.

The classical (scikit-learn) estimators are handled in
:pyfile:`mpd/models/classical.py`.
This module covers the neural branch and provides:

* :pyfunc:`train_ann`    – single train/validation split
* :pyfunc:`train_ann_cv` – *k*-fold cross-validation
* :pyfunc:`initialize_weights` – weight-initialisation helper

The concrete architecture (ANN, simple CNN, DyslexNet) is resolved at run-time
from the `model_type` string so that the pipelines do not need to import the
network definitions directly.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from torch import nn, optim

from mpd.models.nn_archs import ANNModel, CNNModel, DyslexNetTransformer
from mpd.models.utils import prepare_data, prepare_data_fold


def train_ann(
    xdata,
    y,
    lr=1e-4,
    max_epochs=100,
    early_stopping_patience=6,
    classes=2,
    adam=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements  # noqa: E501
    """
    Train a **neural network** with one *train/validation* split.

    Parameters
    ----------
    xdata : ndarray, shape `(n_samples, n_features)`
        Flattened input features.
    y : ndarray, shape `(n_samples,)`
        Labels.  If `classes > 2` they are integer-encoded automatically.
    lr : float, default `1e-4`
        Learning rate.
    max_epochs : int, default `100`
    early_stopping_patience : int, default `6`
        Training is interrupted if the validation loss does not improve for
        this many consecutive epochs.
    classes : int, default `2`
        Number of classes; selects Binary Cross-Entropy vs. Cross-Entropy loss.
    adam : bool, default `False`
        Use Adam (+weight-decay) instead of SGD.

    Returns
    -------
    list[dict]
        One dictionary per metric with keys `"Metric"` and `"Score"` –
        suitable for direct conversion to a pandas DataFrame.

    Examples
    --------
    >>> xdata = np.random.randn(888, 204*31)
    >>> y = np.random.randint(0, 2, 888)
    >>> train_ann(xdata, y, classes=2)
    """
    # Reproducibility
    torch.manual_seed(42)

    input_dim = xdata.shape[1]

    # CPU or GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize features
    xdata = (xdata - np.mean(xdata, axis=0)) / np.std(xdata, axis=0)

    # Prepare data loaders, which now handle label encoding automatically
    train_loader, val_loader = prepare_data(
        xdata, y, batch_size=64, split_ratio=0.2, classes=classes
    )

    # Initialize model
    model = ANNModel(input_dim, classes).to(device)
    model.train()  # Set the model to training mode

    # Select appropriate loss function
    criterion = nn.CrossEntropyLoss()  # if classes > 2 else nn.BCELoss()
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    # Early stopping
    best_val_loss = np.inf
    best_metrics = {}
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        val_loss /= len(val_loader)
        all_outputs = torch.cat(all_outputs).cpu()
        all_labels = torch.cat(all_labels).cpu()

        if classes == 2:
            predictions = torch.round(all_outputs)
            all_outputs = all_outputs[:, 1]  # Get positive class probabilities
        else:
            _, predictions = torch.max(all_outputs, 1)

        # Compute metrics
        accuracy = accuracy_score(all_labels, predictions)
        precision = precision_score(all_labels, predictions, average="macro")
        recall = recall_score(all_labels, predictions, average="macro")
        f1 = f1_score(all_labels, predictions, average="macro")

        # AUC computation for binary and multiclass scenarios
        if classes == 2:
            auc = roc_auc_score(all_labels, all_outputs)
        else:
            all_labels = label_binarize(all_labels, classes=list(range(classes)))
            auc = roc_auc_score(all_labels, all_outputs, multi_class="ovr")

        print(
            (
                f"Epoch {epoch + 1}, Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, "
                f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, "
                f"AUC: {auc:.4f}"
            )
        )

        # Early stopping logic and saving best metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "recall": recall,
                "F1": f1,
                "AUC": auc,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    rows = [{"Metric": n, "Score": s} for n, s in best_metrics.items()]
    return rows


def train_ann_cv(
    xdata,
    y,
    lr=1e-4,
    max_epochs=1000,
    early_stopping_patience=6,
    classes=2,
    n_splits=5,
    seed=42,
    hidden_layers=None,
    model_type=None,
    adam=False,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements  # noqa: E501
    """Cross-validated training for ANN / CNN / DyslexNet.

    The logic mirrors :pyfunc:`train_ann` but wraps it inside a
    :class:`sklearn.model_selection.KFold` loop and aggregates the metric
    dictionaries across folds.

    Only a summary of the most important parameters is shown here; refer to the
    single-split docstring for details.

    Example
    -------
    ```python
    >>> xdata = np.random.randn(888, 204 * 31)
    >>> y = np.random.randint(0, 2, 888)
    >>> model = train_ann_cv(xdata, y)
    ```
    """
    # Reproducibility
    torch.manual_seed(seed)

    # CPU or GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = xdata.shape[-1]
    xdata = (xdata - np.mean(xdata, axis=0)) / np.std(xdata, axis=0)
    results = []

    # K-Fold Cross-validation
    if seed is not None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=n_splits)

    fold = 0
    for train_index, val_index in kf.split(xdata):
        fold += 1
        xdata_train, xdata_val = xdata[train_index], xdata[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_loader, val_loader = prepare_data_fold(
            xdata_train, xdata_val, y_train, y_val, batch_size=64, classes=classes
        )

        # Initialize model
        if "ann" in model_type.lower():
            model = ANNModel(input_dim, classes, hidden_layers=hidden_layers)
        elif "cnn" in model_type.lower():
            model = CNNModel(
                input_dim,
                classes,
                hidden_layers=hidden_layers,
                input_channels=xdata.shape[1],
            )
        elif "dyslexnet" in model_type.lower():
            model = DyslexNetTransformer(
                input_dim=xdata.shape[1],
                seq_len=xdata.shape[2],
                num_classes=classes,
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")
        model = model.to(device)
        model.train()

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # if classes > 2 else nn.BCELoss()
        if adam:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr)

        # Early stopping
        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                # print("labels:", labels)
                # print("outputs:", outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            all_outputs = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    all_outputs.append(outputs)
                    all_labels.append(labels)

            val_loss /= len(val_loader)
            print(f"Fold {fold}, Epoch {epoch + 1}, Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
                continue  # loss did not improve, skip metrics calculation

            fold_results = {}
            all_outputs = torch.cat(all_outputs).cpu()
            all_labels = torch.cat(all_labels).cpu()

            # Correct way to get predictions
            predictions = torch.argmax(all_outputs, dim=1)

            # Get probabilities for AUC
            probs = F.softmax(all_outputs, dim=1)
            positive_probs = probs[:, 1]  # only the positive class

            # Compute metrics
            fold_results["Accuracy"] = accuracy_score(all_labels, predictions)
            fold_results["Precision"] = precision_score(
                all_labels, predictions, average="macro"
            )
            fold_results["Recall"] = recall_score(
                all_labels, predictions, average="macro"
            )
            fold_results["F1"] = f1_score(all_labels, predictions, average="macro")

            # AUC computation for binary and multiclass scenarios
            if classes == 2:
                fold_results["AUC"] = roc_auc_score(all_labels, positive_probs)
            else:
                all_labels_one_hot = label_binarize(
                    all_labels, classes=list(range(classes))
                )
                fold_results["AUC (OvO)"] = roc_auc_score(
                    all_labels_one_hot, all_outputs, multi_class="ovo"
                )
                fold_results["AUC (OvR)"] = roc_auc_score(
                    all_labels_one_hot, all_outputs, multi_class="ovr"
                )

                print(
                    (
                        f"  Acc: {fold_results['Accuracy']:.4f}, "
                        f"Prec: {fold_results['Precision']:.4f}, "
                        f"Rec: {fold_results['Recall']:.4f}, "
                        f"F1: {fold_results['F1']:.4f}, "
                        f"AUC OvO: {fold_results['AUC (OvO)']:.4f}, "
                        f"AUC OvR: {fold_results['AUC (OvR)']:.4f}"
                    )
                )

        results.extend([{"Metric": n, "Score": s} for n, s in fold_results.items()])
    return results


def initialize_weights(model):
    """Apply sensible initialisation schemes to CNN, BN and linear layers.

    * Conv2d:     Kaiming-uniform
    * BatchNorm:  weights = 1, bias = 0
    * Linear:     Xavier-uniform

    The function mutates *model* **in place** and returns `None`.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
