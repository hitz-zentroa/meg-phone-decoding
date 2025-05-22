# MPD - MEG Phone Decoding

[![IberSPEECH 2024 - Paper](https://img.shields.io/badge/doi-10.21437%2fIberSPEECH.2024--16-b31b1b)](https://www.isca-archive.org/iberspeech_2024/dezuazo24_iberspeech.html)
[![arXiv - Paper](https://img.shields.io/badge/cs.CL-2505.15355-b31b1b?&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.15355)
[![Build](https://github.com/hitz-zentroa/meg-phone-decoding/actions/workflows/python-app.yml/badge.svg)](https://github.com/hitz-zentroa/meg-phone-decoding/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HiTZ](https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet)](http://www.hitz.eus/)

Code for the following papers:

* [**Phone Pair Classification During Speech Production Using MEG Recordings (2024)**](https://www.isca-archive.org/iberspeech_2024/dezuazo24_iberspeech.html)
* [**Decoding Phone Pairs from MEG Signals Across Speech Modalities (2025)**](https://arxiv.org/abs/2505.15355)

It trains either linear models or compact neural networks to discriminate
phones from low-frequency MEG activity.

The whole pipeline (data I/O -> epoching -> preprocessing -> modelling ->
cross-validation -> CSV output) is wrapped in a single command-line entry-point
called `mpd-train`.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/hitz-zentroa/meg-phone-decoding.git
cd meg-phone-decoding
pip install -e .
```

**Note:** the
[Bourguignon2020 dataset](https://doi.org/10.1016/j.neuroimage.2020.116788)
used in the paper cannot be redistributed. You may still run the code on your
own recordings (see next sections).

## Quick start (Bourguignon2020 layout)

Assuming the corpus lives in `~/datasets/bourguignon2020/`:

```bash
mpd-train ~/datasets/bourguignon2020 \
    --model elasticnet \
    --meg grad \
    --wavelets \
    --frequencies lpbeta \
    --output logs/bourguignon2020_elasticnet_grad_wavelets_lpbeta.csv
```

* Default is 5-fold cross-validation per **subject × task × phone-pair**.

* Results are aggregated in the `*.csv` file, channel weights (for linear models)
  are written to `*-weights.csv`

## Using a different dataset

Only two things depend on the **folder structure / annotation format**:


You have two options:

### 1. Mimic the Bourguignon2020 layout

```
dataset/
  meg/
    selfp1/
      selfp1_self_tsss.fif
      selfp1_self_norm.wav
      selfp1_self_norm.TextGrid
      selfp1_self_realign.mat   # optional – only if you need sample-accurate sync
```

### 2. Plugging your own dataset

You only need **two functions**:

```python
def get_subjects(root: Path) -> list[str]:
    """Return the list of subjects in the dataset.

    Parameters
    ----------
    root : Path
        Root folder of the corpus (the directory that contains "meg/").

    Returns
    -------
    list[str]
        Subject names such as "subj01", "subj02", ...

    """
    # Implementation example
    return sorted(p.name for p in (root/"subj").iterdir() if p.is_dir())


def read_raw_with_annotations(root: Path, subject: str, task: str)
        -> tuple[mne.io.Raw, dict[str, mne.Annotations]]:
    """Load a single MEG recording together with phone-level annotations.

    Parameters
    ----------
    root : Path
        Dataset root folder.
    subject : str
        Subject identifier, e.g. "selfp05".
    task : str
        Experimental condition like "listen", "reading", ...

    Returns
    -------
    raw : mne.io.Raw
        MEG recording (with `preload=True`).
    annotations : dict[str, mne.Annotations]
        One entry per TextGrid tier.  Each `mne.Annotations` contains onset,
        duration and description arrays that are aligned to the MEG sample rate.
    """
    # Implementation example
    raw = mne.io.read_raw_fif(root / "subj" / subject / f"{task}.fif", preload=True)
    # create mne.Annotations named "phones"
    # [...]
    return raw, {"phones": annotations}
```

With `annotations` being an `mne.Annotations` object whose descriptions are
phones. The rest of the pipeline will work untouched.

Put them in a module, register the key in `mpd.data._DATASETS`, and call

```shell
mpd-train /path/to/your_ds --dataset-format mymeg ...
```

For a real example, you can check the `src/mpd/data/bourguignon2020.py` file.

No other change is required.

## Typical tweaks

| Need | CLI flag (see `mpd-train --help`) |
| ---- | --------------------------------- |
| Disable decimation | `--decim 1` |
| Restrict to gradiometers only | `--meg grad` |
| Try a small CNN | `--model cnn --hidden-layers 128 64` |
| Train with wavelet denoising disabled | (omit `--wavelets`) |
| Band-pass to Delta–Theta only	| `--frequencies lptheta` |
| Multi-class decoder (no pairwise CV) | `--multiclass` |

## Citation

If you find this helpful in your research, please cite:

```bibtex
@inproceedings{dezuazo24_iberspeech,
  title     = {Phone Pair Classification During Speech Production Using MEG Recordings},
  author    = {Xabier {de Zuazo} and Eva Navas and Ibon Saratxaga and Mathieu Bourguignon and Nicola Molinaro},
  year      = {2024},
  booktitle = {IberSPEECH 2024},
  pages     = {76--80},
  doi       = {10.21437/IberSPEECH.2024-16},
}
```

```bibtex
@misc{dezuazo2025decodingphonepairsmeg,
  title         = {Decoding Phone Pairs from MEG Signals Across Speech Modalities},
  author        = {Xabier de Zuazo and Eva Navas and Ibon Saratxaga and Mathieu Bourguignon and Nicola Molinaro},
  year          = {2025},
  eprint        = {2505.15355},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2505.15355},
}
```

Please, check the related papers in
[10.21437/IberSPEECH.2024-16](https://www.isca-archive.org/iberspeech_2024/dezuazo24_iberspeech.html)
and [arXiv:2505.15355](https://arxiv.org/abs/2505.15355)
for more details.
