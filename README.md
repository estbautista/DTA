# DTA

This repository contains an independent Python implementation of Dynamic Tensor Analysis (DTA), proposed by Jimeng Sun, Dacheng Tao, and Christos Faloutsos in [*Beyond Streams and Graphs: Dynamic Tensor Analysis* (KDD 2006)](https://doi.org/10.1145/1150402.1150445). It was developed as a baseline for the experimental evaluation of **MuTGN** and is not the original authors' software.

The implementation detects window-level anomalies in temporal graphs. A window corresponds to the interactions occurring within a contiguous time interval. Each window is represented by a sparse adjacency tensor. The code expects chronologically ordered windows, partitioned into training and test sets that preserve temporal order.

The present implementation uses DTA in the training set to incrementally learn low-rank projection matrices that capture the structure of the training tensors. It then applies the learned projections to the test windows, using reconstruction error as an anomaly score.

## Method

The training procedure follows the Dynamic Tensor Analysis (DTA) algorithm described in Figure 7 of the original paper, which incrementally updates low-rank tensor projections through eigendecomposition. For each window, the sparse tensor is unfolded along its three modes. For each mode $d$, the current unfolding is used to update a low-rank approximation of the accumulated Gram matrix:

$$
C_d = U_d\,\mathrm{diag}(S_d)\,U_d^\top + X_{(d)}^\top X_{(d)},
$$

where $X_{(d)}$ denotes the mode-$d$ unfolding, and $U_d$ and $S_d$ contain the previously retained eigenvectors and eigenvalues, respectively. Rather than explicitly constructing $C_d$, the implementation computes its leading eigenpairs using SciPy's `eigsh` with a matrix-free `LinearOperator`. All three modes use the same configurable rank.

At test time, the learned projection matrices remain fixed and are applied to each test window. Each window receives an anomaly score measuring how poorly it can be reconstructed from its low-rank approximation. The score corresponds to the squared Frobenius norm of the reconstruction error, normalized by the squared Frobenius norm of the original tensor. To avoid constructing the dense reconstruction $\hat X$, the implementation computes the reconstruction error using the equivalence:

$$
\boxed{ \frac{\|X-\hat X\|_F^2}{\|X\|_F^2} = 1-\frac{\|Y\|_F^2}{\|X\|_F^2} }
$$

where $Y$ is the low-rank core tensor. This equivalence follows from the orthonormality of the projection matrices. The code adds a small constant to the denominator for numerical stability.

## Implementation choices relative to the original paper

This implementation of DTA does not reproduce every algorithm or application reported in the original paper. It was developed as a baseline for the experimental evaluation of **MuTGN**. It therefore makes the following implementation choices relative to the original paper:

- **Tensor representation:** the code supports three-dimensional sparse tensors, rather than arbitrary-order tensors.
- **Rank selection:** it uses a fixed `model.rank`, shared by all three modes, rather than the paper's energy-based adaptive-rank procedure.
- **Historical weighting:** it accumulates past information without a forgetting factor, corresponding to $\lambda=1$ in the paper's optional forgetting formulation.
- **Training and testing:** it updates the mode projections on the training portion only, then freezes them for testing.
- **Anomaly granularity:** it outputs only a tensor-level (window-level) score. The paper also considers mode-level and dimension-level anomalies.
- **Scoring and decisions:** it exports a continuous normalized squared reconstruction error as an anomaly score. It does not apply the paper's historical-error threshold for binary decisions or implement the other error definitions and evaluations discussed in the paper.

For a test window tensor $X$ and its projected core $Y = X\times_1 U_1^\top\times_2 U_2^\top\times_3 U_3^\top$, the implemented score is

$$
s(X) = 1 - \frac{\lVert Y\rVert_F^2}{\lVert X\rVert_F^2 + 10^{-8}}.
$$

An empty (zero-norm) window is assigned a score of `0.0`.

## Configuration

The training and testing scripts are controlled through separate YAML files passed via `--config`. An example of a training configuration is:

```yaml
global:
  val_ratio: 0.15

model:
  rank: 8

paths:
  train_dir: /data/train
  model_factors: /models/learned_factors_8.pkl
```

An example of a testing configuration is:

```yaml
paths:
  test_dir: /data/test
  model_factors: /models/learned_factors_8.pkl
  score_file: /outputs/scores.pkl
```

`model.rank` specifies the retained number of eigenvectors in each mode. `global.val_ratio` reserves the last fraction of the supplied training windows for validation: the training script fits on the first `int(N * (1 - val_ratio))` windows. The utility `scores_train_data.py` can inspect scores on that held-out portion. Window tensors are loaded on CPU, and the SciPy eigensolver runs on CPU. All paths refer to locations inside the container. 

Example configuration files are provided in `train.config.example.yaml` and `test.config.example.yaml`.

## Input and output

The implementation expects separate directories of preprocessed training and test windows, specified through `paths.train_dir` and `paths.test_dir`, respectively. Each directory contains PyTorch `.pt` files named `window_*.pt`, which are loaded in lexicographic filename order. Filenames must therefore preserve chronological order, for example by using zero-padded indices.

Each file is loaded using torch.load() and must contain the window's sparse adjacency tensor under the key `A`, represented as a three-dimensional PyTorch sparse COO tensor. From this data structure the scripts gather the non-zero indices, their values, and the tensor shape. Tensor dimensions must be consistent across windows and compatible with the learned projections.

The training script (`train.py`) saves the learned projection matrices and eigenvalues to a pickle file specified by `paths.model_factors`. The testing script (`test.py`) loads these factors and exports a pickle file containing a Python list of floating-point anomaly scores, ordered by test window, to `paths.score_file`.

The creation of the `.pt` window files and the evaluation of anomaly scores are handled externally.

## Reproducible environment

A reconstructed environment with Python 3.15.2 and the versions pinned in [`requirements-reconstructed.txt`](requirements-reconstructed.txt) has been validated in a local environment and a container. The main dependencies are PyTorch `2.15.0`, NumPy `2.5.5`, SciPy `1.18.1`, and PyYAML `6.0.4`.

From the repository root, build the container image using [Podman](https://podman.io/) (Docker can also be used):

```bash
podman build --no-cache -t dta:locked .
```

Verify that the implementation imports successfully:

```bash
podman run --rm dta:locked \
  python -c "import torch, numpy, scipy, yaml; print('DTA imports OK')"
```

The image uses `python:3.15.2-slim-trixie` and runs `train.py` and `test.py` on demand. Datasets, model factors, and score files are stored outside the image.

## Running window-level anomaly detection

Prepare two host directories containing the preprocessed training and test `.pt` windows, respectively, and place the train and test YAML in the repository's root. From the repository root:

```bash
mkdir -p models outputs

TRAIN_WINDOWS="/path/to/preprocessed/windows/train"
TEST_WINDOWS="/path/to/preprocessed/windows/test"

podman run --rm \
  -v "$PWD/config.train.yaml:/train_config.yaml:ro,z" \
  -v "$PWD/config.test.yaml:/test_config.yaml:ro,z" \
  -v "$TRAIN_WINDOWS:/data/train:ro,z" \
  -v "$TEST_WINDOWS:/data/test:ro,z" \
  -v "$PWD/models:/models:Z" \
  -v "$PWD/outputs:/outputs:Z" \
  dta:locked \
  sh -c 'python -u /app/train.py --config /train_config.yaml && \
         python -u /app/test.py --config /test_config.yaml'
```

On systems with SELinux enabled, `:z` and `:Z` apply the appropriate labels to volume mounts. These options can be omitted on systems where SELinux is disabled.

Make sure the train and test configuration files point to the same `paths.model_factors`.

## Repository contents

| File | Purpose |
| --- | --- |
| `train.py` | Incrementally learn and serialize mode-wise DTA factors from training windows |
| `test.py` | Score test windows using learned factors and serialize the scores |
| `data.py` | Discover and load sparse `.pt` window samples |
| `config.py` | Read YAML configuration and create output directories |
| `scores_train_data.py` | Optional analysis of anomaly scores on the held-out training/validation portion |
| `Dockerfile` | Reconstructed container environment |
| `requirements-reconstructed.txt` | Pinned dependency versions |

## License and attribution

This repository contains an independent implementation developed for the MuTGN benchmark. It is not the official software of Sun, Tao, and Faloutsos.

The code is released under the **European Union Public Licence, version 1.2 (EUPL-1.2)**. The license text is provided in [`LICENSE.txt`](LICENSE.txt).

## Citation

If you use the DTA method, please cite its original paper:

```bibtex
@inproceedings{sun2006beyond,
  title     = {Beyond Streams and Graphs: Dynamic Tensor Analysis},
  author    = {Sun, Jimeng and Tao, Dacheng and Faloutsos, Christos},
  booktitle = {Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {374--383},
  year      = {2006},
  publisher = {ACM},
  doi       = {10.1145/1150402.1150445}
}
```

For the original DTA formulation, anomaly-detection procedure, and experimental results, see the [KDD 2006 paper](https://doi.org/10.1145/1150402.1150445).
