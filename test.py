import os
import yaml
import pickle
import torch
import argparse
from torch.utils.data import DataLoader

#####################################################
# Helper imports
#####################################################
from data import WindowDataset, collate_single
from config import get_config, set_dir

#####################################################
# Helpler functions
#####################################################
@torch.no_grad()
def DTA_scoring(X: torch.Tensor, U1: torch.Tensor, U2: torch.Tensor, U3: torch.Tensor):
    """
    X: torch.sparse_coo_tensor, shape (I1,I2,I3)
    U1: (I1,R1), U2: (I2,R2), U3: (I3,R3), assumed orthonormal columns
    returns: reconstruction error (squared by default)
    """
    X = X.coalesce()
    idx = X.indices()
    vals = X.values()

    # ||X||_F^2 for sparse tensor
    x_norm2 = (vals * vals).sum()

    # Handle degenrate case of zero tensor
    if x_norm2 == 0:
        return 0.0

    i1, i2, i3 = idx[0], idx[1], idx[2]

    # Gather factor rows corresponding to each nonzero
    U1 = U1.to(dtype=vals.dtype, device=vals.device)
    U2 = U2.to(dtype=vals.dtype, device=vals.device)
    U3 = U3.to(dtype=vals.dtype, device=vals.device)

    A = U1[i1]  # (nnz, R1)
    B = U2[i2]  # (nnz, R2)
    C = U3[i3]  # (nnz, R3)

    # Core tensor Y = X ×1 U1^T ×2 U2^T ×3 U3^T
    # Y[a,b,c] = sum_n vals[n] * A[n,a]*B[n,b]*C[n,c]
    Y = torch.einsum('n,na,nb,nc->abc', vals, A, B, C)

    y_norm2 = (Y * Y).sum()

    # Since U is orthonormal we have that
    # ||X - X_hat||_F^2 / ||X||_F^2 = 1 - ||Y||_F^2/||X||_F^2 
    err = 1 - (y_norm2/(x_norm2 + 1e-8))

    return err.item()


#####################################################
# Main program
#####################################################
def main():

    # Load config
    config = get_config()

    # Params
    test_dir = config['paths']['test_dir']
    model_factors_path = config['paths']['model_factors']
    score_file_path = config['paths']['score_file']

    # Dirs 
    set_dir( os.path.dirname(score_file_path) )

    # Dataset
    dataset = WindowDataset(test_dir)

    # Load
    test_dataset = DataLoader(
            dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = 0,
            collate_fn = collate_single,
    )

    # Verify sparse tensors
    torch.sparse.check_sparse_tensor_invariants.enable()

    # Load trained factors
    with open(model_factors_path, "rb") as f:
        DTA_params = pickle.load(f)

    # Anomaly scoring
    U1 = torch.as_tensor(DTA_params['U_mode_1'])
    U2 = torch.as_tensor(DTA_params['U_mode_2'])
    U3 = torch.as_tensor(DTA_params['U_mode_3'])
    anomaly_scores = []
    for ix, sample in enumerate(test_dataset):
        A = sample['A']
        anomaly_scores.append( DTA_scoring(A, U1, U2, U3) )

    with open(score_file_path, "wb") as f:
        pickle.dump(anomaly_scores, f)

if __name__ == "__main__":
    main()
