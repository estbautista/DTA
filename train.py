import os
import yaml
import torch
import pickle
import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
from torch.utils.data import Dataset, DataLoader, Subset

#####################################################
# Helper imports
#####################################################
from data import WindowDataset, collate_single
from config import get_config, set_dir

#####################################################
# Helper functions 
#####################################################
def sparse_unfoldings(X: torch.Tensor):
    """
    X: torch.sparse_coo_tensor of shape (I1, I2, I3)
    Returns the mode unfoldings as SciPy CSR matrices:
      X_(1): (I2*I3, I1)
      X_(2): (I1*I3, I2)
      X_(3): (I1*I2, I3)
    """
    X = X.coalesce()
    idx = X.indices().cpu().numpy()   # (3, nnz)
    vals = X.values().cpu().numpy()   # (nnz,)
    I1, I2, I3 = X.shape
    i1, i2, i3 = idx[0], idx[1], idx[2]

    X1 = csr_matrix((vals, (i2 * I3 + i3, i1)), shape=(I2 * I3, I1))
    X2 = csr_matrix((vals, (i1 * I3 + i3, i2)), shape=(I1 * I3, I2))
    X3 = csr_matrix((vals, (i1 * I2 + i2, i3)), shape=(I1 * I2, I3))
    return X1, X2, X3


class OnlineUS:
    def __init__(self, N, k):
        self.N = N
        self.k = k
        self.U = np.eye(N, k)   # Truncated eigenvectors 
        self.S = np.zeros(k)    # Truncated eigenvalues 

    def update_from_unfolding(self, Xd_csr, which="LA"):
        """
        Xd_csr: SciPy CSR matrix with shape (J, N) representing X_(d).
        Updates (U, S) to approximate eigs of:
          C_new = U diag(S) U^T + Xd^T Xd
        """
        U = self.U
        S = self.S
        N = self.N

        def matvec(v):
            # low-rank part
            y = U.T @ v              # (k,)
            y = S * y                # (k,)
            out = U @ y              # (N,)

            # new-data gram part
            out += Xd_csr.T @ (Xd_csr @ v)   # (N,)
            return out

        C_op = LinearOperator((N, N), matvec=matvec)

        # eigsh returns k eigenpairs of the operator
        evals, evecs = eigsh(C_op, k=self.k, which=which)

        # sort descending for LA
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        self.S = evals.copy()
        self.U = evecs.copy()
        return self.S, self.U


class OnlineGramTracker3D:
    def __init__(self, shape, k):
        I1, I2, I3 = shape
        self.t1 = OnlineUS(I1, k)
        self.t2 = OnlineUS(I2, k)
        self.t3 = OnlineUS(I3, k)

    def update(self, X_sparse_torch):
        X1, X2, X3 = sparse_unfoldings(X_sparse_torch)

        ev1, U1 = self.t1.update_from_unfolding(X1)
        ev2, U2 = self.t2.update_from_unfolding(X2)
        ev3, U3 = self.t3.update_from_unfolding(X3)

        return (ev1, U1), (ev2, U2), (ev3, U3)


#####################################################
# Main program
#####################################################
def main():

    # Load config
    config = get_config()

    # Params
    train_dir = config['paths']['train_dir']
    model_factors_path = config['paths']['model_factors']

    val_ratio = float(config['global']['val_ratio'])
    rank = int(config['model']['rank'])

    # Dirs 
    set_dir( os.path.dirname(model_factors_path) )

    # Dataset
    dataset = WindowDataset(train_dir)

    # Split train - val
    train_size = int( len(dataset)*(1-val_ratio) )
    train_dataset = Subset(dataset, range(0, train_size))
    print(val_ratio)
    print(train_size)
    print(len(train_dataset))

    # Load
    train_loader = DataLoader(
            train_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = 0,
            collate_fn = collate_single,
    )

    # Verify sparse tensors
    torch.sparse.check_sparse_tensor_invariants.enable()

    # Train the factor matrices
    I1, I2, I3 = dataset[0]['A'].shape
    runner = OnlineGramTracker3D(shape=(I1, I2, I3), k=rank)
    for ix, sample in enumerate(train_loader):
        print(f'Processing sample {ix}...')
        #if ix < len(train_dataset):  continue
        if sample['A']._nnz() == 0:
            continue
        runner.update(sample['A'])


    DTA_trained_params = dict()
    DTA_trained_params['eigenvalues_mode_1']=runner.t1.S
    DTA_trained_params['eigenvalues_mode_2']=runner.t2.S
    DTA_trained_params['eigenvalues_mode_3']=runner.t3.S
    DTA_trained_params['U_mode_1']=runner.t1.U
    DTA_trained_params['U_mode_2']=runner.t2.U
    DTA_trained_params['U_mode_3']=runner.t3.U

    with open(model_factors_path, "wb") as f:
        pickle.dump(DTA_trained_params, f)

if __name__ == "__main__":
    main()
