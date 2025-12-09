import torch
import torch.nn as nn
import torch.nn.functional as F


class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for
        deterministic output.

        This method ensures that the output remains consistent across different
        runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular
              vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular
              vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = self._svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_

if __name__ == "__main__":
    import numpy as np
    from sklearn.decomposition import PCA as sklearn_PCA
    from sklearn import datasets
    iris = torch.tensor(datasets.load_iris().data)
    _iris = iris.numpy()
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        iris = iris.to(device)
        for n_components in (2, 4, None):
            _pca = sklearn_PCA(n_components=n_components).fit(_iris)
            _components = torch.tensor(_pca.components_)
            pca = PCA(n_components=n_components).to(device).fit(iris)
            components = pca.components_
            assert torch.allclose(components, _components.to(device))
            _t = torch.tensor(_pca.transform(_iris))
            t = pca.transform(iris)
            assert torch.allclose(t, _t.to(device))
        __iris = pca.inverse_transform(t)
        assert torch.allclose(__iris, iris)
    print("passed!")
