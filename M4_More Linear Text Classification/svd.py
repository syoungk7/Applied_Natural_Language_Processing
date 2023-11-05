import numpy as np


class SVD(object):
    def __init__(self):
        pass

    def svd(self, data):
        """
        Do SVD. You could use numpy SVD.
        Args: data: (N, D) TF-IDF features for the data.
        Return: U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V^T: (D,D) numpy array
        """
        U, S, V = np.linalg.svd(data, full_matrices=True)
        return U, S, V

    def rebuild_svd(self, U, S, V, k):
        """
        Rebuild SVD by k componments.
        Args: U: (N,N) numpy array
              S: (min(N,D), ) numpy array
              V: (D,D) numpy array
              k: int corresponding to number of components
        Return: data_rebuild: (N,D) numpy array
        Hint: numpy.matmul may be helpful for reconstruction.
        """
        if S.shape[0] == U.shape[0]:
                data_rebuild = np.matmul(np.dot(U[:, 0:k], np.diag(S[0:k])), V[0:k, :])
        else:
                data_rebuild = np.matmul(U[:, 0:k], np.dot(np.diag(S[0:k]), V[0:k, :]))
        return data_rebuild

    def compression_ratio(self, data, k):
        """
        Compute the compression ratio: (num stored values in compressed)/(num stored values in original)
        Args: data: (N, D) TF-IDF features for the data.
              k: int corresponding to number of components
        Return: compression_ratio: float of proportion of storage used
        """
        compression_ratio = (k*(1+data.shape[0]+data.shape[1])) / (data.shape[0]*data.shape[1])
        return compression_ratio

    def recovered_variance_proportion(self, S, k):
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation
        Args: S: (min(N,D), ) numpy array
              k: int, rank of approximation
        Return: recovered_var: float corresponding to proportion of recovered variance
        """
        dino, recovered_var = 0, 0

        for i in range(S.shape[0]):
                dino += S[i]**2
        for j in range(k):
                recovered_var += (S[j]**2 / dino)

        return recovered_var