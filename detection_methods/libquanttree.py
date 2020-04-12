
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn.decomposition import PCA


def reshape_data(data):
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    return data


class Histogram(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.pi_values = []
        self.ndata = []
        self.pc = []
        self.do_PCA = False

    @abstractmethod
    def find_bin(self, data):
        pass

    @abstractmethod
    def build_histogram(self, data, do_PCA=False):
        data = reshape_data(data)
        if do_PCA:
            self.pc = PCA(whiten=False)
            self.pc.fit(data)
            self.do_PCA = True
        else:
            self.do_PCA = False

    def transform_data(self, data):
        data = reshape_data(data)
        if self.do_PCA:
            data = self.pc.transform(data)
        return data


class QuantTree(Histogram):

    def __init__(self, pi_values):
        super().__init__()
        pi_values = np.array(pi_values)
        if pi_values.size == 1:
            nbins = pi_values
            self.pi_values = np.ones(nbins) / nbins
            self.is_unif = True
        else:
            self.pi_values = np.array(pi_values)
        self.leaves = []

    def build_histogram(self, data, do_PCA=False):
        super().build_histogram(data, do_PCA)
        data = self.transform_data(data)

        ndata = data.shape[0]
        ndim = data.shape[1]
        nbin = self.pi_values.size
        self.ndata = ndata

        # Each leaf is characterized by 3 numbers: 1) the dimension of the split that genrates the leaf, 2) the lower bound of the leaf, 3) the upper bound of the leaf
        self.leaves = np.ones(shape=(3, nbin))

        # set the limits of the available space in each dimension
        limits = np.ones((2, ndim))
        limits[0, :] = -np.inf
        limits[1, :] = np.inf

        # all samples are available
        available = [True] * ndata

        # iteratively generate the leaves
        for i_leaf in range(nbin - 1):
            # select a random components
            i_dim = np.random.randint(ndim)
            x_tilde = data[available, i_dim]

            # find the indices of the available samples
            idx = [i for i in range(len(available)) if available[i]]
            N_tilde = len(idx)

            # sort the samples
            idx_sorted = sorted(range(len(x_tilde)), key=x_tilde.__getitem__)
            x_tilde.sort()

            # compute p_tilde
            p_tilde = self.pi_values[i_leaf] / (1 - np.sum(self.pi_values[0:i_leaf]))
            L = int(np.round(p_tilde * N_tilde))

            # define the leaf
            if np.random.choice([True, False]):
                self.leaves[:, i_leaf] = [i_dim, limits[0, i_dim], x_tilde[L]]
                limits[0, i_dim] = x_tilde[L]
                idx_sorted = idx_sorted[0:L]
            else:
                self.leaves[:, i_leaf] = [i_dim, x_tilde[-L], limits[1, i_dim]]
                limits[1, i_dim] = x_tilde[-L]
                idx_sorted = idx_sorted[-L:]

            # remove the sample in the leaf from the available samples
            for i in idx_sorted:
                available[idx[i]] = False

        # define the last leaf with the remaining samples
        i_dim = np.random.randint(ndim)
        self.leaves[:, -1] = [i_dim, limits[0, i_dim], limits[1, i_dim]]

    def find_bin(self, data):
        data = self.transform_data(data)
        nu = data.shape[0]
        leaf = np.zeros(nu)
        nleaves = len(self.pi_values)
        for i_data in range(nu):
            x = data[i_data, :]
            for i_leaf in range(nleaves):
                if i_leaf == nleaves - 1:
                    leaf[i_data] = i_leaf
                if self.leaves[2, i_leaf] >= x[int(self.leaves[0, i_leaf])] > self.leaves[1, i_leaf]:
                    leaf[i_data] = i_leaf
                    break

        return leaf


class QuantTreeUnivariate(QuantTree):

    def __init__(self, pi_values):
        super().__init__(pi_values)

    def build_histogram(self, data, do_PCA=False):
        data = np.array(data).squeeze()

        ndata = len(data)
        self.ndata = ndata
        L_values = np.round(self.pi_values * ndata)
        L_acc = np.cumsum(L_values)
        L_acc = [np.int(i) for i in L_acc]

        x = np.sort(data)

        self.leaves = np.concatenate(([-np.inf], x[L_acc[:-1]], [np.inf]))

    def find_bin(self, data):
        data = np.array(data).squeeze()
        if len(data.shape) == 0:
            data = [data]
        leaf = np.array([np.sum(x > self.leaves) for x in data]) - 1
        return leaf


class ChangeDetectionTest:

    def __init__(self, model, nu, statistic):
        self.model = model
        self.statistic = statistic
        self.threshold = {}
        self.nu = nu

    def set_threshold(self, alpha, threshold):
        if isinstance(alpha, list):
            self.threshold.update((alpha[i], threshold[i]) for i in range(len(alpha)))
        else:
            self.threshold.update({alpha: threshold})

    def estimate_bootstrap_threshold(self, data, alpha, B):
        data = reshape_data(data)
        if not isinstance(alpha, list):
            alpha_values = [alpha]
        else:
            alpha_values = alpha
        ndata = data.shape[0]

        stats = []
        for i_B in range(B):
            W = data[np.random.choice(ndata, self.nu, replace=True), :]
            stats.append(self.statistic(self.model, W))

        stats.sort()
        stats.insert(0, stats[0]-1)
        threshold_values = [stats[np.int(np.floor((1 - alpha) * B))] for alpha in alpha_values]
        self.set_threshold(alpha_values, threshold_values)

        if len(alpha_values) == 1:
            threshold_values = threshold_values[0]
        return threshold_values

    @staticmethod
    def get_precomputed_quanttree_threshold(stat_name, alpha, nbin, ndata, nu, ):
        threshold = {('pearson', 0.001,  32,  4096,  64): 64,
                     ('pearson', 0.001, 128,  4096,  64): 192,
                     (     'tv', 0.001,  32,  4096,  64): 25,
                     (     'tv', 0.001, 128,  4096,  64): 43,
                     ('pearson', 0.001,  32, 16384, 256): 62.75,
                     ('pearson', 0.001, 128, 16384, 256): 187,
                     (     'tv', 0.001,  32, 16384, 256): 52,
                     (     'tv', 0.001, 128, 16384, 256): 85,
                     ('pearson',  0.01,  32,  4096,  64): 54,
                     ('pearson',  0.01, 128,  4096,  64): 172,
                     (     'tv',  0.01,  32,  4096,  64): 23,
                     (     'tv',  0.01, 128,  4096,  64): 42,
                     ('pearson',  0.01,  32, 16384, 256): 53.25,
                     ('pearson',  0.01, 128, 16384, 256): 171,
                     (     'tv',  0.01,  32, 16384, 256): 47,
                     (     'tv',  0.01, 128, 16384, 256): 81,
                     ('pearson',  0.05,  32,  4096,  64): 46,
                     ('pearson',  0.05, 128,  4096,  64): 156,
                     (     'tv',  0.05,  32,  4096,  64): 21,
                     (     'tv',  0.05, 128,  4096,  64): 41,
                     ('pearson',  0.05,  32, 16384, 256): 45.75,
                     ('pearson',  0.05, 128, 16384, 256): 157,
                     (     'tv',  0.05,  32, 16384, 256): 44,
                     (     'tv',  0.05, 128, 16384, 256): 78,
					 ('pearson',  0.05,  32,   512, 512): 90.625,
					 (     'tv',  0.05,  32,   512, 512): 87,
					 ('pearson',  0.05,  32,  1024, 512): 68.5,
					 (     'tv',  0.05,  32,  1024, 512): 75,
					 ('pearson',  0.05,  32,  2048, 512): 56.25,
					 (     'tv',  0.05,  32,  2048, 512): 69,
					 ('pearson',  0.05,  32,  4096, 512): 50.625, #50.375
					 (     'tv',  0.05,  32,  4096, 512): 65,
					 ('pearson',  0.05,  25,  2000, 500): 46.1,
					 (     'tv',  0.05,  25,  2000, 500): 61,
					 ('pearson',  0.05,  25,  3000, 500): 42.1999,
					 (     'tv',  0.05,  25,  3000, 500): 59,
					 ('pearson',  0.05,  25,  4000, 500): 40.5,
					 (     'tv',  0.05,  25,  4000, 500): 58,
					 ('pearson',  0.05,  25,  5000, 500): 39.8,
					 (     'tv',  0.05,  25,  5000, 500): 58,
                     ('pearson',  0.05,  40,   2000, 500): 68.47999999999999,
                     (     'tv',  0.05,  40,   2000, 500): 75,
                     ('pearson',  0.05,  60,   3000, 600): 94.19999999999999,
                     (     'tv',  0.05,  60,   3000, 600): 95.0,
                     # B = 5000
                     ('pearson',  0.05,  20,   2000, 200): 33.599999999999994,
                     (     'tv',  0.05,  20,   2000, 200): 33.0,
                     ('pearson',  0.05,  20,   3000, 200): 32.400000000000006,
                     (     'tv',  0.05,  20,   3000, 200): 33.0,
                     ('pearson',  0.05,  20,   4000, 200): 31.799999999999997,
                     (     'tv',  0.05,  20,   4000, 200): 32.0,
                     ('pearson',  0.05,  20,   5000, 200): 31.600000000000005,
                     (     'tv',  0.05,  20,   5000, 200): 32.0,
                     ('pearson',  0.05,  40,   2000, 200): 60.0,
                     (     'tv',  0.05,  40,   2000, 200): 44.0,
                     ('pearson',  0.05,  40,   3000, 200): 58.400000000000006,
                     (     'tv',  0.05,  40,   3000, 200): 43.0,
                     ('pearson',  0.05,  40,   4000, 200): 57.2,
                     (     'tv',  0.05,  40,   4000, 200): 43.0,
                     ('pearson',  0.05,  40,   5000, 200): 57.20000000000001,
                     (     'tv',  0.05,  40,   5000, 200): 42.0,
                     ('pearson',  0.05,  10,   500, 500): 33.4,
                     ('pearson',  0.05,  20,   1000, 1000): 59.96,
                     ('pearson',  0.05,  40,   2000, 2000): 109.48,
                     ('pearson',  0.05,  60,   3000, 3000): 156.08
					 }
		
		
		#('pearson',  0.05,  40,   2000, 1000): 58.5625,
		#(     'tv',  0.05,  40,   2000, 1000): 98.0,
		

        tau = threshold.get((stat_name, alpha, nbin, ndata, nu))
        return tau

    def estimate_quanttree_threshold(self, alpha, B=1000000):
        if not isinstance(alpha, list):
            alpha_values = (alpha)
        else:
            alpha_values = (alpha)

        histogram = QuantTreeUnivariate(self.model.pi_values)
        stats = []
        for i_B in range(B):
            data = np.random.uniform(0, 1, self.model.ndata)
            histogram.build_histogram(data)

            W = np.random.uniform(0, 1, self.nu)
            stats.append(self.statistic(histogram, W))

        stats.sort()
        stats.insert(0, stats[0] - 1)
        threshold_values = [stats[np.int(np.ceil((1-alpha) * B))] for alpha in [alpha_values]]
        self.set_threshold(alpha_values, threshold_values)

        if len([alpha_values]) == 1:
            threshold_values = threshold_values[0]
        return threshold_values

    def reject_null_hypothesis(self, W, alpha):
        y = self.statistic(self.model, W)
        return y > self.threshold[alpha], y


def tv_statistic(histogram, data):
    idx = histogram.find_bin(data)

    y_hat, _ = np.histogram(idx, bins=len(histogram.pi_values), range=(0, len(histogram.pi_values)-1))

    tv = 0.5 * np.sum(np.abs(histogram.pi_values * data.shape[0] - y_hat))

    return tv


def pearson_statistic(histogram, data):
    idx = histogram.find_bin(data)

    y_hat, _ = np.histogram(idx, bins=len(histogram.pi_values), range=(0, len(histogram.pi_values) - 1))
    y = histogram.pi_values * data.shape[0]
    pearson = np.sum(np.abs(y - y_hat) ** 2 / y)

    return pearson



