# Authors: Vivien Seguy <vivien.seguy@iip.ist.i.kyoto-u.ac.jp>
#          Rémi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

# sphinx_gallery_thumbnail_number = 2

import numpy as np
import matplotlib.pylab as pl
import ot

N = 2
d = 2

rng = np.random.default_rng(12345)


#Generate Matrices
mat = rng.random((30, 30))
const = np.ones_like(mat)

#Normalize so it sums to 1
I1 = mat / np.sum(mat)
I2 = const / np.sum(const)

c = 0.5 / np.sum(mat)

sz = I2.shape[0]
XX, YY = np.meshgrid(np.arange(sz), np.arange(sz))

x1 = np.stack((XX[I1 < c], YY[I1 < c]), 1) * 1.0
x2 = np.stack((XX[I2 < 0.5] + 80, YY[I2 < 0.5] + 10), 1) * 1.0

measures_locations = [x1, x2]
measures_weights = [ot.unif(x1.shape[0]), ot.unif(x2.shape[0])]

pl.figure(1, (12, 4))
pl.scatter(x1[:, 0], x1[:, 1], alpha=0.5)
pl.scatter(x2[:, 0], x2[:, 1], alpha=0.5)
pl.title('Distributions')


#%% Compute Free Support Barycenter
k = 20  # number of Diracs of the barycenter
X_init = np.random.normal(0., 1., (k, d))  # initial Dirac locations
b = np.ones((k,)) / k  # weights of the barycenter (it will not be optimized, only the locations are optimized)

X = ot.lp.free_support_barycenter(measures_locations, measures_weights, X_init, b)


#%% Plot the Barycenter
pl.figure(2, (8, 3))
pl.scatter(x1[:, 0], x1[:, 1], alpha=0.5)
pl.scatter(x2[:, 0], x2[:, 1], alpha=0.5)
pl.scatter(X[:, 0], X[:, 1], s=b * 1000, marker='s', label='2-Wasserstein barycenter')
pl.title('Data measures and their barycenter')
pl.legend(loc="lower right")
pl.show()