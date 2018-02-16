[![develstat](https://travis-ci.org/fdtomasi/regain.svg?branch=master)](https://travis-ci.org/fdtomasi/regain) [![covdevel](http://codecov.io/github/fdtomasi/regain/coverage.svg?branch=master)](http://codecov.io/github/fdtomasi/regain?branch=master) [![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause) [![PyPI](https://img.shields.io/pypi/v/regain.svg)](https://pypi.python.org/pypi/regain) [![Conda](https://img.shields.io/conda/v/fdtomasi/regain.svg)](https://anaconda.org/fdtomasi/regain) [![Python27](https://img.shields.io/badge/python-2.7-blue.svg)](https://badge.fury.io/py/regain) [![Python34](https://img.shields.io/badge/python-3.5-blue.svg)](https://badge.fury.io/py/regain) [![Requirements Status](https://requires.io/github/fdtomasi/regain/requirements.svg?branch=master)](https://requires.io/github/fdtomasi/regain/requirements/?branch=master)

# regain
Regularised graph inference across multiple time stamps, considering the influence of latent variables.
It inherits functionalities from the [scikit-learn](https://github.com/scikit-learn/scikit-learn) package.

## Getting started
### Dependencies
regain requires:
- Python (>= 2.7 or >= 3.5)
- NumPy (>= 1.8.2)
- scikit-learn (>= 0.17)

To use the parameter selection via gaussian process optimisation, [GPyOpt](https://github.com/SheffieldML/GPyOpt) is required.
You can install dependencies by running:
```bash
pip install -r requirements.txt
```

### Installation
The simplest way to install regain is using pip
```bash
pip install regain
```
or `conda`

```bash
conda install -c fdtomasi regain
```

If you'd like to install from source, or want to contribute to the project (e.g. by sending pull requests via github), read on. Clone the repository in GitHub and add it to your $PYTHONPATH.
```bash
git clone https://github.com/fdtomasi/regain.git
cd regain
python setup.py develop
```

## Quickstart
A simple example for how to use LTGL.
```python
import numpy as np
from regain.admm import LatentTimeGraphLasso
from regain.datasets import generate_dataset
from regain.utils import error_norm_time

np.random.seed(42)
data = generate_dataset(mode='l1l2', n_dim_lat=1, n_dim_obs=10)
X = data.data
theta = data.thetas

mdl = LatentTimeGraphLasso(max_iter=50).fit(X)
print("Error: %.2f" % error_norm_time(theta, mdl.precision_))
```
Note that the input of `LatentTimeGraphLasso` is a three-dimensional matrix with shape `(n_times, n_samples, n_dimensions)`.
If you have a single time (`n_times = 1`), ensure a `X = X.reshape(1, *X.shape)` before using `LatentTimeGraphLasso`, or, alternatively, use `LatentGraphLasso`.


## Citation
```latex
@ARTICLE{2018arXiv180203987T,
   author = {{Tomasi}, F. and {Tozzo}, V. and {Salzo}, S. and {Verri}, A.},
    title = "{Latent variable time-varying network inference}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1802.03987},
 primaryClass = "stat.ML",
 keywords = {Statistics - Machine Learning, Computer Science - Learning},
     year = 2018,
    month = feb,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180203987T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
