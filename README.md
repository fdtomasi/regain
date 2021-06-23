[![develstat](https://travis-ci.com/fdtomasi/regain.svg?branch=master)](https://travis-ci.org/fdtomasi/regain) [![covdevel](http://codecov.io/github/fdtomasi/regain/coverage.svg?branch=master)](http://codecov.io/github/fdtomasi/regain?branch=master) [![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause) [![PyPI](https://img.shields.io/pypi/v/regain.svg)](https://pypi.python.org/pypi/regain) [![Conda](https://img.shields.io/conda/v/fdtomasi/regain.svg)](https://anaconda.org/fdtomasi/regain) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/regain/) [![Requirements Status](https://requires.io/github/fdtomasi/regain/requirements.svg?branch=master)](https://requires.io/github/fdtomasi/regain/requirements/?branch=master)

# regain
Regularised graph inference across multiple time stamps, considering the influence of latent variables.
It inherits functionalities from the [scikit-learn](https://github.com/scikit-learn/scikit-learn) package.

## Getting started
### Dependencies
`REGAIN` requires:
- Python (>= 3.6)
- NumPy (>= 1.8.2)
- scikit-learn (>= 0.17)

You can install (required) dependencies by running:
```bash
pip install -r requirements.txt
```

To use the parameter selection via gaussian process optimisation, [skopt](https://scikit-optimize.github.io/) is required.

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
from regain.covariance import LatentTimeGraphicalLasso
from regain.datasets import make_dataset
from regain.utils import error_norm_time

np.random.seed(42)
data = make_dataset(n_dim_lat=1, n_dim_obs=3)
X = data.X
y = data.y
theta = data.thetas

mdl = LatentTimeGraphicalLasso(max_iter=50).fit(X, y)
print("Error: %.2f" % error_norm_time(theta, mdl.precision_))
```
**IMPORTANT**
We moved the API to be more consistent with `scikit-learn`.
Now the input of `LatentTimeGraphicalLasso` is a two-dimensional matrix `X` with shape `(n_samples, n_dimensions)`, where the belonging of samples to a different index (for example, a different time point) is indicated in `y`.


## Citation

`REGAIN` appeared in the following two publications.
For the `LatentTimeGraphicalLasso` please use

```latex
@inproceedings{Tomasi:2018:LVT:3219819.3220121,
 author = {Tomasi, Federico and Tozzo, Veronica and Salzo, Saverio and Verri, Alessandro},
 title = {Latent Variable Time-varying Network Inference},
 booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \&\#38; Data Mining},
 series = {KDD '18},
 year = {2018},
 isbn = {978-1-4503-5552-0},
 location = {London, United Kingdom},
 pages = {2338--2346},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3219819.3220121},
 doi = {10.1145/3219819.3220121},
 acmid = {3220121},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {convex optimization, graphical models, latent variables, network inference, time-series},
} 
```

and for the `TimeGraphicalLassoForwardBackward` plase use

```latex
@InProceedings{pmlr-v72-tomasi18a,
  title = 	 {Forward-Backward Splitting for Time-Varying Graphical Models},
  author = 	 {Tomasi, Federico and Tozzo, Veronica and Verri, Alessandro and Salzo, Saverio},
  booktitle = 	 {Proceedings of the Ninth International Conference on Probabilistic Graphical Models},
  pages = 	 {475--486},
  year = 	 {2018},
  editor = 	 {Kratochv\'{i}l, V\'{a}clav and Studen\'{y}, Milan},
  volume = 	 {72},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Prague, Czech Republic},
  month = 	 {11--14 Sep},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v72/tomasi18a/tomasi18a.pdf},
  url = 	 {http://proceedings.mlr.press/v72/tomasi18a.html},
  abstract = 	 {Gaussian graphical models have received much attention in the last years, due to their flexibility and expression power. However, the optimisation of such complex models suffer from computational issues both in terms of convergence rates and memory requirements. Here, we present a forward-backward splitting (FBS) procedure for Gaussian graphical modelling of multivariate time-series which relies on recent theoretical studies ensuring convergence under mild assumptions. Our experiments show that a FBS-based implementation achieves, with very fast convergence rates, optimal results with respect to ground truth and standard methods for dynamical network inference. Optimisation algorithms which are usually exploited for network inference suffer from drawbacks when considering large sets of unknowns. Particularly for increasing data sets and model complexity, we argue for the use of fast and theoretically sound optimisation algorithms to be significant to the graphical modelling community.}
}
```
