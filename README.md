[![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)

# regain
Regularised graph inference across multiple time stamps, considering the influence of latent variables.
It inherits functionalities from the ![scikit-learn](https://github.com/scikit-learn/scikit-learn) package.

## Getting started
### Dependencies
regain requires:
- Python (>= 2.7 or >= 3.4)
- NumPy (>= 1.8.2)
- scikit-learn (>= 0.17)

To use the parameter selection via gaussian process optimisation, ![GPyOpt](https://github.com/SheffieldML/GPyOpt) is required.
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
conda install regain
```

If you'd like to install from source, or want to contribute to the project (e.g. by sending pull requests via github), read on. Clone the repository in GitHub and add it to your $PYTHONPATH.
```bash
git clone https://github.com/fdtomasi/regain.git
cd regain
python setup.py develop
```

## Citation
```latex
@{coming soon}
```
