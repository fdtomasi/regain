# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Test datasets module."""
import numpy as np

from regain import datasets


def test_quickstart():
    """Test make_dataset as in quickstart."""
    np.random.seed(42)
    data = datasets.make_dataset(n_dim_lat=1, n_dim_obs=10)
    data.X, data.y, data.thetas


def test_make_dataset_gaussian():
    """Test default make_dataset with Gaussian distribution."""
    data = datasets.make_dataset()
    data.X, data.y, data.thetas


# def test_make_dataset_gaussian_my():
#     """Test Gaussian make_dataset (MY)."""
#     data = datasets.make_dataset(mode='my')
#     data.X, data.y, data.thetas


# def test_make_dataset_gaussian_mys():
#     """Test Gaussian make_dataset (MYS)."""
#     data = datasets.make_dataset(mode='mys')
#     data.X, data.y, data.thetas


def test_make_dataset_gaussian_sin():
    """Test Gaussian make_dataset (sin)."""
    data = datasets.make_dataset(mode='sin')
    data.X, data.y, data.thetas


def test_make_dataset_gaussian_fixed_sparsity():
    """Test Gaussian make_dataset (fixed_sparsity)."""
    data = datasets.make_dataset(mode='fixed_sparsity')
    data.X, data.y, data.thetas


def test_make_dataset_gaussian_sincos():
    """Test Gaussian make_dataset (sincos)."""
    data = datasets.make_dataset(mode='sincos')
    data.X, data.y, data.thetas


def test_make_dataset_gaussian_gp():
    """Test Gaussian make_dataset (gp)."""
    data = datasets.make_dataset(mode='gp')
    data.X, data.y, data.thetas


def test_make_dataset_gaussian_fede():
    """Test Gaussian make_dataset (fede)."""
    data = datasets.make_dataset(mode='fede')
    data.X, data.y, data.thetas


def test_make_dataset_gaussian_sklearn():
    """Test Gaussian make_dataset (sklearn)."""
    data = datasets.make_dataset(mode='sklearn')
    data.X, data.y, data.thetas


def test_make_dataset_gaussian_ma():
    """Test Gaussian make_dataset (ma)."""
    data = datasets.make_dataset(mode='ma')
    data.X, data.y, data.thetas


# def test_make_dataset_gaussian_mak():
#     """Test Gaussian make_dataset (mak)."""
#     data = datasets.make_dataset(mode='mak')
#     data.X, data.y, data.thetas


# def test_make_dataset_gaussian_ticc():
#     """Test Gaussian make_dataset (ticc)."""
#     data = datasets.make_dataset(mode='ticc', n_dim_lat=0)
#     data.X, data.y, data.thetas


# def test_make_dataset_ising():
#     """Test default make_dataset with Ising distribution."""
#     data = datasets.make_dataset(distribution='ising')
#     data.X, data.y, data.thetas


def test_make_dataset_poisson():
    """Test default make_dataset with Poisson distribution."""
    data = datasets.make_dataset(distribution='poisson', update_theta='l1')
    data.X, data.y, data.thetas
