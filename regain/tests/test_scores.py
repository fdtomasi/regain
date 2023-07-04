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
"""Test scores module."""
import numpy as np
from numpy.testing import assert_equal
from sklearn.utils.extmath import fast_logdet

from regain import scores
from regain.math import batch_logdet
from regain.scores import log_likelihood_t


def test_fast_logdet():
    """Test fast_logdet"""
    A = np.diag(np.random.rand(3))
    logdet1 = batch_logdet(A)
    assert_equal(logdet1, fast_logdet(A))

    A = np.array([np.diag(np.random.rand(3)), np.diag(np.random.rand(3))])
    logdet1 = batch_logdet(A)
    assert_equal(logdet1, [fast_logdet(A[0]), fast_logdet(A[1])])


def test_log_likelihood():
    """Test log_likelihood function."""
    emp_cov = precision = np.identity(3)
    logl = scores.log_likelihood(emp_cov, precision)
    assert_equal(logl, -3.0)

    emp_cov = precision = np.array([np.identity(3), np.identity(3)])
    logl = scores.log_likelihood(emp_cov, precision)
    assert_equal(logl, np.array([-3.0, -3.0]))

    assert_equal(logl.sum(), log_likelihood_t(emp_cov, precision))
