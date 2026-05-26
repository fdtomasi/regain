# BSD 3-Clause License
# Copyright (c) 2019, regain authors
"""Smoke tests for regain.datasets.multi_class."""

import numpy as np

from regain.datasets.multi_class import (
    generate_multiple_class_dataset,
    make_multiclass_dataset,
)


def test_generate_multiple_class_dataset_gaussian_shapes():
    np.random.seed(0)
    res = generate_multiple_class_dataset(
        n_dim_obs=6,
        n_edges=2,
        probability=0.3,
        n_classes=3,
        _type="erdos-renyi",
        distribution="gaussian",
        random_state=0,
    )
    assert "gaussian" in res
    assert "binary" in res
    assert len(res["gaussian"]) == 3
    for k in res["gaussian"]:
        assert k.shape == (6, 6)


def test_generate_multiple_class_dataset_scale_free():
    np.random.seed(0)
    res = generate_multiple_class_dataset(
        n_dim_obs=5,
        n_classes=2,
        _type="scale-free",
        distribution="gaussian",
        random_state=0,
    )
    assert len(res["binary"]) == 2


def test_make_multiclass_dataset_returns_data_and_binary():
    np.random.seed(0)
    data, binary = make_multiclass_dataset(
        n_samples=10,
        n_dim_obs=5,
        n_classes=3,
        _type="erdos-renyi",
        distribution="gaussian",
        random_state=0,
    )
    assert "gaussian" in data
    assert len(binary) == 3
