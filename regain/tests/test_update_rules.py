"""Test update_rules module."""
from regain import update_rules


def test_update_rho():
    """Test update_rho function."""
    rho = update_rules.update_rho(1, 100, 0, mu=10, tau_inc=2, tau_dec=2)
    assert rho == 2
    rho = update_rules.update_rho(1, 0, 100, mu=10, tau_inc=2, tau_dec=2)
    assert rho == 0.5


def test_update_gamma():
    """Test update_gamma function."""
    gamma = update_rules.update_gamma(gamma=1, iteration=20, eps=1e-4)

    assert gamma == 0.5
