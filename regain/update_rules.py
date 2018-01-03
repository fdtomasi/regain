"""Update rules."""


def update_rho(rho, rnorm, snorm, iteration=None, mu=10, tau_inc=2, tau_dec=2):
    """See Boyd pag 20-21 for details.

    Parameters
    ----------
    rho : float
    """
    if rnorm > mu * snorm:
        return tau_inc * rho
    elif snorm > mu * rnorm:
        return rho / tau_dec
    return rho
