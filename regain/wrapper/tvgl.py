import sys


def set_path(path):
    sys.path.append(path)


def tvgl(*args, **kwargs):
    import TVGL
    return TVGL.TVGL(*args, **kwargs)
