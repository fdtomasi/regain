set -e

# Install via pip + pyproject.toml. --no-deps because conda manages the
# runtime requirements via meta.yaml; --no-build-isolation because the
# conda build env already has setuptools/wheel.
$PYTHON -m pip install . --no-deps --no-build-isolation -vv
