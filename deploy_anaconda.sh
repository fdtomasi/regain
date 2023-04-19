#!/bin/bash
# this script uses the ANACONDA_TOKEN env var.
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/fdtomasi/$PACKAGENAME --scopes "api:write api:read"
set -e
CONDA_BLD_PATH=$HOME/miniconda/conda-bld
ANACONDA_TOKEN="fd-6b5824f1-a585-45f5-bed0-6130e1b2c5df"

echo "Converting conda package..."
conda convert --platform all $CONDA_BLD_PATH/linux-64/regain-*.tar.bz2 --output-dir $CONDA_BLD_PATH/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload $CONDA_BLD_PATH/**/regain-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
# exit 0
