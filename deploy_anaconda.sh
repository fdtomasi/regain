#!/bin/bash
# this script uses the ANACONDA_TOKEN env var.
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/fdtomasi/$PACKAGENAME --scopes "api:write api:read"
set -e
ANACONDA_TOKEN="fd-61d74ee1-e208-4efe-a79d-80fa49b93e41"
CONDA_BLD_PATH=$HOME/miniconda/conda-bld

echo "Converting conda package..."
conda convert --platform all $CONDA_BLD_PATH/linux-64/regain-*.tar.bz2 --output-dir $CONDA_BLD_PATH/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload $CONDA_BLD_PATH/**/regain-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
# exit 0
