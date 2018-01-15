#!/bin/bash
# this script uses the ANACONDA_TOKEN env var.
# to create a token:
# >>> anaconda login
# >>> anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/fdtomasi/$PACKAGENAME --scopes "api:write api:read"
set -e
ANACONDA_TOKEN="fd-61d74ee1-e208-4efe-a79d-80fa49b93e41"

echo "Converting conda package..."
conda convert --platform all $HOME/miniconda2/conda-bld/linux-64/regain-*.tar.bz2 --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload conda-bld/**/regain-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
# exit 0
