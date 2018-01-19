# pip install .
conda install -q python=$TRAVIS_PYTHON_VERSION conda-build jinja2 anaconda-client

# install additional packages with conda and pip
pip install -r requirements.txt

conda build tools

conda install --use-local regain
