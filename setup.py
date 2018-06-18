#!/usr/bin/python
"""regain setup script.

Author: Federico Tomasi
Copyright (c) 2017, Federico Tomasi.
Licensed under the BSD 3-Clause License (see LICENSE.txt).
"""

from setuptools import setup, find_packages
# import numpy as np

# Package Version
from regain import __version__ as version
setup(
    name='regain',
    version=version,

    description=('REGAIN (Regularised Graph Inference)'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Federico Tomasi',
    author_email='federico.tomasi@dibris.unige.it',
    maintainer='Federico Tomasi',
    maintainer_email='federico.tomasi@dibris.unige.it',
    url='https://github.com/fdtomasi/regain',
    download_url='https://github.com/fdtomasi/regain/archive/'
                 'v%s.tar.gz' % version,
    keywords=['graph inference', 'latent variables'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python'
    ],
    license='FreeBSD',
    packages=find_packages(exclude=["*.__old", "*.tests"]),
    include_package_data=True,
    requires=['numpy (>=1.11)',
              'scipy (>=0.16.1,>=1.0)',
              'sklearn (>=0.17)',
              'six'],
    # scripts=['scripts/ici_run.py', 'scripts/ici_analysis.py'],
    # ext_modules=[ssk_module],
    # include_dirs=[np.get_include()]
)
