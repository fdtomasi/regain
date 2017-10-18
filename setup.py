#!/usr/bin/python
"""RGI   setup script.

Author: Federico Tomasi
Copyright (c) 2017, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""

from setuptools import setup, Extension
import numpy as np

# Package Version
from regain import __version__ as version
# alignment_module = Extension('icing.align.align',
#                              sources=['icing/align/alignment.c'])
setup(
    name='regain',
    version=version,

    description=('REGAIN (Regularised Graph Inference)'),
    long_description=open('README.md').read(),
    author='Federico Tomasi',
    author_email='federico.tomasi@dibris.unige.it',
    maintainer='Federico Tomasi',
    maintainer_email='federico.tomasi@dibris.unige.it',
    url='https://github.com/fdtomasi/regain',
    download_url='https://github.com/fdtomasi/regain/archive/'+version+'.tar.gz',
    keywords=['graph', 'inference'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license='FreeBSD',
    packages=['regain'],
    requires=['numpy (>=1.10.1)',
              'scipy (>=0.16.1)',
              'sklearn (>=0.17)',
              'matplotlib (>=1.5.1)',
              'seaborn (>=0.7.0)'],
    # scripts=['scripts/ici_run.py', 'scripts/ici_analysis.py'],
    # ext_modules=[ssk_module],
    include_dirs=[np.get_include()]
)
