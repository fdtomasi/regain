#!/usr/bin/python
# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""regain setup script."""

import codecs
import os

from setuptools import find_packages, setup

from regain import __version__ as version

HERE = os.path.abspath(os.path.dirname(__file__))


def get_extra_requires(path, add_all=True):
    import re
    from collections import defaultdict

    with open(path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith("#"):
                tags = set()
                if ":" in k:
                    k, v = k.split(":")
                    tags.update(vv.strip() for vv in v.split(","))
                tags.add(re.split("[<=>]", k)[0])
                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps["all"] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


def read(*filenames, **kwargs):
    """
    Build an absolute path from ``*filenames``, and  return contents of
    resulting file.  Defaults to UTF-8 encoding.
    """
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for fl in filenames:
        with codecs.open(os.path.join(HERE, fl), "rb", encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


def read_requirements(filename):
    reqs_txt = read(filename)
    parsed = reqs_txt.split("\n")
    return [r for r in parsed if len(r) > 0 and not r.startswith("-")]


NAME = "regain"
PACKAGE_NAME = "regain"
PACKAGES = find_packages(exclude=["*.__old", "*.tests"])

setup(
    name=NAME,
    version=version,
    description=("REGAIN (Regularised Graph Inference)"),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Federico Tomasi",
    author_email="fdtomasi@gmail.com",
    maintainer="Federico Tomasi",
    maintainer_email="fdtomasi@gmail.com",
    url="https://github.com/fdtomasi/regain",
    download_url="https://github.com/fdtomasi/regain/archive/" "v%s.tar.gz" % version,
    keywords=["graph inference", "latent variables"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Natural Language :: English",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python",
    ],
    license="FreeBSD",
    packages=PACKAGES,
    include_package_data=True,
    # install_requires=read_requirements("requirements.txt"),
    # extras_require=dict(
    #     dev=read_requirements("requirements-optional.txt"),
    # ),
)
