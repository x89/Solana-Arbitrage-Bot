"""Setup script for shreddit.
"""
from setuptools import setup
from codecs import open
from os import path

VERSION = "6.0.7"
DESCRIPTION = " Remove your comment history on Reddit as deleting an account does not do so."

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding='utf-8') as filein:
    long_description = filein.read()

with open(path.join(here, "requirements.txt"), encoding='utf-8') as reqs_fp:
    requires = reqs_fp.read().splitlines()

setup(
    name="shreddit",
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    url="https://github.com/x89/Shreddit",
    author="David John",
    author_email="david@vaunt.eu",
    classifiers=["Development Status :: 4 - Beta",
                 "Intended Audience :: End Users/Desktop",
                 "License :: OSI Approved :: BSD License",
                 "Programming Language :: Python"],
    license="FreeBSD License",
    packages=["shreddit"],
    install_requires=requires,
    package_data={"shreddit": ["*.example"]},
    entry_points={
        "console_scripts": [
            "shreddit=shreddit.app:main"
        ]
    }
)
