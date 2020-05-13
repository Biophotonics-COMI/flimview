import pathlib
import sys
import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

extra_link_args = []
libraries = []
library_dirs = []
include_dirs = []
exec(open('flimview/version.py').read())
setup(
    name='flimview',
    version=__version__,
    description='A software framework to handle, visualize and analyze FLIM data',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Matias Carrasco Kind & COMI Lab',
    author_email='mcarras2@illinois.edu',
    scripts=[],
    packages=['flimview'],
    license='License.txt',
    url='https://github.com/Biophotonics-COMI/flimview',
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "sdtfile",
        "pandas",
        "tqdm",
        "h5py",
        "Pillow",
        "pooch"
    ],
)
