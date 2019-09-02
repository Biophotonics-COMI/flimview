import sys
import os
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
prjdir = os.path.dirname(__file__)

def read(filename):
    return open(os.path.join(prjdir, filename)).read()

extra_link_args = []
libraries = []
library_dirs = []
include_dirs = []
exec(open('flimview/version.py').read())
setup(
    name='flimview',
    version=__version__,
    author='Matias Carrasco Kind , COMI Lab',
    author_email='mcarras2@illinois.edu',
    scripts=[],
    packages=['flimview'],
    license='License.txt',
    description='Python ligtweight flim image processing',
    long_description=read('README.md'),
    url='https://github.com/sahandha/eif',
    install_requires=["numpy","scipy", "pandas", "matplotlib", "sdtfile", "h5py", "tqdm"],
)
