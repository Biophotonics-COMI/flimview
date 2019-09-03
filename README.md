## flimview

### Conda Installation

First create a python 3 conda environment

    conda create -n flimview python=3 -y

To activate such environment

    conda activate flimview

To install the need requirements

    conda install -y -c conda-forge jupyterlab ipywidgets nodejs

To enable the widgets

    jupyter labextension install @jupyter-widgets/jupyterlab-manager

and finally to install flimview from github

    pip install git+https://github.com/mgckind/flimview.git --upgrade



### Examples

Check the notebooks examples [here](notebooks/)

### Development

    pip install -e .
