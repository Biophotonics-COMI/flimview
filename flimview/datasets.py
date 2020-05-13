import pooch
import gzip
from pooch import Decompress
from pooch import HTTPDownloader
import os
from pooch import retrieve
from .version import __version__

GOODBOY = pooch.create(
    # Use the default cache folder for the OS
    path="data",
    # The remote data is on Github
    base_url="https://github.com/Biophotonics-COMI/flimview/raw/master/data/",
    registry={
        "epidermis.sdt.gz": "9cddf7548e0f463ccf12be45a9625500bab0d6d61c579a2b56378a1288634dbe",
        "macrophages.ptu.gz": "38ee764f0156d160bbc39a363c89d14da735c2cb06aa92a15d07f7fa8e17ad2f",
    },
)

def fetch_sdt():
    """
    Retrieve sdt file and write it to disk
    """
    # The file will be downloaded automatically the first time this is run.
    download = HTTPDownloader(progressbar=True)
    name = "epidermis.sdt.gz"
    fdata = GOODBOY.fetch(name, processor=Decompress(),  downloader=download)
    fname= os.path.join(GOODBOY.path, name)
    os.rename(fname+'.decomp', fname.replace('.gz',''))
    print('Done!')
    return

def fetch_ptu():
    """
    Retrieve ptu file and write it to disk
    """
    # The file will be downloaded automatically the first time this is run.
    download = HTTPDownloader(progressbar=True)
    name = "macrophages.ptu.gz"
    fdata = GOODBOY.fetch(name, processor=Decompress(), downloader=download)
    fname= os.path.join(GOODBOY.path, name)
    os.rename(fname+'.decomp', fname.replace('.gz',''))
    print('Done!')
    return
