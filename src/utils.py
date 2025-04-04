__all__ = ['open_directories', 'get_tchain', 'build_pandas']

import os
import glob
import ROOT

import numpy as np
import pandas as pd 

from typing import Iterable, List, Union, Dict, Iterator, Any, Tuple

def open_directories(
        paths: Iterable[str],
        file_ext: str,
        dev: bool = False) -> Iterator[str]:
    """
    Generator that opens all directories in an iterator for
    a specific file extension. This is useful for script cases where
    an user can pass a mix of directories and filepaths.

    Parameters
    ----------
    paths : Iterable[str]
        Iterable with file or dir paths to look for files with file_ext
    file_ext : str
        Te desired file extension to look for
    dev: bool
        If True, the function will yield just the first file found

    Yields
    ------
    str
        The path to a file

    Raises
    ------
    ValueError
        Raised if there is a file that doesn not have file_ext as its extension
    """
    for i, ipath in enumerate(paths):
        if os.path.isdir(ipath):
            dir_paths = iglob(
                os.path.join(ipath, '**', f'*.{file_ext}'),
                recursive=True
            )
            for open_path in open_directories(dir_paths, file_ext):
                yield open_path
        elif ipath.endswith(f'.{file_ext}'):
            yield ipath
        else:
            raise ValueError(
                f'{ipath} does not have the expected {file_ext} extension'
            )
        if dev and i > 0:
            break


def get_tchain(
        filepaths: Iterable[str],
        treepath: str,
        dev: bool = False,
        sorted: bool = False,
        title: str = 'tchain') -> Tuple[Iterable[str], ROOT.TChain]:
    """
    Receives an iterable of files and directories and returns a TChain with all
    the trees from the .root files contained in that Iterable

    Parameters
    ----------
    filepaths : Iterable[str]
        Iterable containing paths to root files.
        Directories are opnened recursively for root files
    treepath : str
        Path for the tree inside the .root file
    dev : bool
        If true, loads only one file, for testing purposes only
    sorted : bool
        If true, the files are sorted before being added to the TChain
    title: str
        TChain title

    Returns
    -------
    Iterable[str]
        Iterable with all the files processed
    ROOT.TChain
        The TChain contianing all the root files TTrees
    """
    if isinstance(filepaths, str):
        raise TypeError(
            "filepaths should be an iterable of strings, not a string"
        )
    chain = ROOT.TChain(treepath, title)
    files = open_directories(filepaths, 'root')
    if sorted:
        files = np.sort(list(files))
    for filepath in files:
        chain.Add(filepath)
        if dev:
            single_file = filepath
            break
    if dev:
        return [single_file], chain
    else:
        if sorted:
            return files, chain
        else:
            return open_directories(filepaths, 'root'), chain


def build_pandas(filepaths: Iterable[str],
                 treepath: str,
                 output_path: str,
                 output_name: str,
                 definitions: Dict[str, str],
                ):

    _, tchain = get_tchain(filepaths, treepath, sorted=True)
    rdf = ROOT.RDataFrame(tchain)
    if definitions:
        for name, op in definitions.items():
            rdf = rdf.Define(name, op)
    
    numpy_dict = rdf.AsNumpy()
    
    if definitions:
        for name, op in definitions.items():
            if 'ring' in name:
                name = 'cl_rings'
            elif '_float' in name:
                name = name.replace('_float', '')
            if name in numpy_dict:
                if name == 'target':
                    continue
                else:
                    del numpy_dict[name]
    
    pdf = pd.DataFrame.from_dict(numpy_dict)
    pdf.to_parquet(os.path.join(output_path, output_name))
    return pdf