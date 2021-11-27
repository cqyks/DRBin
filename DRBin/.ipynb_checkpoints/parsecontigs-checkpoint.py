#code from vamb
import sys as _sys
import os as _os
import numpy as _np
import gzip as _gzip
import DRBin.utils as _vambtools

def _read_contigs_preallocated(filehandle, minlength):
    n_entries = 0
    entries = _vambtools.byte_iterfasta(filehandle)

    for entry in entries:
         if len(entry) >= minlength:
            n_entries += 1

    tnfs = _np.empty((n_entries, 136), dtype=_np.float32)
    contignames = [None] * n_entries
    lengths = _np.empty(n_entries, dtype=_np.int)

    filehandle.seek(0) # Go back to beginning of file
    entries = _vambtools.byte_iterfasta(filehandle)
    entrynumber = 0
    for entry in entries:
        if len(entry) < minlength:
            continue

        tnfs[entrynumber] = entry.fourmer_freq()
        contignames[entrynumber] = entry.header
        lengths[entrynumber] = len(entry)

        entrynumber += 1

    return tnfs

def read_contigs(filehandle, minlength=100):

    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    return _read_contigs_preallocated(filehandle, minlength)