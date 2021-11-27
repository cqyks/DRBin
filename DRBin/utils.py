#code from vamb
import os as _os
import gzip as _gzip
import bz2 as _bz2
import lzma as _lzma
import numpy as _np
import pyximport
pyximport.install()
from DRBin._DRBintools import _kmercounts, _fourmerfreq, zeros, _overwrite_matrix
import collections as _collections

def zscore(array, axis=None, inplace=False):

    if axis is not None and axis >= array.ndim:
        raise _np.AxisError('array only has {} axes'.format(array.ndim))

    if inplace and not _np.issubdtype(array.dtype, _np.floating):
        raise TypeError('Cannot convert a non-float array to zscores')

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)

    if axis is None:
        if std == 0:
            std = 1 # prevent divide by zero

    else:
        std[std == 0.0] = 1 # prevent divide by zero
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape

    if inplace:
        array -= mean
        array /= std
        return None
    else:
        return (array - mean) / std
    
class PushArray:
    __slots__ = ['data', 'capacity', 'length']

    def __init__(self, dtype, start_capacity=1<<16):
        self.capacity = start_capacity
        self.data = _np.empty(self.capacity, dtype=dtype)
        self.length = 0

    def __len__(self):
        return self.length

    def _setcapacity(self, n):
        self.data.resize(n, refcheck=False)
        self.capacity = n

    def _grow(self, mingrowth):
        """Grow capacity by power of two between 1/8 and 1/4 of current capacity, though at
        least mingrowth"""
        growth = max(int(self.capacity * 0.125), mingrowth)
        nextpow2 = 1 << (growth - 1).bit_length()
        self._setcapacity(self.capacity + nextpow2)

    def append(self, value):
        if self.length == self.capacity:
            self._grow(64)

        self.data[self.length] = value
        self.length += 1

    def extend(self, values):
        lenv = len(values)
        if self.length + lenv > self.capacity:
            self._grow(lenv)

        self.data[self.length:self.length+lenv] = values
        self.length += lenv

    def take(self):
        "Return the underlying array"
        self._setcapacity(self.length)
        return self.data

    def clear(self, force=False):
        "Empties the PushArray. If force is true, also truncates the underlying memory."
        self.length = 0
        if force:
            self._setcapacity(0)

def numpy_inplace_maskarray(array, mask):

    if len(mask) != len(array):
        raise ValueError('Lengths of array and mask must match')
    elif len(array.shape) != 2:
        raise ValueError('Can only take a 2 dimensional-array.')

    uints = _np.frombuffer(mask, dtype=_np.uint8)
    index = _overwrite_matrix(array, uints)
    array.resize((index, array.shape[1]), refcheck=False)
    return array

def torch_inplace_maskarray(array, mask):
    if len(mask) != len(array):
        raise ValueError('Lengths of array and mask must match')
    elif array.dim() != 2:
        raise ValueError('Can only take a 2 dimensional-array.')

    np_array = array.numpy()
    np_mask = _np.frombuffer(mask.numpy(), dtype=_np.uint8)
    index = _overwrite_matrix(np_array, np_mask)
    array.resize_((index, array.shape[1]))
    return array

class Reader:
    def __init__(self, filename, readmode='r'):
        if readmode not in ('r', 'rb'):
            raise ValueError("the Reader cannot write, set mode to 'r' or 'rb'")

        self.filename = filename
        self.readmode = readmode

    def __enter__(self):
        readmode = 'rt' if self.readmode == 'r' else self.readmode
        with open(self.filename, 'rb') as f:
            signature = f.peek(8)[:8]

        # Gzipped files begin with the two bytes 0x1F8B
        if tuple(signature[:2]) == (0x1F, 0x8B):
            self.filehandle = _gzip.open(self.filename, readmode)

        # bzip2 files begin with the signature BZ
        elif signature[:2] == b'BZ':
            self.filehandle = _bz2.open(self.filename, readmode)

        # .XZ files begins with 0xFD377A585A0000
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            self.filehandle = _lzma.open(self.filename, readmode)

        # Else we assume it's a text file.
        else:
            self.filehandle = open(self.filename, readmode)

        return self.filehandle

    def __exit__(self, type, value, traceback):
        self.filehandle.close()

class FastaEntry:
    """One single FASTA entry"""

    __slots__ = ['header', 'sequence']

    def __init__(self, header, sequence):
        if header[0] in ('>', '#') or header[0].isspace():
            raise ValueError('Header cannot begin with #, > or whitespace')
        if '\t' in header:
            raise ValueError('Header cannot contain a tab')

        self.header = header
        self.sequence = bytearray(sequence)

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return '>{}\n{}'.format(self.header, self.sequence.decode())

    def format(self, width=60):
        sixtymers = range(0, len(self.sequence), width)
        spacedseq = '\n'.join([self.sequence[i: i+width].decode() for i in sixtymers])
        return '>{}\n{}'.format(self.header, spacedseq)

    def __getitem__(self, index):
        return self.sequence[index]

    def __repr__(self):
        return '<FastaEntry {}>'.format(self.header)

    def kmercounts(self, k):
        if k < 1 or k > 10:
            raise ValueError('k must be between 1 and 10 inclusive')
        return _kmercounts(self.sequence, k)

    def fourmer_freq(self):
        return _fourmerfreq(self.sequence)

def byte_iterfasta(filehandle, comment=b'#'):

    linemask = bytes.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV',
                               b'ACGTTTNNNNNNNNNNNNNNNNNNNNN')

    # Skip to first header
    try:
        for linenumber, probeline in enumerate(filehandle):
            stripped = probeline.lstrip()
            if stripped.startswith(comment):
                pass

            elif probeline[0:1] == b'>':
                break

            else:
                raise ValueError('First non-comment line is not a Fasta header')

        else: # no break
            raise ValueError('Empty or outcommented file')

    except TypeError:
        errormsg = 'First line does not contain bytes. Are you reading file in binary mode?'
        raise TypeError(errormsg) from None

    header = probeline[1:-1].decode()
    buffer = list()

    # Iterate over lines
    for line in filehandle:
        linenumber += 1

        if line.startswith(comment):
            continue

        if line.startswith(b'>'):
            yield FastaEntry(header, b''.join(buffer))
            buffer.clear()
            header = line[1:-1].decode()

        else:
            # Check for un-parsable characters in the sequence
            stripped = line.translate(None, b'acgtuACGTUswkmyrbdhvnSWKMYRBDHVN \t\n')
            if len(stripped) > 0:
                bad_character = chr(stripped[0])
                raise ValueError("Non-IUPAC DNA in line {}: '{}'".format(linenumber + 1,
                                                                         bad_character))
            masked = line.translate(linemask, b' \t\n')
            buffer.append(masked)

    yield FastaEntry(header, b''.join(buffer))

def write_clusters(filehandle, clusters, max_clusters=None, min_size=1,
                 header=None, rename=True):
    if not hasattr(filehandle, 'writable') or not filehandle.writable():
        raise ValueError('Filehandle must be a writable file')

    if isinstance(clusters, dict):
        clusters = clusters.items()

    if max_clusters is not None and max_clusters < 1:
        raise ValueError('max_clusters must None or at least 1, not {}'.format(max_clusters))

    if header is not None and len(header) > 0:
        if '\n' in header:
            raise ValueError('Header cannot contain newline')

        if header[0] != '#':
            header = '# ' + header

        print(header, file=filehandle)

    clusternumber = 0
    ncontigs = 0

    for clustername, contigs in clusters:
        if len(contigs) < min_size:
            continue

        if rename:
            clustername = 'cluster_' + str(clusternumber + 1)

        for contig in contigs:
            print(clustername, contig, sep='\t', file=filehandle)
        filehandle.flush()

        clusternumber += 1
        ncontigs += len(contigs)

        if clusternumber == max_clusters:
            break

    return clusternumber, ncontigs

def read_clusters(filehandle, min_size=1):
    contigsof = _collections.defaultdict(set)

    for line in filehandle:
        stripped = line.strip()

        if not stripped or stripped[0] == '#':
            continue

        clustername, contigname = stripped.split('\t')
        contigsof[clustername].add(contigname)

    contigsof = {cl: co for cl, co in contigsof.items() if len(co) >= min_size}

    return contigsof

def loadfasta(byte_iterator, keep=None, comment=b'#', compress=False):

    entries = dict()

    for entry in byte_iterfasta(byte_iterator, comment=comment):
        if keep is None or entry.header in keep:
            if compress:
                entry.sequence = bytearray(_gzip.compress(entry.sequence))

            entries[entry.header] = entry

    return entries

def validate_input_array(array):
    "Returns array similar to input array but C-contiguous and with own data."
    if not array.flags['C_CONTIGUOUS']:
        array = _np.ascontiguousarray(array)
    if not array.flags['OWNDATA']:
        array = array.copy()

    assert (array.flags['C_CONTIGUOUS'] and array.flags['OWNDATA'])
    return array

def write_bins(directory, bins, fastadict, compressed=False, maxbins=250):

    # Safety measure so someone doesn't accidentally make 50000 tiny bins
    # If you do this on a compute cluster it can grind the entire cluster to
    # a halt and piss people off like you wouldn't believe.
    if maxbins is not None and len(bins) > maxbins:
        raise ValueError('{} bins exceed maxbins of {}'.format(len(bins), maxbins))

    # Check that the directory is not a non-directory file,
    # and that its parent directory indeed exists
    abspath = _os.path.abspath(directory)
    parentdir = _os.path.dirname(abspath)

    if parentdir != '' and not _os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    if _os.path.isfile(abspath):
        raise NotADirectoryError(abspath)

    # Check that all contigs in all bins are in the fastadict
    allcontigs = set()

    for contigs in bins.values():
        allcontigs.update(set(contigs))

    allcontigs -= fastadict.keys()
    if allcontigs:
        nmissing = len(allcontigs)
        raise IndexError('{} contigs in bins missing from fastadict'.format(nmissing))

    # Make the directory if it does not exist - if it does, do nothing
    try:
        _os.mkdir(directory)
    except FileExistsError:
        pass
    except:
        raise

    # Now actually print all the contigs to files
    for binname, contigs in bins.items():
        filename = _os.path.join(directory, binname + '.fna')

        with open(filename, 'w') as file:
            for contig in contigs:
                entry = fastadict[contig]

                if compressed:
                    uncompressed = bytearray(_gzip.decompress(entry.sequence))
                    entry = FastaEntry(entry.header, uncompressed)

                print(entry.format(), file=file)

def read_npz(file):
    npz = _np.load(file)
    array = npz['arr_0']
    npz.close()

    return array

def write_npz(file, array):
    _np.savez_compressed(file, array)

def filtercontigs(infile, outfile, minlength=2000):

    fasta_entries = byte_iterfasta(infile)

    for entry in fasta_entries:
        if len(entry) > minlength:
            print(entry.format(), file=outfile)

def concatenate_fasta(outpath, inpaths, minlength=2000):

    with open(outpath, "w") as outfile:
        for (inpathno, inpath) in enumerate(inpaths):

            with open(inpath, "rb") as infile:
                identifiers = set()
                entries = byte_iterfasta(infile)
                for entry in entries:

                    if len(entry) < minlength:
                        continue

                    header = entry.header

                    identifier = header.split()[0]
                    if identifier in identifiers:
                        raise ValueError("Multiple sequences have identifier {}"
                                         "in file {}".format(identifier, inpath))
                    newheader = "S{}C{}".format(inpathno + 1, identifier)
                    entry.header = newheader
                    print(entry.format(), file=outfile)

def _hash_refnames(refnames):
    "Hashes an iterable of strings of reference names using MD5."
    hasher = _md5()
    for refname in refnames:
        hasher.update(refname.encode().rstrip())

    return hasher.digest()

def _load_jgi(filehandle, minlength, refhash):
    "This function can be merged with load_jgi below in the next breaking release (post 3.0)"
    header = next(filehandle)
    fields = header.strip().split('\t')
    if not fields[:3] == ["contigName", "contigLen", "totalAvgDepth"]:
        raise ValueError('Input file format error: First columns should be "contigName,"'
        '"contigLen" and "totalAvgDepth"')

    columns = tuple([i for i in range(3, len(fields)) if not fields[i].endswith("-var")])
    array = PushArray(_np.float32)
    identifiers = list()

    for row in filehandle:
        fields = row.split('\t')
        # We use float because very large numbers will be printed in scientific notation
        if float(fields[1]) < minlength:
            continue

        for col in columns:
            array.append(float(fields[col]))
        
        identifiers.append(fields[0])
    
    if refhash is not None:
        hash = _hash_refnames(identifiers)
        if hash != refhash:
            errormsg = ('JGI file has reference hash {}, expected {}. '
                        'Verify that all BAM headers and FASTA headers are '
                        'identical and in the same order.')
            raise ValueError(errormsg.format(hash.hex(), refhash.hex()))

    result = array.take()
    result.shape = (len(result) // len(columns), len(columns))
    return validate_input_array(result)

def load_jgi(filehandle):
    return _load_jgi(filehandle, 0, None)

def _split_bin(binname, headers, separator, bysample=_collections.defaultdict(set)):
    "Split a single bin by the prefix of the headers"

    bysample.clear()
    for header in headers:
        if not isinstance(header, str):
            raise TypeError('Can only split named sequences, not of type {}'.format(type(header)))

        sample, _sep, identifier = header.partition(separator)

        if not identifier:
            raise KeyError("Separator '{}' not in sequence label: '{}'".format(separator, header))

        bysample[sample].add(header)

    for sample, splitheaders in bysample.items():
        newbinname = "{}{}{}".format(sample, separator, binname)
        yield newbinname, splitheaders

def _binsplit_generator(cluster_iterator, separator):
    "Return a generator over split bins with the function above."
    for binname, headers in cluster_iterator:
        for newbinname, splitheaders in _split_bin(binname, headers, separator):
            yield newbinname, splitheaders

def binsplit(clusters, separator):
    if iter(clusters) is clusters: # clusters is an iterator
        return _binsplit_generator(clusters, separator)

    elif isinstance(clusters, dict):
        return dict(_binsplit_generator(clusters.items(), separator))

    else:
        raise TypeError("clusters must be iterator of pairs or dict")
#end code from vamb

def get_contigs_with_marker_genes(output, 
                                    mg_length_threshold, 
                                    contig_lengths, 
                                    min_length):

    marker_contigs = {}
    marker_contig_counts = {}
    contig_markers = {}

    with open(f"{output}/contigs.hmmout", "r") as myfile:
        for line in myfile.readlines():
            if not line.startswith("#"):
                strings = line.strip().split()

                contig = strings[0]

                # Marker gene name
                marker_gene = strings[3]

                # Marker gene length
                marker_gene_length = int(strings[5])

                # Mapped marker gene length
                mapped_marker_length = int(strings[16]) - int(strings[15])

                # Get contig name
                name_strings = contig.split("_")
                name_strings = name_strings[:len(name_strings)-3]
                contig_name = "_".join(name_strings)

                contig_length = contig_lengths[contig_name]

                if contig_length >= min_length and mapped_marker_length > marker_gene_length*mg_length_threshold:

                    marker_repeated_in_contig = False

                    # Get marker genes in each contig
                    if contig_name not in contig_markers:
                        contig_markers[contig_name] = [marker_gene]
                    else:
                        if marker_gene not in contig_markers[contig_name]:
                            contig_markers[contig_name].append(marker_gene)

                    # Get contigs containing each marker gene
                    if marker_gene not in marker_contigs:
                        marker_contigs[marker_gene] = [contig_name]
                    else:
                        if contig_name not in marker_contigs[marker_gene]:
                            marker_contigs[marker_gene].append(contig_name)
                        else:
                            marker_repeated_in_contig = True

                    # Get contig counts for each marker
                    if marker_gene not in marker_contig_counts:
                        marker_contig_counts[marker_gene] = 1
                    else:
                        if not marker_repeated_in_contig:
                            marker_contig_counts[marker_gene] += 1

    return marker_contigs, marker_contig_counts, contig_markers