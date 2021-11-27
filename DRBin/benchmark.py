#code from vamb
import collections as _collections
from operator import add as _add
from itertools import product as _product
import sys as _sys
from math import sqrt as _sqrt
import DRBin.utils as _vambtools

class Contig:
    __slots__ = ['name', 'subject', 'start', 'end']

    def __init__(self, name, subject, start, end):
        self.name = name
        self.subject = subject
        self.start = start
        self.end = end

    @classmethod
    def subjectless(cls, name, length):
        return cls(name, name, 0, length)

    def __repr__(self):
        return 'Contig({}, subject={}, {}:{})'.format(self.name, self.subject, self.start, self.end)

    def __len__(self):
        return self.end - self.start

class Genome:
    __slots__ = ['name', 'breadth', 'contigs']

    def __init__(self, name):
        self.name = name
        self.contigs = set()
        self.breadth = 0

    def add(self, contig):
        self.contigs.add(contig)

    def remove(self, contig):
        self.contigs.remove(contig)

    def discard(self, contig):
        self.contigs.discard(contig)

    @property
    def ncontigs(self):
        return len(self.contigs)

    @staticmethod
    def getbreadth(contigs):
        bysubject = _collections.defaultdict(list)
        for contig in contigs:
            bysubject[contig.subject].append(contig)

        breadth = 0
        for contiglist in bysubject.values():
            contiglist.sort(key=lambda contig: contig.start)
            rightmost_end = float('-inf')

            for contig in contiglist:
                breadth += max(contig.end, rightmost_end) - max(contig.start, rightmost_end)
                rightmost_end = max(contig.end, rightmost_end)

        return breadth

    def update_breadth(self):
        self.breadth = self.getbreadth(self.contigs)

    def __repr__(self):
        return 'Genome({}, ncontigs={}, breadth={})'.format(self.name, self.ncontigs, self.breadth)

class Reference:
    def __init__(self, genomes, taxmaps=list()):
        self.genomes = dict() # genome_name : genome dict
        self.contigs = dict() # contig_name : contig dict
        self.genomeof = dict() # contig : genome dict

        self.taxmaps = taxmaps

        genomes_backup = list(genomes) if iter(genomes) is genomes else genomes

        # Check that there are no genomes with same name
        if len({genome.name for genome in genomes_backup}) != len(genomes_backup):
            raise ValueError('Multiple genomes with same name not allowed in Reference.')

        for genome in genomes_backup:
            self.add(genome)

        self.breadth = sum(genome.breadth for genome in genomes_backup)

    def load_tax_file(self, line_iterator, comment='#'):
        taxmaps = list()
        isempty = True

        for line in line_iterator:
            if line.startswith(comment):
                continue

            genomename, *clades = line[:-1].split('\t')

            if isempty:
                for i in clades:
                    taxmaps.append(dict())
                isempty = False
            previousrank = genomename
            for nextrank, rankdict in zip(clades, taxmaps):
                existing = rankdict.get(previousrank, nextrank)
                rankdict[previousrank] = nextrank
                previousrank = nextrank

        self.taxmaps = taxmaps

    @property
    def ngenomes(self):
        return len(self.genomes)

    @property
    def ncontigs(self):
        return len(self.contigs)

    def __repr__(self):
        ranks = len(self.taxmaps) + 1
        return 'Reference(ngenomes={}, ncontigs={}, ranks={})'.format(self.ngenomes, self.ncontigs, ranks)

    @staticmethod
    def _parse_subject_line(line):
        contig_name, genome_name, subject, start, end = line[:-1].split('\t')
        start = int(start)
        end = int(end) + 1 # semi-open interval used in internals, like range()
        contig = Contig(contig_name, subject, start, end)
        return contig, genome_name

    @staticmethod
    def _parse_subjectless_line(line):
        contig_name, genome_name, length = line[:-1].split('\t')
        length = int(length)
        contig = Contig.subjectless(contig_name, length)
        return contig, genome_name

    @classmethod
    def _parse_file(cls, filehandle, subjectless=False):
        function = cls._parse_subjectless_line if subjectless else cls._parse_subject_line

        genomes = dict()
        for line in filehandle:
            # Skip comments
            if line.startswith('#'):
                continue

            contig, genome_name = function(line)
            genome = genomes.get(genome_name)
            if genome is None:
                genome = Genome(genome_name)
                genomes[genome_name] = genome
            genome.add(contig)

        # Update all genomes
        genomes = list(genomes.values())
        for genome in genomes:
            genome.update_breadth()

        return genomes

    @classmethod
    def from_file(cls, filehandle, subjectless=False):
        genomes = cls._parse_file(filehandle, subjectless=subjectless)
        return cls(genomes)

    def add(self, genome):
        "Adds a genome to this Reference. If already present, do nothing."
        if genome.name not in self.genomes:
            self.genomes[genome.name] = genome
            for contig in genome.contigs:
                if contig.name in self.contigs:
                    raise KeyError("Contig name '{}' multiple times in Reference.".format(contig.name))

                self.contigs[contig.name] = contig
                self.genomeof[contig] = genome

    def remove(self, genome):
        "Removes a genome from this Reference, raising an error if it is not present."
        del self.genomes[genome.name]

        for contig in genome.contigs:
            del self.contigs[contig.name]
            del self.genomeof[contig]

    def discard(self, genome):
        "Remove a genome if it is present, else do nothing."
        if genome.name in self.genomes:
            self.remove(genome)

class Binning:
    _DEFAULTRECALLS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    _DEFAULTPRECISIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    @property
    def nbins(self):
        return len(self.contigsof)

    @property
    def ncontigs(self):
        return len(self.binof)

    def _iter_intersections(self, genome):
        # Get set of all binning bin names with contigs from that genome
        bin_names = set()
        for contig in genome.contigs:
            bin_name = self.binof.get(contig)
            if bin_name is None:
                continue
            elif isinstance(bin_name, set):
                bin_names.update(bin_name)
            else:
                bin_names.add(bin_name)

        for bin_name in bin_names:
            intersecting_contigs = genome.contigs.intersection(self.contigsof[bin_name])
            intersection = Genome.getbreadth(intersecting_contigs)
            yield bin_name, intersection

    def confusion_matrix(self, genome, bin_name):
        true_positives = self.intersectionsof[genome].get(bin_name, 0)
        false_positives = self.breadthof[bin_name] - true_positives
        false_negatives = genome.breadth - true_positives
        true_negatives = self.reference.breadth - false_negatives - false_positives + true_positives

        return true_positives, true_negatives, false_positives, false_negatives

    def mcc(self, genome, bin_name):
        tp, tn, fp, fn = self.confusion_matrix(genome, bin_name)
        mcc_num = tp * tn - fp * fn
        mcc_den = (tp + fp) * (tp + fn)
        mcc_den *= (tn + fp) * (tn + fn)
        return 0 if mcc_den == 0 else mcc_num / _sqrt(mcc_den)

    def f1(self, genome, bin_name):
        tp, tn, fp, fn = self.confusion_matrix(genome, bin_name)
        return 2*tp / (2*tp + fp + fn)

    def _getseen(self, recprecof):
        isseen = dict()
        for genome, _dict in recprecof.items():
            seen = 0
            for binname, (recall, precision) in _dict.items():
                for i, (min_recall, min_precision) in enumerate(_product(self.recalls, self.precisions)):
                    if recall < min_recall:
                        break

                    if precision >= min_precision:
                        seen |= 1 << i
            isseen[genome] = seen
        return isseen

    def _accumulate(self, seen, counts):
        "Given a 'seen' dict, make a dict of counts at each threshold level"
        nsums = (len(self.recalls) * len(self.precisions))
        sums = [0] * nsums
        for v in seen.values():
            for i in range(nsums):
                sums[i] += (v >> i) & 1 == 1

        for i, (recall, precision) in enumerate(_product(self.recalls, self.precisions)):
            counts[(recall, precision)] = sums[i]

    def _get_prec_rec_dict(self):
        recprecof = _collections.defaultdict(dict)
        for genome, intersectiondict in self.intersectionsof.items():
            for binname in intersectiondict:
                tp, tn, fp, fn = self.confusion_matrix(genome, binname)
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                recprecof[genome.name][binname] = (recall, precision)

        return recprecof

    def _getcounts(self):
        # One count per rank (+1 for inclusive "genome" rank)
        counts = [_collections.Counter() for i in range(len(self.reference.taxmaps) + 1)]
        recprecof = self._get_prec_rec_dict()
        seen = self._getseen(recprecof)
        # Calculate counts for each taxonomic level
        for counter, taxmap in zip(counts, self.reference.taxmaps):
            self._accumulate(seen, counter)
            newseen = dict()
            for clade, v in seen.items():
                newclade = taxmap[clade]
                newseen[newclade] = newseen.get(newclade, 0) | v
            seen = newseen

        self._accumulate(seen, counts[-1])

        return counts

    def __init__(self, contigsof, reference, recalls=_DEFAULTRECALLS,
              precisions=_DEFAULTPRECISIONS, checkpresence=True, disjoint=True,
              binsplit_separator=None, minsize=None, mincontigs=None):

        self.precisions = tuple(sorted(precisions))
        self.recalls = tuple(sorted(recalls))
        self.reference = reference

        self.contigsof = dict() # bin_name: {contigs} dict
        self.binof = dict() # contig: bin_name or {bin_names} dict
        self.breadthof = dict() # bin_name: int dict
        self._parse_bins(contigsof, checkpresence, disjoint, binsplit_separator, minsize, mincontigs)
        self.breadth = sum(self.breadthof.values())
        
        intersectionsof = dict()
        for genome in reference.genomes.values():
            intersectionsof[genome] = dict()
            for bin_name, intersection in self._iter_intersections(genome):
                intersectionsof[genome][bin_name] = intersection
        self.intersectionsof = intersectionsof

        # Set counts
        self.counters = self._getcounts()

    def _parse_bins(self, contigsof, checkpresence, disjoint, binsplit_separator, minsize, mincontigs):
        if binsplit_separator is not None:
            contigsof = _vambtools.binsplit(contigsof, binsplit_separator)

        if minsize is not None or mincontigs is not None:
            minsize = 1 if minsize is None else minsize
            mincontigs = 1 if mincontigs is None else mincontigs
            contigsof = filter_clusters(contigsof, self.reference, minsize, mincontigs, checkpresence=checkpresence)

        for bin_name, contig_names in contigsof.items():
            contigset = set()
            # This stores each contig by their true genome name.
            contigsof_genome = _collections.defaultdict(list)

            for contig_name in contig_names:
                contig = self.reference.contigs.get(contig_name)

                # Check that the contig is in the reference
                if contig is None:
                    if checkpresence:
                        raise KeyError('Contig {} not in reference.'.format(contig_name))
                    else:
                        continue

                # Check that contig is only present one time in input
                existing = self.binof.get(contig)
                if existing is None:
                    self.binof[contig] = bin_name
                else:
                    if disjoint:
                        #raise KeyError('Contig {} found in multiple bins'.format(contig_name))
                        continue
                    elif isinstance(existing, str):
                        self.binof[contig] = {existing, bin_name}
                    else:
                        self.binof[contig].add(bin_name)

                contigset.add(contig)
                genome = self.reference.genomeof[self.reference.contigs[contig_name]]
                contigsof_genome[genome.name].append(contig)

            self.contigsof[bin_name] = contigset

            breadth = 0
            for contigs in contigsof_genome.values():
                breadth += Genome.getbreadth(contigs)
            self.breadthof[bin_name] = breadth

    @classmethod
    def from_file(cls, filehandle, reference, recalls=_DEFAULTRECALLS,
                  precisions=_DEFAULTPRECISIONS, checkpresence=True, disjoint=True,
                  binsplit_separator=None, minsize=None, mincontigs=None):
        contigsof = dict()
        for line in filehandle:
            if line.startswith('#'):
                continue

            line = line.rstrip()
            bin_name, tab, contig_name = line.partition('\t')

            if bin_name not in contigsof:
                contigsof[bin_name] = [contig_name]
            else:
                contigsof[bin_name].append(contig_name)

        return cls(contigsof, reference, recalls, precisions, checkpresence, disjoint,
                   binsplit_separator, minsize, mincontigs)

    def print_matrix(self, rank, file=_sys.stdout):
        if rank >= len(self.counters):
            raise IndexError("Taxonomic rank out of range")

        print('\tRecall', file=file)
        print('Prec.', '\t'.join([str(r) for r in self.recalls]), sep='\t', file=file)

        for min_precision in self.precisions:
            row = [self.counters[rank][(min_recall, min_precision)] for min_recall in self.recalls]
            print(min_precision, '\t'.join([str(i) for i in row]), sep='\t', file=file)

    def __repr__(self):
        fields = (self.ncontigs, self.reference.ncontigs, hex(id(self.reference)))
        return 'Binning({}/{} contigs, ReferenceID={})'.format(*fields)

    def summary(self, precision=0.9, recalls=None):
        if recalls is None:
            recalls = self.recalls
        return [[counter[(recall, precision)] for recall in recalls] for counter in self.counters]

def filter_clusters(clusters, reference, minsize, mincontigs, checkpresence=True):
    filtered = dict()
    for binname, contignames in clusters.items():
        if len(contignames) < mincontigs:
            continue

        size = 0
        for contigname in contignames:
            contig = reference.contigs.get(contigname)

            if contig is not None:
                size += len(contig)
            elif checkpresence:
                raise KeyError('Contigname {} not in reference'.format(contigname))
            else:
                pass

        if size >= minsize:
            filtered[binname] = contignames.copy()

    return filtered
