from collections import defaultdict, Counter
import collections

def calcu_prec(bins):
    
    help_intersections = dict()
    help_intersections = collections.defaultdict(set)
    
    for k, v in bins.intersectionsof.items():
        for cluster in v:
            help_intersections[cluster].add(k)
    col_cnt = 0
    
    for k, v in help_intersections.items():
        t = 0
        for genome in v:
            cnt = 0
            for contig in genome.contigs:
                if contig in bins.contigsof[k]:
                    cnt = cnt + 1
            t = max(t, cnt)
        col_cnt = col_cnt + t
        
    return col_cnt / bins.ncontigs

def calcu_recall(bins):
    
    row_cnt = 0
    
    for k, v in bins.intersectionsof.items():
        t = 0
        for cluster in v:
            cnt = 0
            for contig in bins.contigsof[cluster]:
                if contig in k.contigs:
                    cnt = cnt + 1
            t = max(t, cnt)
        row_cnt = row_cnt + t
        
    return row_cnt / bins.ncontigs

def calcu_f1score(bins):
    
    precision = calcu_prec(bins)
    recall = calcu_recall(bins)
    
    return 2 * recall * precision / (recall + precision)