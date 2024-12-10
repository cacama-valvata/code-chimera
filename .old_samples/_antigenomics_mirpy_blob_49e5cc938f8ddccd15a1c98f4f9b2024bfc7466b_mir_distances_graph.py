import typing as t
from dataclasses import dataclass
from multiprocessing import Pool
from itertools import starmap
import igraph as ig
# pip install -e git+https://github.com/life4/textdistance.git#egg=textdistance
# pip install "textdistance[Hamming]"
import textdistance


class LvWrapper(object):
    def __call__(self, s1, s2) -> tuple[str, str, float]:
        return (s1, s2, textdistance.levenshtein(s1, s2))


class HWrapper(object):
    def __call__(self, s1, s2) -> tuple[str, str, float]:
        return (s1, s2, textdistance.hamming(s1, s2))


class EditDistances:
    def __init__(self,
                 seqs: t.Iterable[str], seqs2: t.Iterable[str] = None,
                 indels=False,
                 nproc: int = 8, chunk_sz: int = 4096):
        self.seqs = seqs
        self.seqs2 = seqs2
        self.indels = indels
        self.nproc = nproc
        self.chunk_sz = chunk_sz
        if self.indels:
            self.dfun = LvWrapper()
        else:
            self.dfun = HWrapper()

    def get_edges(self, threshold: float = 1) -> t.Iterable[tuple[str, str]]:
        if not self.seqs2:
            gen = ((s1, s2) for s1 in self.seqs for s2 in self.seqs if s1 > s2)
        else:
            gen = ((s1, s2) for s1 in self.seqs for s2 in self.seqs2)
        if self.nproc == 1:
            dist = starmap(self.dfun, gen)
        else:
            # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
            with Pool(self.nproc) as pool:
                dist = pool.starmap(self.dfun, gen, self.chunk_sz)
        return ((x[0], x[1]) for x in dist if x[2] <= threshold)


@dataclass
class SequenceVertex:
    seq: str
    degree: int
    cluster_id: int
    pos: tuple[float, float]


class SequenceGraph:
    def __init__(self, edges: t.Iterable[tuple[str, str]]):
        self.graph = ig.Graph.TupleList(edges)
        self.degree = ig.Graph.indegree(self.graph)
        self.clusters = self.graph.components()

    def do_layout(self, method='graphopt'):
        self.layout = self.graph.layout(method)

    def get_vertices(self) -> t.Iterable[SequenceVertex]:
        return (SequenceVertex(s,
                               self.degree[i],
                               self.clusters.membership[i],
                               self.layout.coords[i] if self.layout else (0, 0)
                               ) for (i, s) in enumerate(self.graph.vs()['name']))
