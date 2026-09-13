"""Small streaming adapters for PyTorch's maintained Dataset/DataLoader APIs.

Only the three operations FAOD previously took from torchdata are implemented.
They preserve index order, lazy evaluation and per-worker stream construction.
"""
from itertools import zip_longest
from torch.utils.data import IterDataPipe, MapDataPipe


class MapToIter(IterDataPipe):
    def __init__(self, source: MapDataPipe):
        self.source = source

    def __iter__(self):
        for index in range(len(self.source)):
            yield self.source[index]

    def __len__(self):
        return len(self.source)


class Repeat(IterDataPipe):
    """Reiterate the source, without caching samples (unlike itertools.cycle)."""
    def __init__(self, source, count=None):
        if count is not None and count < 0:
            raise ValueError('count must be nonnegative or None')
        self.source, self.count = source, count

    def __iter__(self):
        iteration = 0
        while self.count is None or iteration < self.count:
            found = False
            for value in self.source:
                found = True
                yield value
            if not found:
                return
            iteration += 1

    def __len__(self):
        if self.count is None:
            raise TypeError('An infinite repeated stream has no length')
        return len(self.source) * self.count


class ZipperLongest(IterDataPipe):
    def __init__(self, *sources, fill_value=None):
        self.sources, self.fill_value = sources, fill_value

    def __iter__(self):
        yield from zip_longest(*self.sources, fillvalue=self.fill_value)

    def __len__(self):
        return max((len(source) for source in self.sources), default=0)
