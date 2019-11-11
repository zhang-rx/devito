from itertools import groupby

from devito.ir.clusters.cluster import ClusterGroup

__all__ = ['Queue', 'QueueCG']


class Queue(object):

    """
    A special queue to process objects in nested IterationSpaces based on
    a divide-and-conquer algorithm.

    Notes
    -----
    Subclasses must override :meth:`_callback`, which may get executed either
    before (fdta -- first divide then apply) or after (fatd -- first apply
    then divide) the divide phase of the algorithm.
    """

    def callback(self, *args):
        return self._callback(*args)

    def _callback(self, *args):
        raise NotImplementedError

    def process(self, elements):
        return self._process_fdta(elements, 1)

    def _process_fdta(self, elements, level, prefix=None):
        """
        fdta -> First Divide Then Apply
        """
        prefix = prefix or []

        # Divide part
        processed = []
        for pfx, g in groupby(elements, key=lambda i: i.itintervals[:level]):
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process_fdta(list(g), level + 1, pfx))

        # Apply callback
        processed = self.callback(processed, prefix)

        return processed

    def _process_fatd(self, elements, level):
        """
        fatd -> First Apply Then Divide
        """
        # Divide part
        processed = []
        for pfx, g in groupby(elements, key=lambda i: i.itintervals[:level]):
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Apply callback
                _elements = self.callback(list(g), pfx)
                # Recursion
                processed.extend(self._process_fatd(_elements, level + 1))

        return processed


class QueueCG(Queue):

    """
    A Queue operating on ClusterGroups, instead of Clusters.
    """

    def callback(self, cgroups, prefix):
        cgroups = self._callback(cgroups, prefix)
        cgroups = [ClusterGroup(cgroups, prefix)]
        return cgroups

    def process(self, clusters):
        cgroups = [ClusterGroup(c, c.itintervals) for c in clusters]
        cgroups = self._process_fdta(cgroups, 1)
        clusters = ClusterGroup.concatenate(*cgroups)
        return clusters
