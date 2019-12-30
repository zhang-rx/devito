from devito.ir.clusters import Queue

__all__ = ['Interchange']


class Interchange(Queue):

    def process(self, elements):
        return self._process_fatd(elements, 1)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        from IPython import embed; embed()

        return clusters
