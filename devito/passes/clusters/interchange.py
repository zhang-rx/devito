from devito.ir.clusters import Queue
from devito.ir.support import PARALLEL, PARALLEL_INDEP

__all__ = ['Interchange']


class Interchange(Queue):

    def process(self, elements):
        return self._process_fatd(elements, 1)

    def callback(self, clusters, prefix):
        if len(clusters) > 1:
            # => "imperfect loop nest"
            return clusters
        cluster = clusters[0]

        if not prefix:
            return [cluster]
        d = prefix[-1].dim

        #if PARALLEL not in cluster.properties[d]:
        #    # No hope if `d` itself isn't PARALLEL
        #    return [cluster]

        return clusters
