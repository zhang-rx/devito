import tracemalloc

def print_memory():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)


def print_biggest_trace():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('traceback')

    # pick the biggest memory block
    stat = top_stats[0]
    print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
    for line in stat.traceback.format():
        print(line)


# ... run your application ...

from devito import *
from devito.types.basic import _SymbolCache

#tracemalloc.start()
tracemalloc.start(25)

grid = Grid(shape=(16, 16, 16))

queue = []
for i in range(2000):
    f = Function(name='f%d' % i, grid=grid)
    f.data
    queue.append(f)

print_memory()

print("Now killing `queue` and clearing devito cache...")
del queue
del f
clear_cache()

print_memory()
print('---')
print_biggest_trace()


# ... run your application ...
