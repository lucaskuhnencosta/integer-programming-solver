import heapq
import itertools

class BranchAndBoundTree:
    def __init__(self):
        self.queue = []
        self.counter = itertools.count()  # Unique counter to break ties

    def push(self, node):
        count = next(self.counter)
        heapq.heappush(self.queue, (node.bound,count, node))

    def pop(self,mode="best-first",last_node=None):
        return heapq.heappop(self.queue)[2]

    def empty(self):
        return len(self.queue) == 0

    def get_best_bound(self):
        return min(n.bound for _, _, n in self.queue) if self.queue else float("inf")

