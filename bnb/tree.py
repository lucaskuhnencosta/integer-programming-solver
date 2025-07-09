import heapq
import itertools

class BranchAndBoundTree:
    def __init__(self):
        #Min-heap for best-bound search
        self.best_bound_queue = []
        self.dfs_stack=[]
        self.counter = itertools.count()  # Unique counter to break ties

    def __len__(self):
        #The number of unique nodes still in the tre
        return len(self.dfs_stack)

    def empty(self):
        return not self.dfs_stack

    def push(self, node):
        """
        Pushes a new node to BOTH data structures
        """

        if node.processed:
            return

        count = next(self.counter)
        heapq.heappush(self.best_bound_queue, (node.bound,count, node))
        self.dfs_stack.append(node)

    def pop_best_bound(self):
        """
        Pops the node with the best (lowest) bound from the heap
        Skips nodes that have already been processed via the DFS stack
        """
        while self.best_bound_queue:
            _bound,_count,node=heapq.heappop(self.best_bound_queue)
            if not node.processed:
                node.processed = True
                self.dfs_stack.remove(node)
                return node
        return None

    def pop_dfs(self):
        """
        Pops the most recently added node (LIFO)
        Skips nodes that have already been processed via the best-bound queue
        """
        while self.dfs_stack:
            node = self.dfs_stack.pop()
            if not node.processed:
                node.processed = True
                return node
        return None

    def get_best_bound(self):
        """
        Efficiently peeks at the best bound from the top of the heap.
        """
        # Clean up the heap from any already-processed nodes at the top
        while self.best_bound_queue and self.best_bound_queue[0][2].processed:
            heapq.heappop(self.best_bound_queue)

        if not self.best_bound_queue:
            return float('inf')
        return self.best_bound_queue[0][0]  # The best bound is always at the root of the heap

    def push_children(self,left_node,right_node):
        """
        Pushes the two children nodes to the tree
        For DFS, the child with the better (lower) bound is pushed last
        to ensure it is processed first (LIFO)
        """
        # Determine the order: better node is the one with the lower bound
        if left_node.bound <= right_node.bound:
            better_node, worse_node = left_node, right_node
        else:
            better_node, worse_node = right_node, left_node

        # Push the worse node first, then the better node.
        # This way, the better node is at the top of the DFS stack.
        if not worse_node.is_infeasible:
            self.push(worse_node)
        if not better_node.is_infeasible:
            self.push(better_node)
