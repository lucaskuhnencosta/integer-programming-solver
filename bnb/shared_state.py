class SharedState:
    def __init__(self):
        self.best_solution = None
        self.best_cost = float('inf')
        self.has_solution = False

    def update_best_solution(self, solution, cost):
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_solution = solution
            self.has_solution = True
            return True
        return False
