class SolutionTracker:
    def __init__(self):
        self.best_obj = float("inf")
        self.best_solution = None

    def update(self, model):
        self.best_obj = model.ObjVal
        self.best_solution = model.X

    def should_prune(self, model):
        return model.ObjVal >= self.best_obj
