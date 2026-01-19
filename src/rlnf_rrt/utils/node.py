class Node:
    def __init__(self, x, y, parent=None, cost=0):
        self.x, self.y = x, y
        self.parent = parent
        self.cost = cost
        self.children = []

    def is_same(self, other, eps=1e-6):
        return abs(self.x - other.x) < eps and abs(self.y - other.y) < eps