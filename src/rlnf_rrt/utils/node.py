from typing import Self

class Node:
    def __init__(self, x:int, y:int, parent:Self|None = None, cost:float =0, hcost:float =0):
        self.x:int = x
        self.y:int = y
        self.parent:Self|None = parent
        self.cost:float = cost
        self.hcost:float = hcost
        self.children:list[Self] = []

    def is_same(self, other:Self, eps:float=1e-6) -> bool:
        return abs(self.x - other.x) < eps and abs(self.y - other.y) < eps