import time
import numpy as np
from collections import deque

from rlnf_rrt.utils.node import Node
from rlnf_rrt.data_pipeline.utils import load_cspace_img_to_np

class RRTStar:
    def __init__(self, cspace_path, iter_num:int=500, clearance:int=1, step_size:int=1,
                 min_near_radius:float=15, gamma:float=20, robot_radius:float=1):
        self.cspace:np.ndarray = load_cspace_img_to_np(cspace_path)
        self.iter_num:int = iter_num
        self.clearance:int = clearance
        self.step_size:int = step_size
        self.min_near_radius:float = min_near_radius
        self.gamma:float = gamma
        self.robot_radius:float = robot_radius

        self.is_goal:bool = False
        self.graph:list[Node] = []
        self.graph_size:int = 0
        self.final_node:Node | None = None

    def nearest_node(self, rand:Node) -> Node:
        distances:list[float] = [np.hypot(node.x - rand.x, node.y - rand.y) for node in self.graph]
        return self.graph[np.argmin(distances)]

    def steer(self, from_node:Node, to_node:Node) -> Node:
        dist:float = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        if dist <= self.step_size:
            to_node.parent = from_node
            cost = from_node.cost + dist
            to_node.cost = cost
            return to_node
        
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(from_node.x + self.step_size * np.cos(theta),
                        from_node.y + self.step_size * np.sin(theta),
                        parent=from_node, cost=from_node.cost + self.step_size)
        return new_node
    
    def is_collision(self, node:Node) -> bool:
        if not (0 <= node.x < self.cspace.shape[0] and 0 <= node.y < self.cspace.shape[1]):
            return True

        x, y = int(node.x), int(node.y)
        pass

    def near_nodes(self, node:Node) -> list:
        r:float = min(self.gamma * np.sqrt(np.log(self.graph_size) / self.graph_size),
                self.min_near_radius)
        
        node_index_list:list[int] = []
        for i, node in enumerate(self.paths):
            if np.hypot(node.x - node.x, node.y - node.y) <= r:
                node_index_list.append(i)
        return node_index_list

    def choose_parent(self, new:Node, nearest:Node, near_node_idxs:list) -> tuple[Node, float]:
        candi_parent: Node = nearest
        min_cost: float = new.cost

        for nid in near_node_idxs:
            near = self.graph[nid]
            dist = np.hypot(new.x - near.x, new.y - near.y)
            if not self.is_collision(new) and near.cost + dist < min_cost:
                candi_parent, min_cost = near, near.cost + dist
        return candi_parent, min_cost
    

    def update_children_cost(self, parent:Node) -> None:
        queue:deque[Node] = deque([parent])

        while queue:
            node = queue.popleft()
            for child in node.children:
                child.cost = node.cost + np.hypot(child.x - node.x, child.y - node.y)
                queue.append(child)

    def rewire(self, new:Node, parent:Node, near_node_idxs:list[int]) -> None:
        for nid in near_node_idxs:
            near:Node = self.graph[nid]
            if near is parent: continue

            dist: float = np.hypot(new.x - near.x, new.y - near.y)
            if new.cost + dist < near.cost and not self.is_collision(near):
                if near.parent:
                    near.parent.children.remove(near)
                near.parent = new
                near.cost = new.cost + dist
                new.children.append(near)
                self.update_children_cost(near)

    def sampling_node(self):
        # unifrom sampling
        x:int = np.random.randint(0, self.cspace.shape[0])
        y:int = np.random.randint(0, self.cspace.shape[1])
        return Node(x, y)


    def planning(self, start_x: int, start_y: int, goal_x: int, goal_y: int, 
                 is_rewiring:bool =True, is_break:bool =False) -> float:
        # setting start node and goal node
        start:Node = Node(start_x, start_y)
        goal:Node = Node(goal_x, goal_y)
        self.graph.append(start)
        self.graph_size += 1

        cpu_start_time:float = time.perf_counter()
        for _ in range(self.iter_num):
            rand:Node = self.sampling_node()
            nearest:Node = self.nearest_node(rand)
            new:Node = self.steer(nearest, rand)

            if self.is_collision(new): continue

            near_node_idxs:list[int] = self.near_nodes(new)

            parent:Node; min_cost:float
            parent, min_cost = self.choose_parent(new, nearest, near_node_idxs)
            
            new.cost = min_cost
            new.parent = parent
            parent.children.append(new)

            self.graph.append(new)
            self.graph_size += 1

            if is_rewiring:
                self.rewire(new, parent, near_node_idxs)

            dist_to_goal:float = np.hypot(new.x - goal.x, new.y - goal.y)
            if dist_to_goal <= self.robot_radius:
                self.final_node = Node((self.goal_x, self.goal_y), parent=new, cost=new.cost + dist_to_goal)
                if not new.is_same(self.final_node):
                    new.children.append(self.final_node)
                    self.graph.append(self.final_node)
                    self.graph_size += 1
                else:
                    self.final_node = new
                
                if is_break:
                    break
        cpu_end_time:float = time.perf_counter()
        cpu_time:float = cpu_end_time - cpu_start_time
        return cpu_time
    
    def get_final_path(self) -> list[Node]:
        path:list[Node] = []
        node:Node = self.graph[-1]
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]