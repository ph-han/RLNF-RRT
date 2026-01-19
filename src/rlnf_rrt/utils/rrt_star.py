import time
import numpy as np
from collections import deque

from node import Node
from src.data_pipeline.utils import load_cspace_img_to_np

class RRTStar:
    def __init__(self, cspace_path, iter_num=500, clearance=1, step_size=1,
                 min_near_radius=15, gamma=20, robot_radius=1):
        self.cspace = load_cspace_img_to_np(cspace_path)
        self.iter_num = iter_num
        self.clearance = clearance
        self.step_size = step_size
        self.min_near_radius = min_near_radius
        self.gamma = gamma
        self.robot_radius = robot_radius

        self.is_goal = False
        self.graph = []
        self.graph_size = 0
        self.final_node = None

    def nearest_node(self, rand):
        distances = [np.hypot(node.x - rand.x, node.y - rand.y) for node in self.paths]
        return self.paths[np.argmin(distances)]

    def steer(self, from_node, to_node):
        dist = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
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
    
    def is_collision(self, node):
        if not (0 <= node.x < self.cspace.shape[0] and 0 <= node.y < self.cspace.shape[1]):
            return True

        x, y = int(node.x), int(node.y)
        pass

    def near_nodes(self, node):
        r = min(self.gamma * np.sqrt(np.log(self.graph_size) / self.graph_size),
                self.min_near_radius)
        
        node_index_list = []
        for i, node in enumerate(self.paths):
            if np.hypot(node.x - node.x, node.y - node.y) <= r:
                node_index_list.append(i)
        return node_index_list

    def choose_parent(self, new, nearest, near_node_idxs):
        candi_parent, min_cost = nearest, new.cost
        for nid in near_node_idxs:
            near = self.graph[nid]
            dist = np.hypot(new.x - near.x, new.y - near.y)
            if not self.is_collision(new) and near.cost + dist < min_cost:
                candi_parent, min_cost = near, near.cost + dist
        return candi_parent, min_cost
    

    def update_children_cost(self, parent):
        queue = deque([parent])

        while queue:
            node = queue.popleft()
            for child in node.children:
                child.cost = node.cost + np.hypot(child.x - node.x, child.y - node.y)
                queue.append(child)

    def rewire(self, new, parent, near_node_idxs):
        for nid in near_node_idxs:
            near = self.graph[nid]
            if near is parent: continue

            dist = np.hypot(new.x - near.x, new.y - near.y)
            if new.cost + dist < near.cost and not self.is_collision(near):
                if near.parent:
                    near.parent.children.remove(near)
                near.parent = new
                near.cost = new.cost + dist
                new.children.append(near)
                self.update_children_cost(near)

    def sampling_node(self):
        # unifrom sampling
        x = np.random.randint(0, self.cspace.shape[0])
        y = np.random.randint(0, self.cspace.shape[1])
        return Node(x, y)


    def planning(self, start_x, start_y, goal_x, goal_y, 
                 is_rewiring=True, is_break=False):
        # setting start node and goal node
        start = Node(start_x, start_y)
        goal = Node(goal_x, goal_y)
        self.graph.append(start)
        self.graph_size += 1

        cpu_start_time = time.perf_counter()
        for _ in range(self.iter_num):
            rand = self.sampling_node()
            nearest = self.nearest_node(rand)
            new = self.steer(nearest, rand)

            if self.is_collision(new): continue

            near_node_idxs = self.near_nodes(new)
            parent, min_cost = self.choose_parent(new, nearest, near_node_idxs)
            
            new.cost = min_cost
            new.parent = parent
            parent.children.append(new)

            self.graph.append(new)
            self.graph_size += 1

            if is_rewiring:
                self.rewire(new, parent, near_node_idxs)

            dist_to_goal = np.hypot(new.x - goal.x, new.y - goal.y)
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
        cpu_end_time = time.perf_counter()
        cpu_time = cpu_end_time - cpu_start_time
        return cpu_time
    
    def get_final_path(self):
        path = []
        node = self.graph[-1]
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]