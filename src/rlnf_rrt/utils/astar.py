import numpy as np

from rlnf_rrt.utils.node import Node

class AStar:
    def __init__(self, map_info: np.ndarray, clearance:int, step_size:int, resolution:int = 1, robot_radius:float = 1):
        self.grid_map:np.ndarray = map_info
        self.width:int = map_info.shape[1]
        self.height:int = map_info.shape[0]
        self.clearance:int = clearance
        self.step_size:int = step_size
        self.resolution:int = resolution
        self.robot_radius:float = robot_radius
        self.motions:list[tuple[int, int, float]] = [
            (0, step_size, step_size),
            (step_size, 0, step_size),
            (0, -step_size, step_size),
            (-step_size, 0, step_size),
            (step_size, step_size, np.sqrt(2 * step_size ** 2)),
            (-step_size, step_size, np.sqrt(2 * step_size ** 2)),
            (step_size, -step_size, np.sqrt(2 * step_size ** 2)),
            (-step_size, -step_size, np.sqrt(2 * step_size ** 2)),
        ]
        self.is_success:bool = False
        self.endpoint:Node | None = None

    def _calc_grid_index(self, node:Node) -> int:
        return node.y * self.width + node.x
    
    def _verify_node(self, node: Node) -> bool:
        if not (0 <= node.x < self.width and 0 <= node.y < self.height):
            return False
        
        x, y = int(node.x), int(node.y)
        cl = self.clearance

        x_min, x_max = x - cl, x + cl + 1
        y_min, y_max = y - cl, y + cl + 1

        if x_min < 0 or x_max > self.width or y_min < 0 or y_max > self.height:
            return False

        if np.any(self.grid_map[y_min:y_max, x_min:x_max] == 0):
            return False

        return True

    def planning(self, start_x:int, start_y:int, goal_x:int, goal_y:int) -> bool:
        start:Node = Node(start_x, start_y, hcost=np.hypot(start_x - goal_x, start_y - goal_y))
        goal:Node = Node(goal_x, goal_y)

        open_set: dict[Node] = {}
        closed_set: dict[Node] = {}
        
        open_set[self._calc_grid_index(start)] = start

        while open_set:
            curr_id:int|None = min(open_set, key=lambda o: open_set[o].cost + open_set[o].hcost)
            if curr_id is None:
                return False
            
            curr:Node = open_set[curr_id]
            del open_set[curr_id]
            closed_set[curr_id] = curr

            if curr.is_same(goal, eps=self.clearance):
                goal.parent = curr
                self.endpoint = goal
                self.is_success = True
                break

            for motion in self.motions:
                dx, dy, cost = motion
                next:Node = Node(curr.x + dx, curr.y + dy, parent=curr, cost=curr.cost + cost)
                next.hcost = np.hypot(goal.x - next.x, goal.y - next.y)

                next_idx = self._calc_grid_index(next)

                if not self._verify_node(next):
                    continue
                
                if next_idx in closed_set:
                    continue

                if next_idx not in open_set or open_set[next_idx].cost > next.cost:
                    open_set[next_idx] = next
        
        return self.is_success
    
    def get_final_path(self) -> list[tuple[int, int]]:
        path:list[tuple[int, int]] = []

        if not self.is_success:
            print(f"No path found")
            return path
        
        node:Node = self.endpoint
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]