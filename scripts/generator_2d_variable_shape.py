import os
import cv2
import csv
import random
import numpy as np

from rlnf_rrt.utils.astar import AStar



import numpy as np
import random

def _mask_circle(H, W, cx, cy, r):
    y, x = np.ogrid[:H, :W]
    return (x - cx)**2 + (y - cy)**2 <= r*r

def _mask_triangle(H, W, p1, p2, p3):
    y, x = np.ogrid[:H, :W]
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3

    def sign(px, py, ax, ay, bx, by):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)

    b1 = sign(x, y, x1, y1, x2, y2) < 0
    b2 = sign(x, y, x2, y2, x3, y3) < 0
    b3 = sign(x, y, x3, y3, x1, y1) < 0
    return (b1 == b2) & (b2 == b3)

def _tri_area(p1, p2, p3):
    # 2D polygon area (absolute)
    x1,y1=p1; x2,y2=p2; x3,y3=p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0

def generate_2d_grid_map(width:int, height:int,
                         obs_num=None,
                         margin:int=1,
                         rect_size=(5, 60),
                         circle_r=(3, 30),
                         tri_bbox=(10, 80),
                         tri_min_area=80):
    """
    - 맵 전체에 랜덤 배치 (균일)
    - margin: 테두리(외곽선) 보호용 여백
    - tri_bbox: 삼각형을 만들 '박스' 크기 범위(박스 안에서 꼭짓점 3개 랜덤)
    - tri_min_area: 너무 얇은 삼각형 필터링
    """
    grid = np.ones((height, width), dtype=np.uint8)

    # outline
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0

    if obs_num is None:
        obs_num = random.randint(3, 30)

    H, W = height, width

    for _ in range(obs_num):
        shape = random.choice(["rect", "circle", "tri"])

        # 실패하면 다시 뽑기(맵 전체 랜덤이지만, 유효한 샘플만 채택)
        for _try in range(50):
            if shape == "rect":
                w = random.randint(rect_size[0], min(rect_size[1], W - 2*margin - 1))
                h = random.randint(rect_size[0], min(rect_size[1], H - 2*margin - 1))
                x = random.randint(margin, W - margin - w - 1)
                y = random.randint(margin, H - margin - h - 1)
                grid[y:y+h, x:x+w] = 0
                break

            elif shape == "circle":
                r = random.randint(circle_r[0], min(circle_r[1], (min(H, W)//2) - margin - 1))
                cx = random.randint(margin + r, W - margin - r - 1)
                cy = random.randint(margin + r, H - margin - r - 1)
                grid[_mask_circle(H, W, cx, cy, r)] = 0
                break

            else:  # triangle
                bw = random.randint(tri_bbox[0], min(tri_bbox[1], W - 2*margin - 1))
                bh = random.randint(tri_bbox[0], min(tri_bbox[1], H - 2*margin - 1))
                x0 = random.randint(margin, W - margin - bw - 1)
                y0 = random.randint(margin, H - margin - bh - 1)

                # 박스 내부에서 꼭짓점 3개 랜덤
                p1 = (random.randint(x0, x0 + bw), random.randint(y0, y0 + bh))
                p2 = (random.randint(x0, x0 + bw), random.randint(y0, y0 + bh))
                p3 = (random.randint(x0, x0 + bw), random.randint(y0, y0 + bh))

                if _tri_area(p1, p2, p3) < tri_min_area:
                    continue  # 너무 얇으면 다시

                grid[_mask_triangle(H, W, p1, p2, p3)] = 0
                break

    return grid


def generate_2d_start_goal(map_info: np.ndarray, dist: float = 15.0, max_tries: int = 10_000):
    free = np.argwhere(map_info != 0)
    if free.shape[0] < 2:
        raise ValueError("Not enough free cells to sample start/goal.")

    for _ in range(max_tries):
        sy, sx = free[np.random.randint(len(free))]
        gy, gx = free[np.random.randint(len(free))]

        if (sx - gx) ** 2 + (sy - gy) ** 2 >= dist ** 2:
            return np.array([(sx, sy), (gx, gy)], dtype=int)

    raise RuntimeError(f"Failed to sample start/goal with dist>={dist} in {max_tries} tries.")

def generate_2d_gt_path(map_info:np.ndarray, start:tuple[int, int], goal:tuple[int, int], clr:int=1, ss:int=1) -> np.ndarray:
    thickness:int = 3

    astar = AStar(map_info, clr, ss)

    is_success = astar.planning(start[0], start[1], goal[0], goal[1])
    if is_success == False:
        return np.array([])
    
    opt_path:list[tuple[int, int]] = astar.get_final_path()
    wp_list:list[tuple[int, int]] = []
    for wp_idx in range(1, len(opt_path)):
        prev_wp = np.array(opt_path[wp_idx - 1])
        curr_wp = np.array(opt_path[wp_idx])

        dist = np.linalg.norm(curr_wp - prev_wp)
        if dist == 0: continue

        num_interp = int(dist / 0.5)
        for s in range(num_interp + 1):
            interp_wp = prev_wp + (curr_wp - prev_wp) * (s / num_interp)
            cx, cy = interp_wp
            
            for dx in range(-thickness, thickness + 1):
                for dy in range(-thickness, thickness + 1):
                    px = int(round(cx + dx))
                    py = int(round(cy + dy))
                    if 0 <= px < map_info.shape[1] and 0 <= py < map_info.shape[0]:
                        if map_info[py, px] != 0:
                            wp_list.append((px, py))
    
    return wp_list

def generate_2d_dataset(num_of_map_data: int = 100, num_of_start_goal: int = 10, split:str ="train") -> None:
    base_path = f"../data/{split}"
    for sub_dir in ["map", "start_goal", "gt_path"]:
        os.makedirs(os.path.join(base_path, sub_dir), exist_ok=True)
    
    meta_file_path = os.path.join(base_path, "meta.csv")
    with open(meta_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "map_file", "start_goal_file", "gt_path_file", "clearance", "step_size"])

        data_id = 0
        for i in range(num_of_map_data):
            grid_map = generate_2d_grid_map(224, 224)
            map_filename = f"map2d_{i:06d}.png"
            cv2.imwrite(os.path.join(base_path, "map", map_filename), grid_map * 255)

            for j in range(num_of_start_goal):
                start_goal_dist = random.randint(15, 100)
                start_goal = generate_2d_start_goal(grid_map, start_goal_dist)
                
                sg_idx = i * num_of_start_goal + j
                sg_filename = f"start_goal_2d_{sg_idx:06d}.npy"
                np.save(os.path.join(base_path, "start_goal", sg_filename), start_goal)

                for clr in [1, 2, 4, 6]:
                    for ss in [1, 2, 4, 6]:
                        gt_path = generate_2d_gt_path(grid_map, start_goal[0], start_goal[1], clr, ss)
                        
                        if len(gt_path) == 0:
                            continue
                        
                        path_filename = f"path_2d_{sg_idx:06d}_{clr}_{ss}.npy"
                        np.save(os.path.join(base_path, "gt_path", path_filename), gt_path)

                        writer.writerow([
                            data_id, 
                            map_filename, 
                            sg_filename, 
                            path_filename, 
                            clr, 
                            ss
                        ])
                        data_id += 1
            
            print(f"Progress: Map {i+1}/{num_of_map_data} done.")

    print(f"Dataset generation complete. Meta data saved to {meta_file_path}")


def main():
    grid_map:np.ndarray = generate_2d_grid_map(224, 224)
    print("map ok")
    start_goal:np.ndarray[tuple[int, int], tuple[int, int]] = generate_2d_start_goal(grid_map)
    print("start/goal ok")
    gt_path:np.ndarray = generate_2d_gt_path(grid_map, start_goal[0], start_goal[1])
    print("gt_path ok")

    # Test: save dataset
    np.save("../../data/train/start_goal/start_goal_2d_000001.npy", start_goal)
    np.save("../../data/train/gt_path/path_2d_000001.npy", gt_path)
    cv2.imwrite("../../data/train/map/map2d_000001.png", grid_map * 255)

    print("save done!")
    # Test: Visualization
    canvas:np.ndarray = cv2.cvtColor((grid_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if gt_path is not None and len(gt_path) > 1:
        pts = np.array(gt_path, dtype=np.int32).reshape((-1, 1, 2))
        
        # pts를 리스트([])에 담아서 전달
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 0), thickness=1)
    start:tuple[int, int]= tuple(map(int, start_goal[0]))
    goal:tuple[int, int] = tuple(map(int, start_goal[1]))
    cv2.circle(canvas, start, 3, (0, 0, 255), -1)
    cv2.circle(canvas, goal, 3, (255, 0, 0), -1)

    cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("canvas", 800, 800)
    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_2d_dataset(100, 10, "valid")
    # main()
