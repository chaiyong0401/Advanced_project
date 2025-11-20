import numpy as np
import heapq
from PIL import Image
import math
import os
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R  # ★ 자세(Rotation) 계산용
from scipy.spatial.transform import Slerp

# =============================================================================
# 1. A* Algorithm 
# =============================================================================
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0; self.h = 0; self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f

def heuristic(a, b): return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(maze_array, start, end):
    start_node = Node(None, start)
    end_node = Node(None, end)
    open_list = []; heapq.heappush(open_list, start_node)
    closed_set = set()
    rows, cols = maze_array.shape

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []
            curr = current_node
            while curr:
                path.append(curr.position)
                curr = curr.parent
            return path[::-1]

        for new_pos in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_pos = (current_node.position[0] + new_pos[0], current_node.position[1] + new_pos[1])
            if node_pos[0] > rows-1 or node_pos[0] < 0 or node_pos[1] > cols-1 or node_pos[1] < 0: continue
            if maze_array[node_pos[0]][node_pos[1]] != 0: continue
            if node_pos in closed_set: continue

            new_node = Node(current_node, node_pos)
            new_node.g = current_node.g + np.sqrt(new_pos[0]**2 + new_pos[1]**2)
            new_node.h = heuristic(new_node.position, end_node.position)
            new_node.f = new_node.g + new_node.h

            if any(child.position == new_node.position and child.g <= new_node.g for child in open_list): continue
            heapq.heappush(open_list, new_node)
    return None

# =============================================================================
# 2. Helpers
# =============================================================================
def _Ry(angle_deg):
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def _endpoint_local(side, center_t, boundary, length_m):
    L = float(length_m)
    if side == "left": return np.array([boundary - L, center_t, 0.0])
    elif side == "right": return np.array([boundary + L, center_t, 0.0])
    elif side == "top": return np.array([center_t, boundary + L, 0.0])
    elif side == "bottom": return np.array([center_t, boundary - L, 0.0])
    return np.array([0,0,0])

def find_opening_pixel(img_arr, side):
    H, W = img_arr.shape
    candidates = []
    if side == "left":
        for r in range(H):
            if img_arr[r, 0] == 0: candidates.append((r, 0))
    elif side == "right":
        for r in range(H):
            if img_arr[r, W-1] == 0: candidates.append((r, W-1))
    elif side == "top":
        for c in range(W):
            if img_arr[0, c] == 0: candidates.append((0, c))
    elif side == "bottom":
        for c in range(W):
            if img_arr[H-1, c] == 0: candidates.append((H-1, c))
    if not candidates: raise ValueError(f"No opening found on side {side}")
    return candidates[len(candidates)//2]

def inflate_walls(maze_img, mpp, buffer_m):
    if buffer_m <= 0: return maze_img.copy()
    buffer_px = int(buffer_m / mpp)
    wall_mask = maze_img > 128
    struct = binary_dilation(wall_mask, iterations=buffer_px)
    inflated = np.zeros_like(maze_img)
    inflated[struct] = 255
    return inflated



# =============================================================================
# 3. Orientation Helpers
# =============================================================================
def get_flat_orientation_quat():
    r = R.from_euler('x', 180, degrees=True)
    return r.as_quat() # [x, y, z, w]

def get_tilted_orientation_quat(tilt_deg_y):

    r_flat = R.from_euler('x', 180, degrees=True)
    r_tilt = R.from_euler('y', tilt_deg_y, degrees=True)
    
    r_final = r_tilt * r_flat
    return r_final.as_quat()

# =============================================================================
# 3. ★★★ Smoothing Function (New) ★★★
# =============================================================================
def smooth_orientation_transition(full_path, transition_idx, blend_steps):

    start_idx = max(0, transition_idx - blend_steps)
    end_idx = min(len(full_path) - 1, transition_idx + blend_steps)
    
    q_start = R.from_quat(full_path[start_idx][3:])
    q_end   = R.from_quat(full_path[end_idx][3:])
    
    key_rots = R.from_quat([q_start.as_quat(), q_end.as_quat()])
    slerp = Slerp([0, 1], key_rots)
    
    print(f"Smoothing orientation from index {start_idx} to {end_idx}...")

    for i in range(start_idx, end_idx + 1):
        ratio = (i - start_idx) / (end_idx - start_idx)
        q_interp = slerp([ratio])[0]
        
        full_path[i][3] = q_interp.as_quat()[0] # x
        full_path[i][4] = q_interp.as_quat()[1] # y
        full_path[i][5] = q_interp.as_quat()[2] # z
        full_path[i][6] = q_interp.as_quat()[3] # w

    return full_path

def generate_corridor_path(p_start, p_end, num_steps=100):
    path = []
    for i in range(num_steps):
        ratio = (i + 1) / (num_steps + 1) 
        pos = (1 - ratio) * np.array(p_start[:3]) + ratio * np.array(p_end[:3])
        quat = p_start[3:] 
        path.append(list(pos) + list(quat))
    return path

# =============================================================================
# 4. Main
# =============================================================================
def main():
    MPP = 0.001
    TILT_DEG = -45.0
    SAFETY_BUFFER_M = 0.01 # 1cm
    
    # Maze Specs
    W1_m, H1_m = 0.24, 0.36
    EXIT1_SIDE = "right"
    EXIT_CORRIDOR1_LEN = 0.06
    W2_m, H2_m = 0.24, 0.36
    ENTRANCE2_SIDE = "left"
    ENTRANCE_CORRIDOR2_LEN = 0.06
    
    # Load Images
    try:
        img1_raw = np.array(Image.open("maze_imgs/maze1.png").convert("L"))
        img2_raw = np.array(Image.open("maze_imgs/maze2.png").convert("L"))
    except FileNotFoundError:
        print("Run maze_create.py first!")
        return

    # Inflate Walls
    img1_safe = inflate_walls(img1_raw, MPP, SAFETY_BUFFER_M)
    img2_safe = inflate_walls(img2_raw, MPP, SAFETY_BUFFER_M)

    # Planning
    print("Planning Maze 1...")
    start1 = find_opening_pixel(img1_raw, "top")
    goal1 = find_opening_pixel(img1_raw, "right")
    img1_safe[start1] = 0; img1_safe[goal1] = 0 # Clear start/goal
    path1 = astar(img1_safe, start1, goal1)
    
    print("Planning Maze 2...")
    start2 = find_opening_pixel(img2_raw, "left")
    goal2 = find_opening_pixel(img2_raw, "top")
    img2_safe[start2] = 0; img2_safe[goal2] = 0
    path2 = astar(img2_safe, start2, goal2)
    
    if not path1 or not path2:
        print("Path planning failed.")
        return

    # --- Generate Full Path with Orientation ---
    full_path_data = [] # [x, y, z, qx, qy, qz, qw]

    # 1. Maze 1 (Flat)
    quat1 = get_flat_orientation_quat()
    quat2 = get_tilted_orientation_quat(TILT_DEG)
    
    for r, c in path1:
        x = (c - img1_raw.shape[1]/2.0) * MPP
        y = (img1_raw.shape[0]/2.0 - r) * MPP
        z = 0.0
        # 높이를 약간 띄워야 바닥에 안 긁힘 (예: 15mm)
        z_safe = z + 0.005
        full_path_data.append([x, y, z_safe, *quat1])

    p1_last = full_path_data[-1]

    # transition_index = len(full_path_data)

    # 2. Maze 2 (Tilted & Translated)
    # Calc Offset
    goal1_y = (img1_raw.shape[0]/2.0 - goal1[0]) * MPP
    p1_end = _endpoint_local(EXIT1_SIDE, goal1_y, W1_m/2.0, EXIT_CORRIDOR1_LEN)
    
    start2_y = (img2_raw.shape[0]/2.0 - start2[0]) * MPP
    p2_start = _endpoint_local(ENTRANCE2_SIDE, start2_y, -W2_m/2.0, ENTRANCE_CORRIDOR2_LEN)
    
    R2_mat = _Ry(TILT_DEG)
    pos2 = p1_end - (R2_mat @ p2_start)

    # Maze 2의 첫 번째 점 계산
    r0, c0 = path2[0]
    x_local = (c0 - img2_raw.shape[1]/2.0) * MPP
    y_local = (img2_raw.shape[0]/2.0 - r0) * MPP
    z_local = 0.005
    p2_first_world = (R2_mat @ np.array([x_local, y_local, z_local])) + pos2
    p2_first = list(p2_first_world) + list(quat2)

    # 복도 채우기 (약 100 스텝 = 1초 분량)
    corridor_path = generate_corridor_path(p1_last, p2_first, num_steps=100)
    full_path_data.extend(corridor_path)
    
    # 연결 부위 인덱스 (복도의 중간 지점)
    transition_index = len(full_path_data) - 50
    
    
    for r, c in path2:
        x_local = (c - img2_raw.shape[1]/2.0) * MPP
        y_local = (img2_raw.shape[0]/2.0 - r) * MPP
        z_local = 0.0
        # 로컬 프레임에서도 바닥에서 5mm 띄움
        z_local_safe = z_local + 0.005 
        
        p_local = np.array([x_local, y_local, z_local_safe])
        p_world = (R2_mat @ p_local) + pos2
        
        full_path_data.append([p_world[0], p_world[1], p_world[2], *quat2])

    # --- ★★★ Orientation Smoothing ★★★ ---
    BLEND_STEPS = 60 
    full_path_data = smooth_orientation_transition(full_path_data, transition_index, BLEND_STEPS)

    world_offset_pos = np.array([0.4,0.0,0.4])
    # Save
    with open("planned_path.txt", "w") as f:
        for p in full_path_data:
            x = p[0] + world_offset_pos[0]
            y = p[1] + world_offset_pos[1]
            z = p[2] + world_offset_pos[2]
            qx, qy, qz, qw = p[3], p[4], p[5], p[6]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            
    print(f"Saved {len(full_path_data)} waypoints (Pos + Quat) to planned_path.txt")

if __name__ == "__main__":
    main()