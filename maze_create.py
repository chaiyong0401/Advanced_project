# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
import random

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required. Install via: pip install pillow")


# ---------------------------------------------------------------------------
# 1) Utils: weighted integer pixel allocation / offset computation
# ---------------------------------------------------------------------------
def _alloc_sizes_by_weights(weights: List[float], interior_px: int) -> List[int]:
    """Distribute integer pixel sizes proportionally to weights (sum == interior_px). Ensure each entry >= 1 px."""
    if interior_px < len(weights):
        raise ValueError("Resolution is too low. Decrease meters_per_pixel or increase size.")
    S = float(sum(weights)) or float(len(weights))
    if S <= 0:
        weights = [1.0] * len(weights)
        S = float(len(weights))
    raw = [interior_px * (w / S) for w in weights]
    base = [max(1, int(v)) for v in raw]
    rem = interior_px - sum(base)
    residuals = [(raw[i] - int(raw[i]), i) for i in range(len(raw))]
    residuals.sort(reverse=True)
    i = 0
    while rem > 0 and i < len(residuals):
        idx = residuals[i][1]
        base[idx] += 1
        rem -= 1
        i += 1
    i = 0
    while rem < 0 and i < len(base):
        if base[i] > 1:
            base[i] -= 1
            rem += 1
        i += 1
    if sum(base) != interior_px:
        raise RuntimeError("Pixel allocation sum mismatch.")
    return base


def _compute_offsets_from_sizes(sizes_px: List[int], wall_px: int, total_px: int):
    """Compute each cell's top-left start and each wall-band start."""
    cell_tl = []
    wall_starts = [0]
    pos = 0
    for sz in sizes_px:
        cell_tl.append(pos + wall_px)
        pos += wall_px + sz
        wall_starts.append(pos)
    assert pos + wall_px == total_px, "Total pixel count mismatch."
    return np.array(cell_tl, dtype=int), np.array(wall_starts, dtype=int)


# ---------------------------------------------------------------------------
# 2) Maze generator in metric units (image: 0=free, 255=wall)
# ---------------------------------------------------------------------------
# The commented-out signature below is the previous version kept for reference.
# def generate_maze_metric(
#     rows: int,
#     cols: int,
#     *,
#     width_m: float,          # total width [m]
#     height_m: float,         # total height [m]
#     wall_thickness_m: float, # wall thickness [m] (horizontal cross-section)
#     wall_height_m: float,    # wall height [m] (vertical)
#     meters_per_pixel: float = 0.01,
#     seed: Optional[int] = None,
#     wide_row_prob: float = 0.0,
#     wide_col_prob: float = 0.0,
#     wide_factor: int = 2,
#     # opening options
#     entrance_side: str = "left",    # 'left'|'right'|'top'|'bottom'
#     exit_side: str = "right",
#     entrance_pos: float = 0.5,      # 0..1 along that border
#     exit_pos: float = 0.5,
#     # optional overrides: force row/col index depending on side
#     entrance_row_idx: Optional[int] = None,
#     entrance_col_idx: Optional[int] = None,
#     exit_row_idx: Optional[int] = None,
#     exit_col_idx: Optional[int] = None,
# ) -> Tuple[np.ndarray, float, float]:
def generate_maze_metric(
    rows: int,
    cols: int,
    *,
    width_m: float,
    height_m: float,
    wall_thickness_m: float,
    wall_height_m: float,
    meters_per_pixel: float = 0.01,
    seed: Optional[int] = None,
    wide_row_prob: float = 0.0,
    wide_col_prob: float = 0.0,
    wide_factor: int = 2,
    entrance_side: str = "left",
    exit_side: str = "right",
    entrance_pos: float = 0.5,
    exit_pos: float = 0.5,
    entrance_row_idx: Optional[int] = None,
    entrance_col_idx: Optional[int] = None,
    exit_row_idx: Optional[int] = None,
    exit_col_idx: Optional[int] = None,
) -> Tuple[np.ndarray, float, float, dict]:
    """
    Given overall size/wall thickness/height in meters, generate a (0=free, 255=wall) image.
    Openings are specified by side + 0..1 position (or directly by row/col override).
    Returns: (maze_img, meters_per_pixel, wall_height_m, openings_meta)
    """
    if rows < 2 or cols < 2:
        raise ValueError("rows, cols must be >= 2")
    if width_m <= 0 or height_m <= 0 or wall_thickness_m <= 0 or wall_height_m <= 0:
        raise ValueError("All dimensions must be positive.")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be > 0")
    if wide_factor < 1:
        raise ValueError("wide_factor must be >= 1")
    if seed is not None:
        random.seed(seed)

    # Resolution
    W_px = int(round(width_m / meters_per_pixel))
    H_px = int(round(height_m / meters_per_pixel))
    wall_px = max(1, int(round(wall_thickness_m / meters_per_pixel)))
    if W_px < (cols + 1) * wall_px + cols or H_px < (rows + 1) * wall_px + rows:
        raise ValueError("Insufficient resolution. Use smaller meters_per_pixel or larger size.")

    # Perfect-maze topology by DFS
    h_walls = np.ones((rows + 1, cols), dtype=bool)
    v_walls = np.ones((rows, cols + 1), dtype=bool)
    visited = np.zeros((rows, cols), dtype=bool)

    def neighbors(r: int, c: int) -> List[Tuple[int, int, str]]:
        out = []
        if r > 0 and not visited[r - 1, c]: out.append((r - 1, c, "up"))
        if r < rows - 1 and not visited[r + 1, c]: out.append((r + 1, c, "down"))
        if c > 0 and not visited[r, c - 1]: out.append((r, c - 1, "left"))
        if c < cols - 1 and not visited[r, c + 1]: out.append((r, c + 1, "right"))
        return out

    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        cr, cc = stack[-1]
        nbrs = neighbors(cr, cc)
        if not nbrs:
            stack.pop()
            continue
        nr, nc, d = random.choice(nbrs)
        if d == "up":      h_walls[cr, cc]     = False
        elif d == "down":  h_walls[cr + 1, cc] = False
        elif d == "left":  v_walls[cr, cc]     = False
        else:              v_walls[cr, cc + 1] = False
        visited[nr, nc] = True
        stack.append((nr, nc))

    # Interior corridor area (pixels)
    interior_w_px = W_px - (cols + 1) * wall_px
    interior_h_px = H_px - (rows + 1) * wall_px
    if interior_w_px < cols or interior_h_px < rows:
        raise ValueError("Interior too small. Adjust parameters.")

    # Variable corridor widths by row/column weights
    row_weights = [float(wide_factor if random.random() < wide_row_prob else 1.0) for _ in range(rows)]
    col_weights = [float(wide_factor if random.random() < wide_col_prob else 1.0) for _ in range(cols)]

    cell_h = _alloc_sizes_by_weights(row_weights, interior_h_px)  # each row cell height [px]
    cell_w = _alloc_sizes_by_weights(col_weights, interior_w_px)  # each column cell width [px]

    # Offsets
    y_cells, y_walls = _compute_offsets_from_sizes(cell_h, wall_px, H_px)
    x_cells, x_walls = _compute_offsets_from_sizes(cell_w, wall_px, W_px)

    # Raster (0=free, 255=wall)
    img = np.full((H_px, W_px), 255, dtype=np.uint8)

    # Cell interiors
    for r in range(rows):
        for c in range(cols):
            y0 = y_cells[r]; x0 = x_cells[c]
            img[y0:y0 + cell_h[r], x0:x0 + cell_w[c]] = 0

    # Vertical passages
    for r in range(rows):
        for c in range(1, cols):
            if not v_walls[r, c]:
                y0 = y_cells[r]
                x_wall = x_walls[c]
                img[y0:y0 + cell_h[r], x_wall:x_wall + wall_px] = 0

    # Horizontal passages
    for r in range(1, rows):
        for c in range(cols):
            if not h_walls[r, c]:
                y_wall = y_walls[r]
                x0 = x_cells[c]
                img[y_wall:y_wall + wall_px, x0:x0 + cell_w[c]] = 0

    def _choose_index_by_pos(lengths: List[int], p: float) -> int:
        p = max(0.0, min(1.0, float(p)))
        total = sum(lengths)
        target = p * total
        acc = 0
        idx = 0
        for i, L in enumerate(lengths):
            if target <= acc + L or i == len(lengths) - 1:
                idx = i; break
            acc += L
        return idx

    # Opening metadata for later corridor placement in XML
    world_W = W_px * meters_per_pixel
    world_H = H_px * meters_per_pixel
    openings_meta = {
        "entrance": None,
        "exit": None,
        "world_W": world_W,
        "world_H": world_H,
        "wall_thickness_m": wall_thickness_m,
    }

    def carve_opening(side: str, pos_frac: float,
                      row_idx_override: Optional[int],
                      col_idx_override: Optional[int],
                      tag: str):
        side = side.lower()
        if side not in ("left", "right", "top", "bottom"):
            raise ValueError("side must be 'left'|'right'|'top'|'bottom'")

        if side in ("left", "right"):
            # choose row index
            if row_idx_override is not None:
                idx = int(np.clip(row_idx_override, 0, rows - 1))
            else:
                idx = _choose_index_by_pos(cell_h, pos_frac)
            y0 = y_cells[idx]
            h = cell_h[idx]
            x0 = 0 if side == "left" else x_walls[-1]
            img[y0:y0 + h, x0:x0 + wall_px] = 0

            # opening in world coords / opening width (tangent axis = y)
            center_tangent_m = (H_px/2.0 - (y0 + h*0.5)) * meters_per_pixel
            width_m_local = h * meters_per_pixel
            boundary_coord = (-world_W*0.5) if side == "left" else (world_W*0.5)

            openings_meta[tag] = {
                "side": side,
                "center_tangent_m": center_tangent_m,  # y-center
                "width_m": width_m_local,              # corridor width (same as opening width)
                "boundary_coord": boundary_coord,      # x boundary coordinate
            }
        else:
            # choose column index
            if col_idx_override is not None:
                idx = int(np.clip(col_idx_override, 0, cols - 1))
            else:
                idx = _choose_index_by_pos(cell_w, pos_frac)
            x0 = x_cells[idx]
            w = cell_w[idx]
            y0 = 0 if side == "top" else y_walls[-1]
            img[y0:y0 + wall_px, x0:x0 + w] = 0

            # opening in world coords / opening width (tangent axis = x)
            center_tangent_m = ((x0 + w*0.5) - W_px/2.0) * meters_per_pixel
            width_m_local = w * meters_per_pixel
            boundary_coord = (world_H*0.5) if side == "top" else (-world_H*0.5)

            openings_meta[tag] = {
                "side": side,
                "center_tangent_m": center_tangent_m,  # x-center
                "width_m": width_m_local,              # corridor width (same as opening width)
                "boundary_coord": boundary_coord,      # y boundary coordinate
            }

    # Open entrance/exit
    carve_opening(entrance_side, entrance_pos, entrance_row_idx, entrance_col_idx, tag="entrance")
    carve_opening(exit_side,     exit_pos,     exit_row_idx,     exit_col_idx,     tag="exit")

    return img, meters_per_pixel, wall_height_m, openings_meta


# ---------------------------------------------------------------------------
# 3) Image → MJCF (merge wall pixels into rectangle boxes)
# ---------------------------------------------------------------------------
def _row_runs(mask_row: np.ndarray) -> List[Tuple[int, int]]:
    W = mask_row.shape[0]
    runs, i = [], 0
    while i < W:
        if mask_row[i]:
            j = i + 1
            while j < W and mask_row[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _wall_rectangles(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    H, W = mask.shape
    rects: List[Tuple[int, int, int, int]] = []
    prev_active = {}
    for y in range(H):
        runs = _row_runs(mask[y])
        curr_active = {}
        for (x1, x2) in runs:
            key = (x1, x2)
            if key in prev_active:
                r = prev_active.pop(key); r['y2'] = y + 1
                curr_active[key] = r
            else:
                curr_active[key] = {'x1': x1, 'x2': x2, 'y1': y, 'y2': y + 1}
        for r in prev_active.values():
            rects.append((r['x1'], r['y1'], r['x2'], r['y2']))
        prev_active = curr_active
    for r in prev_active.values():
        rects.append((r['x1'], r['y1'], r['x2'], r['y2']))
    return rects


def maze_array_to_mjcf(
    maze_name: str,
    arr: np.ndarray,
    meters_per_pixel: float,
    wall_height_m: float,
    xml_path: str = "maze.xml",
    walls_threshold: int = 128,
    walls_are_high_values: bool = True,
    wall_rgba: Tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0),
    floor_rgba: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0),
    add_camera: bool = True,
    add_floor: bool = True,
    floor_thickness_m: float = 0.01,
    corridors: Optional[List[dict]] = None,   # ★ add: corridor specs
) -> str:
    """Convert 0=free, 255=wall image to MJCF(XML) and save to file."""
    if arr.ndim != 2:
        raise ValueError("arr must be 2D grayscale")
    if meters_per_pixel <= 0 or wall_height_m <= 0:
        raise ValueError("meters_per_pixel and wall_height_m must be > 0")

    arr_u8 = arr.astype(np.uint8, copy=False)
    wall_mask = (arr_u8 >= walls_threshold) if walls_are_high_values else (arr_u8 < walls_threshold)

    H, W = wall_mask.shape
    world_W = W * meters_per_pixel
    world_H = H * meters_per_pixel

    rects = _wall_rectangles(wall_mask)
    def f4(x: float) -> str: return f"{x:.6f}"

    # Walls (existing)
    geoms_xml: List[str] = []
    half_h = 0.5 * wall_height_m
    def f4(x: float) -> str: return f"{x:.6f}"

    for i, (x1, y1, x2, y2) in enumerate(rects):
        cx_px = 0.5 * (x1 + x2); cy_px = 0.5 * (y1 + y2)
        w_px  = x2 - x1;         h_px  = y2 - y1
        cx = (cx_px - W / 2.0) * meters_per_pixel
        cy = (H / 2.0 - cy_px) * meters_per_pixel
        sx = 0.5 * (w_px * meters_per_pixel)
        sy = 0.5 * (h_px * meters_per_pixel)
        sz = half_h
        geoms_xml.append(
            f'    <geom name="{maze_name}_wall_{i}" type="box" '
            f'pos="{f4(cx)} {f4(cy)} {f4(sz)}" '
            f'size="{f4(sx)} {f4(sy)} {f4(sz)}" '
            f'rgba="{wall_rgba[0]} {wall_rgba[1]} {wall_rgba[2]} {wall_rgba[3]}" '
            f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
        )

    # Floor (existing)
    floor = ""
    if add_floor:
        half_tx = 0.5 * floor_thickness_m
        floor = (
            f'    <geom name="{maze_name}_floor" type="box" '
            f'pos="0 0 {-half_tx:.6f}" '
            f'size="{(world_W*0.5):.6f} {(world_H*0.5):.6f} {half_tx:.6f}" '
            f'rgba="{floor_rgba[0]} {floor_rgba[1]} {floor_rgba[2]} {floor_rgba[3]}" '
            f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
        )

    # Camera (existing)
    cam_xml = ""
    if add_camera:
        cam_h = 1.3 * max(world_W, world_H)
        cam_xml = f'    <camera name="{maze_name}_top" pos="0 0 {f4(cam_h)}" quat="0 1 0 0" mode="trackcom"/>'

    # ★ Corridor geoms
    def _corridor_geoms(spec: dict) -> List[str]:
        """
        spec keys:
          name, side, boundary_coord, center_tangent_m, width_m, length_m,
          wall_thickness_m, wall_height_m, floor_thickness_m
        """
        name = spec.get("name", "corridor")
        side = spec["side"].lower()
        L = float(spec["length_m"])
        Wgap = float(spec["width_m"])
        t = float(spec["wall_thickness_m"])
        h = float(spec["wall_height_m"])
        half_h_local = 0.5 * h
        tx = float(spec["floor_thickness_m"])
        half_tx_local = 0.5 * tx

        rgba = wall_rgba
        parts = []

        if side in ("left", "right"):
            # Corridor along +x (right) or -x (left)
            dir_sign = -1.0 if side == "left" else +1.0
            x_center = spec["boundary_coord"] + dir_sign * (L * 0.5)
            y_center = float(spec["center_tangent_m"])

            # Two side walls at ±(gap/2 + t/2)
            y_upper = y_center + (Wgap * 0.5 + t * 0.5)
            y_lower = y_center - (Wgap * 0.5 + t * 0.5)

            # Walls (length L, thickness t)
            sx = L * 0.5
            sy = t * 0.5
            sz = half_h_local

            parts.append(
                f'    <geom name="{maze_name}_{name}_wall_upper" type="box" '
                f'pos="{f4(x_center)} {f4(y_upper)} {f4(sz)}" '
                f'size="{f4(sx)} {f4(sy)} {f4(sz)}" '
                f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" '
                f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
            )
            parts.append(
                f'    <geom name="{maze_name}_{name}_wall_lower" type="box" '
                f'pos="{f4(x_center)} {f4(y_lower)} {f4(sz)}" '
                f'size="{f4(sx)} {f4(sy)} {f4(sz)}" '
                f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" '
                f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
            )

            # Floor
            parts.append(
                f'    <geom name="{maze_name}_{name}_floor" type="box" '
                f'pos="{f4(x_center)} {f4(y_center)} {-half_tx_local:.6f}" '
                f'size="{f4(sx)} {f4(Wgap*0.5)} {f4(half_tx_local)}" '
                f'rgba="{floor_rgba[0]} {floor_rgba[1]} {floor_rgba[2]} {floor_rgba[3]}" '
                f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
            )

        else:
            # side in ("top", "bottom"): corridor along +y (top) or -y (bottom)
            dir_sign = +1.0 if side == "top" else -1.0
            y_center = spec["boundary_coord"] + dir_sign * (L * 0.5)
            x_center = float(spec["center_tangent_m"])

            # Two side walls at ±(gap/2 + t/2)
            x_right = x_center + (Wgap * 0.5 + t * 0.5)
            x_left  = x_center - (Wgap * 0.5 + t * 0.5)

            # Walls (length L, thickness t)
            sx = t * 0.5
            sy = L * 0.5
            sz = half_h_local

            parts.append(
                f'    <geom name="{maze_name}_{name}_wall_right" type="box" '
                f'pos="{f4(x_right)} {f4(y_center)} {f4(sz)}" '
                f'size="{f4(sx)} {f4(sy)} {f4(sz)}" '
                f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" '
                f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
            )
            parts.append(
                f'    <geom name="{maze_name}_{name}_wall_left" type="box" '
                f'pos="{f4(x_left)} {f4(y_center)} {f4(sz)}" '
                f'size="{f4(sx)} {f4(sy)} {f4(sz)}" '
                f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" '
                f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
            )

            # Floor
            parts.append(
                f'    <geom name="{maze_name}_{name}_floor" type="box" '
                f'pos="{f4(x_center)} {f4(y_center)} {-half_tx_local:.6f}" '
                f'size="{f4(Wgap*0.5)} {f4(sy)} {f4(half_tx_local)}" '
                f'rgba="{floor_rgba[0]} {floor_rgba[1]} {floor_rgba[2]} {floor_rgba[3]}" '
                f'contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>'
            )

        return parts

    # Add corridor geoms if requested
    if corridors:
        for spec in corridors:
            geoms_xml.extend(_corridor_geoms(spec))

    # Serialize XML (existing)
    geom_xml = "\n".join(geoms_xml)

    xml = (
        "<mujocoinclude>\n"
        "<body>\n"
        f"{floor}\n"
        f"{cam_xml}\n"
        f"{geom_xml}\n"
        "</body>\n"
        "</mujocoinclude>\n"
        )

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return xml


# ---------------------------------------------------------------------------
# 4) One-shot pipeline (metric params → optional PNG → MJCF)
# ---------------------------------------------------------------------------
def build_maze_mjcf_metric(
    maze_name: str,
    rows: int,
    cols: int,
    *,
    width_m: float,
    height_m: float,
    wall_thickness_m: float,
    wall_height_m: float,
    meters_per_pixel: float = 0.01,
    seed: Optional[int] = None,
    wide_row_prob: float = 0.0,
    wide_col_prob: float = 0.0,
    wide_factor: int = 2,
    png_path: Optional[str] = "maze.png",
    xml_path: str = "maze.xml",
    add_camera: bool = True,
    add_floor: bool = True,
    floor_thickness_m: float = 0.01,
    # opening specification
    entrance_side: str = "left",
    exit_side: str = "right",
    entrance_pos: float = 0.5,
    exit_pos: float = 0.5,
    entrance_row_idx: Optional[int] = None,
    entrance_col_idx: Optional[int] = None,
    exit_row_idx: Optional[int] = None,
    exit_col_idx: Optional[int] = None,
    # ★ corridor options (only in XML, not baked into image)
    entrance_corridor: bool = False,
    exit_corridor: bool = False,
    entrance_corridor_length_m: float = 0.0,
    exit_corridor_length_m: float = 0.0,
) -> Tuple[np.ndarray, str]:
    """
    End-to-end:
      - generate maze image from metric parameters
      - optionally save PNG (corridors are not rasterized into the image)
      - generate MJCF (corridors added only into XML)
    """
    ret = generate_maze_metric(
        rows, cols,
        width_m=width_m, height_m=height_m,
        wall_thickness_m=wall_thickness_m,
        wall_height_m=wall_height_m,
        meters_per_pixel=meters_per_pixel,
        seed=seed,
        wide_row_prob=wide_row_prob,
        wide_col_prob=wide_col_prob,
        wide_factor=wide_factor,
        entrance_side=entrance_side,
        exit_side=exit_side,
        entrance_pos=entrance_pos,
        exit_pos=exit_pos,
        entrance_row_idx=entrance_row_idx,
        entrance_col_idx=entrance_col_idx,
        exit_row_idx=exit_row_idx,
        exit_col_idx=exit_col_idx,
    )

    # Backward compatibility guard (older tuple shape)
    if len(ret) == 4:
        maze, mpp, wall_h, openings = ret
    else:
        maze, mpp, wall_h = ret
        openings = {
            "entrance": None, "exit": None,
            "world_W": width_m, "world_H": height_m,
            "wall_thickness_m": wall_thickness_m
        }

    if png_path is not None:
        Image.fromarray(maze).save(png_path)

    # Build corridor specs (only for XML)
    corridors = []
    if entrance_corridor and openings.get("entrance") and entrance_corridor_length_m > 0:
        spec = dict(openings["entrance"])
        spec.update({
            "name": "entrance_corridor",
            "length_m": float(entrance_corridor_length_m),
            "wall_thickness_m": openings["wall_thickness_m"],
            "wall_height_m": wall_h,
            "floor_thickness_m": floor_thickness_m,
        })
        corridors.append(spec)

    if exit_corridor and openings.get("exit") and exit_corridor_length_m > 0:
        spec = dict(openings["exit"])
        spec.update({
            "name": "exit_corridor",
            "length_m": float(exit_corridor_length_m),
            "wall_thickness_m": openings["wall_thickness_m"],
            "wall_height_m": wall_h,
            "floor_thickness_m": floor_thickness_m,
        })
        corridors.append(spec)

    xml_text = maze_array_to_mjcf(
        maze_name=maze_name,
        arr=maze,
        meters_per_pixel=mpp,
        wall_height_m=wall_h,
        xml_path=xml_path,
        walls_threshold=128,
        walls_are_high_values=True,
        add_camera=add_camera,
        add_floor=add_floor,
        floor_thickness_m=floor_thickness_m,
        corridors=corridors,   # ★ add
    )
    return maze, xml_text

# === Utility to combine two mazes into one mujocoinclude ===
import math
from typing import Optional, Tuple, Dict, Any, List

def _qy(angle_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) for a rotation about +Y by angle_deg."""
    a = math.radians(angle_deg) * 0.5
    return (math.cos(a), 0.0, math.sin(a), 0.0)

def _Ry(angle_deg: float):
    """Rotation matrix about +Y (3x3)."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    # [[ c, 0, s],
    #  [ 0, 1, 0],
    #  [-s, 0, c]]
    return ((c, 0.0, s),
            (0.0, 1.0, 0.0),
            (-s, 0.0, c))

def _endpoint_from_opening(opening: Dict[str, Any], length_m: float) -> Tuple[float, float, float]:
    """
    opening: openings_meta['entrance' or 'exit'] from generate_maze_metric
      - keys: side, center_tangent_m, width_m, boundary_coord
    length_m: corridor length
    returns: (x, y, z) in the maze's local frame
    """
    side = opening["side"].lower()
    center_t = float(opening["center_tangent_m"])
    boundary = float(opening["boundary_coord"])
    L = float(length_m)

    if side == "left":
        dir_sign = -1.0
        x = boundary + dir_sign * L
        y = center_t
    elif side == "right":
        dir_sign = +1.0
        x = boundary + dir_sign * L
        y = center_t
    elif side == "top":
        dir_sign = +1.0
        x = center_t
        y = boundary + dir_sign * L
    elif side == "bottom":
        dir_sign = -1.0
        x = center_t
        y = boundary + dir_sign * L
    else:
        raise ValueError("opening['side'] must be left/right/top/bottom")
    return (x, y, 0.0)  # endpoint on corridor floor (z=0)

def _apply_Ry(R: Tuple[Tuple[float,float,float], ...], p: Tuple[float,float,float]) -> Tuple[float,float,float]:
    x = R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2]
    y = R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2]
    z = R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2]
    return (x, y, z)

def build_two_mazes_include_xml(
    # --- Common ---
    meters_per_pixel: float,
    wall_height_m: float,
    wall_thickness_m: float,
    floor_thickness_m: float = 0.01,
    # --- Maze 1 params ---
    maze1_name: str = "maze_1",
    rows1: int = 8, cols1: int = 8,
    width1_m: float = 0.36, height1_m: float = 0.36,
    seed1: Optional[int] = 1,
    entrance1_side: str = "left",
    exit1_side: str = "right",
    entrance1_pos: float = 0.5,
    exit1_pos: float = 0.5,
    # Corridor at the exit of maze 1
    exit_corridor_length1_m: float = 0.40,
    # --- Maze 2 params ---
    maze2_name: str = "maze_2",
    rows2: int = 8, cols2: int = 8,
    width2_m: float = 0.36, height2_m: float = 0.36,
    seed2: Optional[int] = 2,
    entrance2_side: str = "left",   # maze 2 has an entrance corridor
    exit2_side: str = "right",
    entrance2_pos: float = 0.5,
    exit2_pos: float = 0.5,
    # Corridor at the entrance of maze 2
    entrance_corridor_length2_m: float = 0.40,
    # --- Tilt of maze 2 about Y (no z-rotation) ---
    maze2_tilt_deg: float = 0.0,
    # --- MuJoCo GLOBAL (parent) offset for printing ---
    world_offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    world_offset_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),  # (w,x,y,z)
    # --- File paths ---
    png1_path: Optional[str] = "maze_1.png",
    xml1_path: str = "maze_1.xml",
    png2_path: Optional[str] = "maze_2.png",
    xml2_path: str = "maze_2.xml",
    combined_xml_path: str = "maze_combined.xml",
) -> str:
    """
    1) Maze 1: exit corridor 생성 → maze_1.xml
    2) Maze 2: entrance corridor 생성 → maze_2.xml
    3) Maze 2를 Y축으로만 기울이고(자기 z-yaw는 0), 두 통로 endpoint가 world에서 일치하도록 평행이동
    4) 두 미로를 포함하는 mujocoinclude 출력
    5) ★ 출력(모두 '실제 MuJoCo 전역 좌표계' = world_offset 적용):
       - 각 미로의 '포털(통로 제외) 중심' 위치
       - 각 미로 포털의 회전행렬 (미로1 = I, 미로2 = R_y(tilt); 이후 world_offset 적용)
       - 두 통로 교차점(만나는 지점) 위치 및 방위(미로1 프레임 기반, 이후 world_offset 적용)
    """
    import numpy as _np
    import math as _m

    # --- helpers ---
    def _quat_to_R(q: Tuple[float, float, float, float]) -> _np.ndarray:
        w, x, y, z = q
        return _np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=float)

    def _to_np_R(Rtuple: Tuple[Tuple[float, float, float], ...]) -> _np.ndarray:
        return _np.array(Rtuple, dtype=float)

    def _opening_local_pose(opening: Dict[str, Any]) -> Tuple[_np.ndarray]:
        """포털 중심점(통로 제외) 위치만 로컬로 반환 (회전은 여기서 쓰지 않음)."""
        side = opening["side"].lower()
        ct = float(opening["center_tangent_m"])
        bc = float(opening["boundary_coord"])
        if side == "left":
            p = _np.array([bc, ct, 0.0])
        elif side == "right":
            p = _np.array([bc, ct, 0.0])
        elif side == "top":
            p = _np.array([ct, bc, 0.0])
        elif side == "bottom":
            p = _np.array([ct, bc, 0.0])
        else:
            raise ValueError("opening['side'] must be left/right/top/bottom")
        return p

    def _apply_Rt(R: _np.ndarray, t: _np.ndarray, p: _np.ndarray) -> _np.ndarray:
        return R @ p + t

    # 1) Maze 1 (with exit corridor)
    img1, mpp1, wall_h1, openings1 = generate_maze_metric(
        rows1, cols1,
        width_m=width1_m, height_m=height1_m,
        wall_thickness_m=wall_thickness_m,
        wall_height_m=wall_height_m,
        meters_per_pixel=meters_per_pixel,
        seed=seed1,
        entrance_side=entrance1_side, exit_side=exit1_side,
        entrance_pos=entrance1_pos,   exit_pos=exit1_pos,
    )
    corridor_specs_1: List[Dict[str, Any]] = []
    if openings1.get("exit") and exit_corridor_length1_m > 0:
        spec1 = dict(openings1["exit"])
        spec1.update({
            "name": "exit_corridor",
            "length_m": float(exit_corridor_length1_m),
            "wall_thickness_m": wall_thickness_m,
            "wall_height_m": wall_height_m,
            "floor_thickness_m": floor_thickness_m,
        })
        corridor_specs_1.append(spec1)
    if png1_path is not None:
        Image.fromarray(img1).save(png1_path)
    maze_array_to_mjcf(
        maze_name=maze1_name,
        arr=img1,
        meters_per_pixel=mpp1,
        wall_height_m=wall_h1,
        xml_path=xml1_path,
        walls_threshold=128,
        walls_are_high_values=True,
        add_camera=True,
        add_floor=True,
        floor_thickness_m=floor_thickness_m,
        corridors=corridor_specs_1,
    )

    # 2) Maze 2 (with entrance corridor)
    img2, mpp2, wall_h2, openings2 = generate_maze_metric(
        rows2, cols2,
        width_m=width2_m, height_m=height2_m,
        wall_thickness_m=wall_thickness_m,
        wall_height_m=wall_height_m,
        meters_per_pixel=meters_per_pixel,
        seed=seed2,
        entrance_side=entrance2_side, exit_side=exit2_side,
        entrance_pos=entrance2_pos,   exit_pos=exit2_pos,
    )
    corridor_specs_2: List[Dict[str, Any]] = []
    if openings2.get("entrance") and entrance_corridor_length2_m > 0:
        spec2 = dict(openings2["entrance"])
        spec2.update({
            "name": "entrance_corridor",
            "length_m": float(entrance_corridor_length2_m),
            "wall_thickness_m": wall_thickness_m,
            "wall_height_m": wall_height_m,
            "floor_thickness_m": floor_thickness_m,
        })
        corridor_specs_2.append(spec2)
    if png2_path is not None:
        Image.fromarray(img2).save(png2_path)
    maze_array_to_mjcf(
        maze_name=maze2_name,
        arr=img2,
        meters_per_pixel=mpp2,
        wall_height_m=wall_h2,
        xml_path=xml2_path,
        walls_threshold=128,
        walls_are_high_values=True,
        add_camera=True,
        add_floor=True,
        floor_thickness_m=floor_thickness_m,
        corridors=corridor_specs_2,
    )

    # 3) Endpoint alignment (world frame, before global offset)
    p1_end_world = _endpoint_from_opening(openings1["exit"], exit_corridor_length1_m)
    p2_end_local = _endpoint_from_opening(openings2["entrance"], entrance_corridor_length2_m)

    R2_tuple = _Ry(maze2_tilt_deg)    # only Y-tilt, no z-rotation in maze2 frame
    R2 = _to_np_R(R2_tuple)
    Rp2 = R2 @ _np.array(p2_end_local, dtype=float)  # rotated endpoint of maze2
    pos2 = (p1_end_world[0] - Rp2[0],
            p1_end_world[1] - Rp2[1],
            p1_end_world[2] - Rp2[2])
    quat2 = _qy(maze2_tilt_deg)

    # ----------------------- POSE REPORT (GLOBAL MuJoCo frame) -----------------------
    # Maze-level transforms in local "include" world:
    Rw1 = _np.eye(3)                # maze1 has no rotation
    tw1 = _np.zeros(3)
    Rw2 = R2                        # maze2 is only Y-tilted
    tw2 = _np.array(pos2, dtype=float)

    # Portal centers (corridor EXCLUDED) in local maze frames:
    p1_ent_L = _opening_local_pose(openings1["entrance"])
    p1_exit_L = _opening_local_pose(openings1["exit"])
    p2_ent_L = _opening_local_pose(openings2["entrance"])
    p2_exit_L = _opening_local_pose(openings2["exit"])

    # In "include world" (before global offset):
    p1_ent_W = _apply_Rt(Rw1, tw1, p1_ent_L)
    p1_exit_W = _apply_Rt(Rw1, tw1, p1_exit_L)
    p2_ent_W = _apply_Rt(Rw2, tw2, p2_ent_L)
    p2_exit_W = _apply_Rt(Rw2, tw2, p2_exit_L)

    # ★ Desired portal orientations (no z-rotation about each maze):
    #   - Maze1: Identity
    #   - Maze2: R_y(tilt)
    R1_portal_W = Rw1           # = I
    R2_portal_W = Rw2           # = Ry(tilt)

    # Global (MuJoCo parent) offset:
    Rg = _quat_to_R(world_offset_quat)
    tg = _np.array(world_offset_pos, dtype=float)

    # Transform to true MuJoCo GLOBAL:
    def _glob(pW: _np.ndarray, RW: _np.ndarray) -> Tuple[_np.ndarray, _np.ndarray]:
        return (Rg @ pW + tg, Rg @ RW)

    p1_ent_G, R1_ent_G = _glob(p1_ent_W, R1_portal_W)
    p1_exit_G, R1_exit_G = _glob(p1_exit_W, R1_portal_W)
    p2_ent_G, R2_ent_G = _glob(p2_ent_W, R2_portal_W)
    p2_exit_G, R2_exit_G = _glob(p2_exit_W, R2_portal_W)

    # Corridors intersection: point is p1_end_world → apply global offset
    intersection_W = _np.array(p1_end_world, dtype=float)
    intersection_G = Rg @ intersection_W + tg
    R_intersect_G1 = Rg @ R1_portal_W
    R_intersect_G2 = Rg @ R2_portal_W

    def _fmt_v(v: _np.ndarray) -> str:
        return f"[{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]"

    def _fmt_R(R: _np.ndarray) -> str:
        return ("[" + ", ".join(f"{R[0,i]:.6f}" for i in range(3)) + "]\n" +
                "\t     [" + ", ".join(f"{R[1,i]:.6f}" for i in range(3)) + "]\n" +
                "\t     [" + ", ".join(f"{R[2,i]:.6f}" for i in range(3)) + "]")

    print("\n================ POSE REPORT (WORLD frame) ================\n")
    print(f"{maze1_name} ENTRANCE:\n\tpos = {_fmt_v(p1_ent_G)}\n\tR =  {_fmt_R(R1_ent_G)}\n")
    print(f"{maze1_name} EXIT    :\n\tpos = {_fmt_v(p1_exit_G)}\n\tR =  {_fmt_R(R1_exit_G)}\n")
    print(f"{maze2_name} ENTRANCE:\n\tpos = {_fmt_v(p2_ent_G)}\n\tR =  {_fmt_R(R2_ent_G)}\n")
    print(f"{maze2_name} EXIT    :\n\tpos = {_fmt_v(p2_exit_G)}\n\tR =  {_fmt_R(R2_exit_G)}\n")
    print(f"CORRIDORS INTERSECTION:\n\tpos = {_fmt_v(intersection_G)}\n\tR1 = {_fmt_R(R_intersect_G1)}\n\tR2 = {_fmt_R(R_intersect_G2)}\n")
    # -------------------------------------------------------------------------------

    def f4(x: float) -> str: return f"{x:.6f}"

    combined = (
        "<mujocoinclude>\n"
        "  <body name=\"maze_1\" pos=\"0 0 0\" quat=\"1 0 0 0\">\n"
        f"    <include file=\"{xml1_path}\"/>\n"
        "  </body>\n"
        f"  <body name=\"maze_2\" pos=\"{f4(pos2[0])} {f4(pos2[1])} {f4(pos2[2])}\" "
        f"quat=\"{f4(quat2[0])} {f4(quat2[1])} {f4(quat2[2])} {f4(quat2[3])}\">\n"
        f"    <include file=\"{xml2_path}\"/>\n"
        "  </body>\n"
        "</mujocoinclude>\n"
    )
    with open(combined_xml_path, "w", encoding="utf-8") as f:
        f.write(combined)
    return combined


# ---------------------------------------------------------------------------
# 5) Main
# ---------------------------------------------------------------------------
import argparse
import os

def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    maze_img_folder = os.path.join(current_dir, "maze_imgs")
    if not os.path.exists(maze_img_folder):
        os.makedirs(maze_img_folder)
    
    build_two_mazes_include_xml(
        meters_per_pixel=0.001,
        wall_height_m=0.05,
        wall_thickness_m=0.01,
        floor_thickness_m=0.01,
        # Maze 1: has an exit corridor
        maze1_name="maze_1", rows1=8, cols1=6,
        width1_m=0.24, height1_m=0.36, seed1=args.seed1,
        entrance1_side="top",  entrance1_pos=0,
        exit1_side="right",    exit1_pos=1,
        exit_corridor_length1_m=0.05,
        # Maze 2: has an entrance corridor
        maze2_name="maze_2", rows2=8, cols2=6,
        width2_m=0.24, height2_m=0.36, seed2=args.seed2,
        entrance2_side="left", entrance2_pos=1,
        exit2_side="top",   exit2_pos=1,
        entrance_corridor_length2_m=0.05,
        # Tilt of maze 2 about +Y
        maze2_tilt_deg=args.tilt,
        # Output file paths
        png1_path=os.path.join(maze_img_folder, "maze1.png"),
        png2_path=os.path.join(maze_img_folder, "maze2.png"),
        xml1_path=os.path.join(current_dir, "franka_fr3", "maze1.xml"),
        xml2_path=os.path.join(current_dir, "franka_fr3", "maze2.xml"),
        combined_xml_path=os.path.join(current_dir, "franka_fr3", "maze.xml"),
        # Print
        world_offset_pos= (0.4, 0, 0.4),
    )
    print("Saved maze_1.xml, maze_2.xml, and maze.xml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1", type=int, default=0, help="Seed for build maze1")
    parser.add_argument("--seed2", type=int, default=1, help="Seed for build maze2")
    parser.add_argument("--tilt", type=float, default=-45, help="Degree for  tilt")

    args = parser.parse_args()
    main(args)
