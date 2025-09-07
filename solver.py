import tkinter as tk
from itertools import permutations
import copy
import time

from PIL import ImageGrab, Image, ImageStat, ImageTk, ImageChops, ImageDraw


# Optional OpenCV-based pipeline
import cv2  # type: ignore
import numpy as np  # type: ignore


# ------------------ Block Definitions ------------------
# Each block is defined as a list of (row, col) relative coordinates
BLOCKS = {
    "Dot": [(0, 0)],
    "I2": [(0, 0), (1, 0)],
    "I3": [(0, 0), (1, 0), (2, 0)],
    "I4": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "I5": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    "ZIGR": [(0, 1), (1, 0)],
    "ZIGL": [(0, 0), (1, 1)],
    "ZIGLL": [(0, 0), (1, 1), (2, 2)],
    "ZIGRL": [(0, 2), (1, 1), (2, 0)],
    "O2": [(0, 0), (0, 1), (1, 0), (1, 1)],
    # 2x3 and 3x2 rectangles
    "Rect2x3": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    "Rect3x2": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    # 3x3 square
    "O3": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    # 2x1 L-shapes (2x2 minus one corner) — four orientations
    "L2x1_mTR": [(0, 0), (1, 0), (1, 1)],  # missing Top-Right
    "L2x1_mTL": [(0, 1), (1, 0), (1, 1)],  # missing Top-Left
    "L2x1_mBR": [(0, 0), (0, 1), (1, 0)],  # missing Bottom-Right
    "L2x1_mBL": [(0, 0), (0, 1), (1, 1)],  # missing Bottom-Left
    # 3x1 L-shapes (4 blocks: long arm length 3, foot length 1)
    # Vertical main, foot to right
    "L3V_TR": [(0, 0), (1, 0), (2, 0), (0, 1)],  # top-right
    "L3V_BR": [(0, 0), (1, 0), (2, 0), (2, 1)],  # bottom-right (same as L3 below)
    # Vertical main, foot to left
    "L3V_TL": [(0, 1), (1, 1), (2, 1), (0, 0)],  # top-left
    "L3V_BL": [(0, 1), (1, 1), (2, 1), (2, 0)],  # bottom-left
    # Horizontal main, foot up
    "L3H_UL": [(1, 0), (1, 1), (1, 2), (0, 0)],  # up-left
    "L3H_UR": [(1, 0), (1, 1), (1, 2), (0, 2)],  # up-right
    # Horizontal main, foot down
    "L3H_DL": [(0, 0), (0, 1), (0, 2), (1, 0)],  # down-left
    "L3H_DR": [(0, 0), (0, 1), (0, 2), (1, 2)],  # down-right
    "L4_BUR": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    "L4_BUL": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
    "L4_BDR": [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],
    "L4_BDL": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
    "T3": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "B3": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "R3": [(0, 0), (1, 0), (1, 1), (2, 0)],
    "L3": [(0, 1), (1, 0), (1, 1), (2, 1)],
    "T4": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)],
    "S3": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "X3": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "W3": [(0, 1), (1, 0), (1, 1), (2, 0)],
    "Z3": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "WL2": [(0, 0), (0, 1)],
    "WL3": [(0, 0), (0, 1), (0, 2)],
    "WL4": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "WL5": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
}

GRID_SIZE = 8
CELL_SIZE = 50
MAX_BRANCHING_PER_BLOCK = 50  # limit per-block placements explored for speed
PALETTE_COLS = 5


# ------------------ Game Logic ------------------
def can_place(grid, block, r, c):
    for dr, dc in block:
        rr, cc = r + dr, c + dc
        if not (0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE):
            return False
        if grid[rr][cc] == 1:
            return False
    return True


def place_block(grid, block, r, c):
    new_grid = copy.deepcopy(grid)
    for dr, dc in block:
        new_grid[r + dr][c + dc] = 1
    # clear lines
    new_grid, _ = clear_lines(new_grid)
    return new_grid


def clear_lines(grid):
    new_grid = copy.deepcopy(grid)
    cleared = 0
    # Clear full rows
    for r in range(GRID_SIZE):
        if all(new_grid[r][c] == 1 for c in range(GRID_SIZE)):
            for c in range(GRID_SIZE):
                new_grid[r][c] = 0
            cleared += 1
    # Clear full columns
    for c in range(GRID_SIZE):
        if all(new_grid[r][c] == 1 for r in range(GRID_SIZE)):
            for r in range(GRID_SIZE):
                new_grid[r][c] = 0
            cleared += 1
    return new_grid, cleared


def score_grid(grid):
    # Composite heuristic:
    # - Reward more empty cells
    # - Reward longer contiguous empty runs in rows/cols
    # - Penalize isolated empty cells (no 4-neighbors empty)
    # - Reward near-complete lines (0,1,2 empties)
    empty = 0
    isolated_penalty = 0
    near_line_reward = 0

    # Count empties and isolated empties, and near-complete lines
    for r in range(GRID_SIZE):
        row_empties = 0
        for c in range(GRID_SIZE):
            if grid[r][c] == 0:
                empty += 1
                row_empties += 1
                up = r > 0 and grid[r - 1][c] == 0
                down = r < GRID_SIZE - 1 and grid[r + 1][c] == 0
                left = c > 0 and grid[r][c - 1] == 0
                right = c < GRID_SIZE - 1 and grid[r][c + 1] == 0
                if not (up or down or left or right):
                    isolated_penalty += 1
        if row_empties <= 2:
            near_line_reward += 2 - row_empties + 1

    for c in range(GRID_SIZE):
        col_empties = 0
        for r in range(GRID_SIZE):
            if grid[r][c] == 0:
                col_empties += 1
        if col_empties <= 2:
            near_line_reward += 2 - col_empties + 1

    # Reward contiguous empty runs squared (rows and cols)
    def runs_score_lines():
        score = 0
        # Rows
        for r in range(GRID_SIZE):
            run = 0
            for c in range(GRID_SIZE):
                if grid[r][c] == 0:
                    run += 1
                if grid[r][c] == 1 or c == GRID_SIZE - 1:
                    if grid[r][c] == 1 and run > 0:
                        score += run * run
                        run = 0
                    elif c == GRID_SIZE - 1:
                        score += run * run
                        run = 0
        # Cols
        for c in range(GRID_SIZE):
            run = 0
            for r in range(GRID_SIZE):
                if grid[r][c] == 0:
                    run += 1
                if grid[r][c] == 1 or r == GRID_SIZE - 1:
                    if grid[r][c] == 1 and run > 0:
                        score += run * run
                        run = 0
                    elif r == GRID_SIZE - 1:
                        score += run * run
                        run = 0
        return score

    runs_score = runs_score_lines()

    # Weights tuned for speed and reasonable play
    return 1.0 * empty + 0.1 * runs_score - 2.0 * isolated_penalty + 3.0 * near_line_reward


def normalize_shape(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted([(r - min_r, c - min_c) for r, c in cells])


# ------------------ Shape Utilities ------------------
def shape_key(cells):
    # tuple of sorted (r,c) pairs so it can be used as dict key
    norm = normalize_shape(cells)
    return tuple(norm)


def flip_horizontal(cells):
    # mirror across vertical axis, then normalize
    if not cells:
        return []
    max_c = max(c for _, c in cells)
    return normalize_shape([(r, max_c - c) for r, c in cells])


def flip_vertical(cells):
    # mirror across horizontal axis, then normalize
    if not cells:
        return []
    max_r = max(r for r, _ in cells)
    return normalize_shape([(max_r - r, c) for r, c in cells])


# Precompute a lookup of base (normalized, orientation-specific) shapes → name
BLOCK_BASE_SHAPES = {name: normalize_shape(shape) for name, shape in BLOCKS.items()}
BLOCK_SHAPE_LOOKUP = {tuple(coords): name for name, coords in BLOCK_BASE_SHAPES.items()}


def rotate_shape(cells):
    # 90 deg rotation (r, c) -> (c, -r), then normalize
    return normalize_shape([(c, -r) for r, c in cells])


def all_rotations(cells):
    base = normalize_shape(cells)
    shapes = [base]
    cur = base
    for _ in range(3):
        cur = rotate_shape(cur)
        if cur not in shapes:
            shapes.append(cur)
    return shapes


def match_block_name(detected_cells, allow_mirror: bool = False):
    target = normalize_shape(detected_cells)
    # Try all rotations of the TARGET against the precomputed lookup
    for rot in all_rotations(target):
        k = tuple(rot)
        if k in BLOCK_SHAPE_LOOKUP:
            name = BLOCK_SHAPE_LOOKUP[k]
            return name, BLOCK_BASE_SHAPES[name]
    # Optionally consider mirrored variants (treat mirror-equals as the same class)
    if allow_mirror:
        mirrors = [flip_horizontal(target), flip_vertical(target)]
        for m in mirrors:
            for rot in all_rotations(m):
                k = tuple(rot)
                if k in BLOCK_SHAPE_LOOKUP:
                    name = BLOCK_SHAPE_LOOKUP[k]
                    return name, BLOCK_BASE_SHAPES[name]
    # Fuzzy fallback: symmetric difference distance between rotated targets and base shapes
    target_rots = all_rotations(target)

    # Helper: dims and counts
    def dims(cells):
        if not cells:
            return 0, 0
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    target_count = len(target)
    target_dims = dims(target)
    best_name = None
    best_shape = None
    best_diff = 10**9
    for name, base in BLOCK_BASE_SHAPES.items():
        # Strong filter: same number of tiles
        if len(base) != target_count:
            continue
        # Prefer exact same bounding box dimensions
        base_dims = dims(base)
        if base_dims != target_dims:
            continue
        bset = set(base)
        for rot in target_rots:
            diff = len(set(rot) ^ bset)
            if diff < best_diff:
                best_diff = diff
                best_name = name
                best_shape = base
    # Only accept very close matches
    if best_name is not None and best_diff <= 1:
        try:
            print(f"[vision] approx match -> {best_name} (diff={best_diff})")
        except Exception:
            pass
        return best_name, best_shape
    # Unknown: return normalized cells and human-readable tag
    # Compute bounding box size
    if target:
        rows = max(r for r, _ in target) - min(r for r, _ in target) + 1
        cols = max(c for _, c in target) - min(c for _, c in target) + 1
        return f"Unknown({rows}x{cols})", target
    return None, None


def best_move(grid, blocks):
    best_score = -float("inf")
    best_moves = None

    def all_positions_sorted(g, blk):
        # blk may be a tuple (name, shape) or just a shape list for legacy
        if isinstance(blk, tuple) and len(blk) == 2:
            name, shape = blk
        else:
            name, shape = None, blk
        positions = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if can_place(g, shape, r, c):
                    new_g = place_block(g, shape, r, c)
                    sc = score_grid(new_g)
                    positions.append((sc, r, c, new_g))
        # explore best scoring child states first
        positions.sort(key=lambda x: x[0], reverse=True)
        # cap branching for speed
        if len(positions) > MAX_BRANCHING_PER_BLOCK:
            positions = positions[:MAX_BRANCHING_PER_BLOCK]
        return positions

    for order in permutations(blocks):
        # DFS over placements for this order
        def dfs(g, idx, moves_so_far):
            nonlocal best_score, best_moves
            if idx == len(order):
                sc = score_grid(g)
                if sc > best_score:
                    best_score = sc
                    best_moves = list(moves_so_far)
                return

            blk = order[idx]
            # blk may be (name, shape)
            if isinstance(blk, tuple) and len(blk) == 2:
                name, shape = blk
            else:
                name, shape = None, blk
            positions = all_positions_sorted(g, blk)
            if not positions:
                return
            for _, r, c, new_g in positions:
                # Store (name, shape, r, c)
                moves_so_far.append((name, shape, r, c))
                dfs(new_g, idx + 1, moves_so_far)
                moves_so_far.pop()

        dfs(copy.deepcopy(grid), 0, [])

    return best_moves, best_score


# ------------------ UI ------------------
class BlockBlastSolver:
    def __init__(self, root):
        self.root = root
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.blocks = []
        self.dragging = False
        self.paint_value = 1  # 1 to fill, 0 to erase
        self.last_moves = None
        self.last_score = None
        # Vision state
        self.board_rect = None
        self.block_rects = []
        self.threshold = 160
        self._auto_detect = False
        self._last_signature = None
        self._vision_debug = False
        self._vision_top = None
        self._vision_imgs = {}
        # Vision thresholds (tunable) - calibrated defaults
        self.block_s_thr = 50
        self.block_v_thr = 185
        self.board_s_off = -50
        self.board_v_off = -1
        # Board occupancy decision: minimum fraction of pixels inside a cell
        # that must exceed S/V thresholds to mark the cell as filled.
        self.board_fill_ratio = 0.12
        # Plan lock: persist a solution until all three blocks are cleared
        self._plan_locked = False
        self._locked_grid = None
        self._locked_signature = None
        self._expected_grid = None

        self.canvas = tk.Canvas(
            root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, cursor="crosshair"
        )
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.draw_grid()

        # Mouse bindings for board editing
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Button-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_up)

        # Visual block palette on the right (scrollable container; we'll fit to content)
        self.palette_container = tk.Frame(root)
        self.palette_container.grid(row=0, column=4, sticky="nw")
        palette_height = GRID_SIZE * CELL_SIZE
        palette_width = 180
        self.palette_canvas = tk.Canvas(
            self.palette_container,
            width=palette_width,
            height=palette_height,
            highlightthickness=0,
        )
        self.palette_canvas.pack(side="left", fill="y")
        self.palette_scrollbar = tk.Scrollbar(
            self.palette_container, orient="vertical", command=self.palette_canvas.yview
        )
        self.palette_scrollbar.pack(side="right", fill="y")
        self.palette_canvas.configure(yscrollcommand=self.palette_scrollbar.set)
        self.palette_inner = tk.Frame(self.palette_canvas)
        self.palette_canvas.create_window((0, 0), window=self.palette_inner, anchor="nw")
        # Update scrollregion whenever inner frame resizes
        self.palette_inner.bind(
            "<Configure>",
            lambda e: self.palette_canvas.configure(scrollregion=self.palette_canvas.bbox("all")),
        )
        self.build_palette(self.palette_inner)
        # Fit palette to content (no scroll) and maximize window
        self.root.update_idletasks()
        self._fit_palette_to_content()
        self._maximize_window()

        # Controls row
        self.controls = tk.Frame(root)
        self.controls.grid(row=1, column=0, columnspan=5, sticky="we", padx=8, pady=8)
        # Make main grid columns expand
        try:
            root.grid_columnconfigure(0, weight=1)
            root.grid_columnconfigure(1, weight=0)
            root.grid_columnconfigure(2, weight=0)
            root.grid_columnconfigure(3, weight=0)
            root.grid_columnconfigure(4, weight=0)
        except Exception:
            pass

        tk.Button(self.controls, text="Solve", command=self.solve).pack(side="left", padx=6)
        tk.Button(self.controls, text="Apply Solution", command=self.apply_solution).pack(
            side="left", padx=6
        )
        tk.Button(self.controls, text="Clear Blocks", command=self.clear_blocks).pack(
            side="left", padx=6
        )
        tk.Button(self.controls, text="Reset Board", command=self.reset_board).pack(
            side="left", padx=6
        )
        tk.Button(self.controls, text="Clear Completed", command=self.clear_completed).pack(
            side="left", padx=6
        )
        tk.Button(self.controls, text="Calibrate", command=self.calibrate).pack(
            side="left", padx=12
        )
        tk.Button(self.controls, text="Detect Blocks", command=self.detect_blocks_once).pack(
            side="left", padx=6
        )
        tk.Button(self.controls, text="Auto On", command=lambda: self.set_auto(True)).pack(
            side="left", padx=6
        )
        tk.Button(self.controls, text="Auto Off", command=lambda: self.set_auto(False)).pack(
            side="left", padx=6
        )
        tk.Button(self.controls, text="Vision View", command=self.toggle_vision_view).pack(
            side="left", padx=12
        )
        tk.Button(self.controls, text="Tune Vision", command=self.open_tuning).pack(
            side="left", padx=6
        )
        self.selected_var = tk.StringVar(value="Selected: 0/3")
        tk.Label(self.controls, textvariable=self.selected_var).pack(side="left", padx=12)

    def draw_grid(self):
        self.canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                fill = "black" if self.grid[r][c] else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="gray")

    def add_block(self, name):
        if len(self.blocks) < 3:
            # store (name, shape)
            self.blocks.append((name, BLOCKS[name]))
        self.selected_var.set(f"Selected: {len(self.blocks)}/3")
        # ensure UI refresh is immediate
        self.root.update_idletasks()

    def clear_blocks(self):
        self.blocks = []
        if hasattr(self, "selected_var"):
            self.selected_var.set("Selected: 0/3")

    def solve(self):
        if len(self.blocks) != 3:
            print("Please select 3 blocks!")
            return
        moves, score = best_move(self.grid, self.blocks)
        if moves:
            print("Best moves:", [(n, r, c) for (n, _, r, c) in moves], "Score:", score)
            self.last_moves = moves
            self.last_score = score
            # highlight placements with step numbers
            step_colors = ["#e53935", "#fb8c00", "#1e88e5", "#8e24aa", "#43a047"]
            for step, (name, shape, r, c) in enumerate(moves, start=1):
                color = step_colors[(step - 1) % len(step_colors)]
                for dr, dc in shape:
                    rr, cc = r + dr, c + dc
                    x1, y1 = cc * CELL_SIZE, rr * CELL_SIZE
                    x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
                # draw step number near first cell
                rr0, cc0 = r + shape[0][0], c + shape[0][1]
                tx = cc0 * CELL_SIZE + CELL_SIZE // 2
                ty = rr0 * CELL_SIZE + CELL_SIZE // 2
                self.canvas.create_text(
                    tx, ty, text=str(step), fill=color, font=("Helvetica", 14, "bold")
                )
        else:
            print("No valid placement!")

    # --------- Board editing with mouse ---------
    def reset_board(self):
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.draw_grid()

    def _event_to_cell(self, event):
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c
        return None, None

    def _paint_cell(self, r, c, value):
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            self.grid[r][c] = value
            x1, y1 = c * CELL_SIZE, r * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            fill = "black" if value else "white"
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="gray")

    def on_left_down(self, event):
        r, c = self._event_to_cell(event)
        if r is None:
            return
        self.dragging = True
        self.paint_value = 1
        self._paint_cell(r, c, self.paint_value)

    def on_left_drag(self, event):
        if not self.dragging:
            return
        r, c = self._event_to_cell(event)
        if r is None:
            return
        self._paint_cell(r, c, self.paint_value)

    def on_left_up(self, event):
        self.dragging = False

    def on_right_down(self, event):
        r, c = self._event_to_cell(event)
        if r is None:
            return
        self.dragging = True
        self.paint_value = 0
        self._paint_cell(r, c, self.paint_value)

    def on_right_drag(self, event):
        if not self.dragging:
            return
        r, c = self._event_to_cell(event)
        if r is None:
            return
        self._paint_cell(r, c, self.paint_value)

    def on_right_up(self, event):
        self.dragging = False

    # --------- Vision: calibration & auto-detect ---------
    def calibrate(self):
        if ImageGrab is None:
            print("Pillow is required. Install with: pip install pillow")
            return
        # Take a full-resolution screenshot and display a scaled preview.
        try:
            shot = ImageGrab.grab()
        except Exception as e:
            print("Screenshot failed:", e)
            return
        shot_w, shot_h = shot.size
        # Logical screen size (Tk units)
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        # Scale factors from canvas coords -> screenshot pixels
        scale_x = shot_w / max(1, scr_w)
        scale_y = shot_h / max(1, scr_h)

        top = tk.Toplevel(self.root)
        top.title("Calibration: Drag 1) Board, 2) Block A, 3) Block B, 4) Block C  (Esc to cancel)")
        top.geometry(f"{scr_w}x{scr_h}+0+0")
        cv = tk.Canvas(top, width=scr_w, height=scr_h, bg="#222")
        cv.pack()
        try:
            top.lift()
            top.attributes("-topmost", True)
        except Exception:
            pass
        hint = tk.Label(
            top, text="Drag 4 boxes: Board, Block A, Block B, Block C", fg="#eee", bg="#333"
        )
        hint.place(x=12, y=12)
        # Show a scaled screenshot that fills the window
        if ImageTk is not None:
            try:
                disp = shot.resize((scr_w, scr_h))
                photo = ImageTk.PhotoImage(disp)
                cv.create_image(0, 0, image=photo, anchor="nw")
                top._photo_ref = photo  # keep reference
            except Exception as e:
                tk.Label(
                    top, text=f"Unable to show screenshot: {e}", fg="#ff5252", bg="#333"
                ).place(x=12, y=36)
        else:
            tk.Label(
                top,
                text="ImageTk not available. Install Pillow: pip install pillow",
                fg="#ffb300",
                bg="#333",
            ).place(x=12, y=36)
        state = {"start": None, "rects": [], "temp": None, "scale_x": scale_x, "scale_y": scale_y}

        def on_down(e):
            state["start"] = (e.x, e.y)
            if state["temp"]:
                cv.delete(state["temp"])
            state["temp"] = cv.create_rectangle(e.x, e.y, e.x, e.y, outline="#00e5ff", width=2)

        def on_drag(e):
            s = state["start"]
            if not s:
                return
            cv.coords(state["temp"], s[0], s[1], e.x, e.y)

        def on_up(e):
            s = state["start"]
            if not s:
                return
            # Map canvas coords back to screenshot pixels using scale factors
            x1, y1 = s[0], s[1]
            x2, y2 = e.x, e.y
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            sx = state["scale_x"]
            sy = state["scale_y"]
            xi1 = int(x1 * sx)
            yi1 = int(y1 * sy)
            xi2 = int(x2 * sx)
            yi2 = int(y2 * sy)
            rect = (xi1, yi1, xi2, yi2)
            state["rects"].append(rect)
            state["start"] = None
            state["temp"] = None
            if len(state["rects"]) >= 4:
                self.board_rect = state["rects"][0]
                self.block_rects = state["rects"][1:4]
                top.destroy()

        cv.bind("<Button-1>", on_down)
        cv.bind("<B1-Motion>", on_drag)
        cv.bind("<ButtonRelease-1>", on_up)
        top.bind("<Escape>", lambda e: top.destroy())

    def set_auto(self, enabled: bool):
        self._auto_detect = enabled
        if enabled:
            self.root.after(1000, self._tick_detect)

    def _tick_detect(self):
        if not self._auto_detect or ImageGrab is None:
            return
        if not self.block_rects or len(self.block_rects) != 3 or not self.board_rect:
            # not calibrated
            self.root.after(1200, self._tick_detect)
            return
        try:
            shot = ImageGrab.grab()
        except Exception:
            self.root.after(1200, self._tick_detect)
            return
        # Always read current board for comparison; only assign to self.grid when unlocked
        new_grid = None
        try:
            new_grid = self._detect_board_from_shot(shot)
        except Exception as e:
            print("[vision] board detect error:", e)
        if self._plan_locked:
            # If board matches expected result, release lock
            if new_grid is not None and self._expected_grid is not None:
                if self._grids_close(new_grid, self._expected_grid):
                    self._plan_locked = False
                    self._locked_grid = None
                    self._locked_signature = None
                    self._expected_grid = None
                    self.grid = new_grid
                    self.draw_grid()
            else:
                # Still locked; do not update grid from screen
                pass
        else:
            if new_grid is not None:
                self.grid = new_grid
                self.draw_grid()
        detected = []
        blocks_debug = []
        for rect in self.block_rects:
            name, shape = self._detect_block(shot.crop(rect))
            blocks_debug.append((name, shape))
            if not shape:
                detected = []
                break
            if name and name in BLOCKS:
                detected.append((name, BLOCKS[name]))
            else:
                detected.append((name or "?", shape))
        if len(detected) == 3:
            sig = tuple((n, tuple(s)) for n, s in detected)
            # Lock handling
            if not self._plan_locked:
                if sig != self._last_signature:
                    self._last_signature = sig
                    self.blocks = detected
                    self.selected_var.set(f"Selected: {len(self.blocks)}/3")
                    print("Detected blocks:", [n for n, _ in self.blocks])
                    # Solve once and lock plan and grid snapshot
                    if len(self.blocks) == 3:
                        # Use latest observed board snapshot (new_grid) for consistency
                        self._locked_grid = copy.deepcopy(
                            new_grid if new_grid is not None else self.grid
                        )
                        # Run solve using the locked grid snapshot
                        moves, score = best_move(self._locked_grid, self.blocks)
                        if moves:
                            self.last_moves = moves
                            self.last_score = score
                            self._plan_locked = True
                            self._locked_signature = sig
                            # Compute expected grid after applying planned moves
                            exp = copy.deepcopy(self._locked_grid)
                            for _, shape, r, c in moves:
                                exp = place_block(exp, shape, r, c)
                            self._expected_grid = exp
                            # Draw overlays based on current canvas grid
                            self.solve()
            else:
                # If currently locked, check whether all three blocks are cleared (blank)
                if self._blocks_all_blank(shot):
                    self._plan_locked = False
                    self._locked_grid = None
                    self._locked_signature = None
                    self._expected_grid = None
                    self.last_moves = None
                    self.last_score = None
                    # Reset selection for next cycle
                    self.blocks = []
                    self.selected_var.set("Selected: 0/3")
        # Optionally render debug
        if self._vision_debug:
            try:
                self._update_vision_view(shot, self.grid, [shot.crop(r) for r in self.block_rects])
            except Exception:
                pass
        self.root.after(1200, self._tick_detect)

    def detect_blocks_once(self):
        if ImageGrab is None:
            print("Pillow is required. Install with: pip install pillow")
            return
        if not self.block_rects or len(self.block_rects) != 3:
            print("Please calibrate first (board + three blocks)")
            return
        try:
            shot = ImageGrab.grab()
        except Exception as e:
            print("Screenshot failed:", e)
            return
        # Refresh board snapshot from the screen
        try:
            new_grid = self._detect_board_from_shot(shot)
            if new_grid is not None:
                self.grid = new_grid
                self.draw_grid()
        except Exception as e:
            print("[vision] board detect error:", e)
        detected = []
        for rect in self.block_rects:
            name, shape = self._detect_block(shot.crop(rect))
            if not shape:
                print("Block detection failed for one or more slots.")
                return
            if name and name in BLOCKS:
                detected.append((name, BLOCKS[name]))
            else:
                detected.append((name or "?", shape))
        self.blocks = detected
        self.selected_var.set(f"Selected: {len(self.blocks)}/3")
        self._last_signature = tuple((n, tuple(s)) for n, s in detected)
        print("Detected blocks:", [n for n, _ in self.blocks])
        # Compute a plan immediately and lock it until applied
        if len(self.blocks) == 3:
            self._locked_grid = copy.deepcopy(self.grid)
            moves, score = best_move(self._locked_grid, self.blocks)
            if moves:
                self.last_moves = moves
                self.last_score = score
                self._plan_locked = True
                self._locked_signature = self._last_signature
                # Expected board after applying the plan
                exp = copy.deepcopy(self._locked_grid)
                for _, shape, r, c in moves:
                    exp = place_block(exp, shape, r, c)
                self._expected_grid = exp
                # Draw overlays
                self.solve()

    def _grids_close(self, g1, g2, tol: int = 2):
        try:
            diff = 0
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if g1[r][c] != g2[r][c]:
                        diff += 1
                        if diff > tol:
                            return False
            return True
        except Exception:
            return False

    def _blocks_all_blank(self, shot):
        # Return True if all three block rects appear empty (blue background)
        try:
            for rect in self.block_rects:
                img = shot.crop(rect)
                w0, h0 = img.size
                ix1 = int(w0 * 0.12)
                iy1 = int(h0 * 0.12)
                ix2 = int(w0 * 0.88)
                iy2 = int(h0 * 0.88)
                inner = img.crop((ix1, iy1, ix2, iy2))
                hsv = inner.convert("HSV")
                _h, s, v = hsv.split()
                s_mask = s.point(lambda x, st=self.block_s_thr: 255 if x >= st else 0)
                v_mask = v.point(lambda x, vt=self.block_v_thr: 255 if x >= vt else 0)
                mask = (
                    ImageChops.logical_and(s_mask.convert("1"), v_mask.convert("1"))
                    if ImageChops is not None
                    else s_mask.convert("1")
                )
                bbox = mask.getbbox()
                if bbox:
                    return False
            return True
        except Exception:
            return False

    def toggle_vision_view(self):
        if self._vision_top is not None:
            try:
                self._vision_top.destroy()
            except Exception:
                pass
            self._vision_top = None
            self._vision_debug = False
            return
        self._vision_debug = True
        self._vision_top = tk.Toplevel(self.root)
        self._vision_top.title("Vision Debug")
        self._vis_board = tk.Canvas(self._vision_top, width=320, height=320, bg="#111")
        self._vis_board.grid(row=0, column=0, padx=6, pady=6)
        self._vis_blocks = [
            tk.Canvas(self._vision_top, width=140, height=180, bg="#111") for _ in range(3)
        ]
        for i, cv in enumerate(self._vis_blocks):
            cv.grid(row=0, column=1 + i, padx=6, pady=6)

    def _update_vision_view(self, shot, grid, block_images):
        if ImageTk is None:
            return
        if self.board_rect:
            br = self.board_rect
            roi = shot.crop(br).resize((320, 320))
            # overlay grid
            img = roi.convert("RGB")
            from PIL import ImageDraw as _ImageDraw

            draw = _ImageDraw.Draw(img)
            cell_w = 320 // GRID_SIZE
            cell_h = 320 // GRID_SIZE
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if grid and grid[r][c] == 1:
                        x1 = c * cell_w
                        y1 = r * cell_h
                        x2 = x1 + cell_w - 1
                        y2 = y1 + cell_h - 1
                        draw.rectangle([x1, y1, x2, y2], outline="#00e676")
            photo = ImageTk.PhotoImage(img)
            self._vision_imgs["board"] = photo
            self._vis_board.create_image(0, 0, image=photo, anchor="nw")
        for i in range(min(3, len(block_images))):
            cv = self._vis_blocks[i]
            src_img = block_images[i]
            disp = src_img.resize((120, 120)).convert("RGB")
            # Re-run lightweight detection to draw overlay squares
            dbg = self._detect_block_debug(src_img)
            if dbg is not None:
                from PIL import ImageDraw as _ImageDraw

                draw = _ImageDraw.Draw(disp)
                # Map 90x90 work-space bbox back into display coordinates of the inner area
                off = int(120 * 0.12)
                inner = 120 - 2 * off
                mx1, my1, mx2, my2 = dbg["bbox"]
                # BBox in display space
                bx1 = off + int(mx1 / 90.0 * inner)
                by1 = off + int(my1 / 90.0 * inner)
                bx2 = off + int(mx2 / 90.0 * inner)
                by2 = off + int(my2 / 90.0 * inner)
                # Draw bbox
                draw.rectangle([bx1, by1, bx2, by2], outline="#64b5f6")
                # Draw detected tiles using dynamically inferred grid runs
                row_runs = dbg.get("row_runs")
                col_runs = dbg.get("col_runs")
                cells = dbg.get("cells", [])
                if row_runs and col_runs and cells:
                    rows = max(1, len(row_runs))
                    cols = max(1, len(col_runs))
                    # Enforce square tiles: use the smaller of per-row/per-col cell size
                    cell_w = (bx2 - bx1) / float(cols)
                    cell_h = (by2 - by1) / float(rows)
                    s = min(cell_w, cell_h)
                    grid_w = s * cols
                    grid_h = s * rows
                    gx1 = int(round(bx1 + ((bx2 - bx1) - grid_w) / 2.0))
                    gy1 = int(round(by1 + ((by2 - by1) - grid_h) / 2.0))
                    pad = max(1, int(0.06 * s))
                    for rr, cc in cells:
                        x1 = int(round(gx1 + cc * s)) + pad
                        y1 = int(round(gy1 + rr * s)) + pad
                        x2 = int(round(gx1 + (cc + 1) * s)) - pad - 1
                        y2 = int(round(gy1 + (rr + 1) * s)) - pad - 1
                        draw.rectangle([x1, y1, x2, y2], outline="#00e676", width=2)
            photo = ImageTk.PhotoImage(disp)
            self._vision_imgs[f"blk{i}"] = photo
            cv.create_image(10, 10, image=photo, anchor="nw")

    def _detect_block_debug(self, img):
        try:
            if Image is None or ImageStat is None:
                return None
            # Mirror of _detect_block, but return cells and bbox for visualization
            w0, h0 = img.size
            ix1 = int(w0 * 0.12)
            iy1 = int(h0 * 0.12)
            ix2 = int(w0 * 0.88)
            iy2 = int(h0 * 0.88)
            inner = img.crop((ix1, iy1, ix2, iy2))
            work_size = 120
            work = inner.convert("RGB").resize((work_size, work_size))
            hsv = work.convert("HSV")
            _h, s, v = hsv.split()
            s_th = self.block_s_thr
            v_th = self.block_v_thr
            s_mask = s.point(lambda x, st=s_th: 255 if x >= st else 0)
            v_mask = v.point(lambda x, vt=v_th: 255 if x >= vt else 0)
            mask = (
                ImageChops.logical_and(s_mask.convert("1"), v_mask.convert("1"))
                if ImageChops is not None
                else s_mask.convert("1")
            )
            bbox = mask.getbbox()
            if not bbox:
                return {"cells": [], "bbox": (0, 0, 90, 90)}
            mx1, my1, mx2, my2 = bbox
            pad = 4
            mx1 = max(0, mx1 - pad)
            my1 = max(0, my1 - pad)
            mx2 = min(mask.size[0], mx2 + pad)
            my2 = min(mask.size[1], my2 + pad)
            # Work on cropped bbox with higher resolution for segmentation
            crop_w = 140
            crop_h = 140
            crop = mask.crop((mx1, my1, mx2, my2)).resize((crop_w, crop_h))
            px = crop.load()

            def projection_runs(size, is_col):
                vals = []
                if is_col:
                    for x in range(size[0]):
                        on = 0
                        for y in range(size[1]):
                            if px[x, y] != 0:
                                on += 1
                        vals.append(on / float(size[1]))
                else:
                    for y in range(size[1]):
                        on = 0
                        for x in range(size[0]):
                            if px[x, y] != 0:
                                on += 1
                        vals.append(on / float(size[0]))
                thr = 0.10
                active = [v >= thr for v in vals]
                min_len = max(2, int((size[0 if is_col else 1]) * 0.08))
                runs = []
                start = None
                for idx, a in enumerate(active + [False]):
                    if a and start is None:
                        start = idx
                    elif not a and start is not None:
                        if idx - start >= min_len:
                            runs.append((start, idx))
                        start = None
                if not runs:
                    runs = [(0, size[0 if is_col else 1])]
                return runs

            col_runs = projection_runs((crop_w, crop_h), True)
            row_runs = projection_runs((crop_w, crop_h), False)

            # Determine occupied cells by checking fill ratio within each run cell
            cells = []
            for r_idx, (ry1, ry2) in enumerate(row_runs):
                for c_idx, (cx1, cx2) in enumerate(col_runs):
                    # Use 10% inset to avoid borders
                    wx1 = int(cx1 + 0.1 * (cx2 - cx1))
                    wx2 = int(cx2 - 0.1 * (cx2 - cx1))
                    wy1 = int(ry1 + 0.1 * (ry2 - ry1))
                    wy2 = int(ry2 - 0.1 * (ry2 - ry1))
                    total = max(1, (wx2 - wx1) * (wy2 - wy1))
                    on = 0
                    for yy in range(wy1, wy2):
                        for xx in range(wx1, wx2):
                            if px[xx, yy] != 0:
                                on += 1
                    if (on / float(total)) >= 0.5:
                        cells.append((r_idx, c_idx))

            # Map cells to normalized grid for matching
            matched_name, matched_rot = match_block_name(cells, allow_mirror=False)
            return {
                "cells": cells,
                "bbox": (mx1, my1, mx2, my2),
                "matched_rot": matched_rot,
                "row_runs": row_runs,
                "col_runs": col_runs,
                "work_size": (crop_w, crop_h),
            }
        except Exception:
            return None

    def _detect_board_from_shot(self, shot):
        # Return GRID_SIZE x GRID_SIZE grid of 0/1
        br = self.board_rect
        if not br:
            return None
        img = shot.crop(br)
        w, h = img.size
        cw = w / GRID_SIZE
        ch = h / GRID_SIZE
        hsv = img.convert("HSV")
        _, s_img, v_img = hsv.split()
        # Background estimation via overall means (board dominates)
        s_stat = ImageStat.Stat(s_img)
        v_stat = ImageStat.Stat(v_img)
        s_bg = s_stat.mean[0]
        v_bg = v_stat.mean[0]
        s_thr = max(0, min(255, s_bg + self.board_s_off))
        v_thr = max(0, min(255, v_bg + self.board_v_off))
        grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1 = int(c * cw + cw * 0.25)
                y1 = int(r * ch + ch * 0.25)
                x2 = int((c + 1) * cw - cw * 0.25)
                y2 = int((r + 1) * ch - ch * 0.25)
                if x2 <= x1 or y2 <= y1:
                    x1 = int(c * cw + cw * 0.4)
                    y1 = int(r * ch + ch * 0.4)
                    x2 = int((c + 1) * cw - cw * 0.4)
                    y2 = int((r + 1) * ch - ch * 0.4)
                s_patch = s_img.crop((x1, y1, x2, y2))
                v_patch = v_img.crop((x1, y1, x2, y2))
                # Decision by fraction of pixels above thresholds (robust to decorated tiles)
                spx = s_patch.load()
                vpx = v_patch.load()
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                hits = 0
                total = 0
                step = 2  # subsample for speed
                margin = max(1, min(w, h) // 10)
                for yy in range(margin, h - margin, step):
                    for xx in range(margin, w - margin, step):
                        total += 1
                        if spx[xx, yy] >= s_thr and vpx[xx, yy] >= v_thr:
                            hits += 1
                ratio = (hits / float(total)) if total else 0.0
                if ratio >= self.board_fill_ratio:
                    grid[r][c] = 1
                else:
                    # Secondary check: unusually bright/contrasty patch vs background
                    v_stat = ImageStat.Stat(v_patch)
                    if (v_stat.mean[0] - v_bg) >= 12 and v_stat.stddev[0] >= 10:
                        grid[r][c] = 1
        return grid

    def _detect_block(self, img):
        try:
            if Image is None or ImageStat is None:
                return None, None
            w0, h0 = img.size
            ix1 = int(w0 * 0.12)
            iy1 = int(h0 * 0.12)
            ix2 = int(w0 * 0.88)
            iy2 = int(h0 * 0.88)
            inner = img.crop((ix1, iy1, ix2, iy2))
            work_size = 120
            work = inner.convert("RGB").resize((work_size, work_size))
            hsv = work.convert("HSV")
            _h, s, v = hsv.split()
            s_th = self.block_s_thr
            v_th = self.block_v_thr
            s_mask = s.point(lambda x, st=s_th: 255 if x >= st else 0)
            v_mask = v.point(lambda x, vt=v_th: 255 if x >= vt else 0)
            mask = (
                ImageChops.logical_and(s_mask.convert("1"), v_mask.convert("1"))
                if ImageChops is not None
                else s_mask.convert("1")
            )
            bbox = mask.getbbox()
            if not bbox:
                return None, None
            mx1, my1, mx2, my2 = bbox
            pad = 4
            mx1 = max(0, mx1 - pad)
            my1 = max(0, my1 - pad)
            mx2 = min(mask.size[0], mx2 + pad)
            my2 = min(mask.size[1], my2 + pad)
            crop_w = 140
            crop_h = 140
            crop = mask.crop((mx1, my1, mx2, my2)).resize((crop_w, crop_h))
            px = crop.load()

            def projection_runs(size, is_col):
                vals = []
                if is_col:
                    for x in range(size[0]):
                        on = 0
                        for y in range(size[1]):
                            if px[x, y] != 0:
                                on += 1
                        vals.append(on / float(size[1]))
                else:
                    for y in range(size[1]):
                        on = 0
                        for x in range(size[0]):
                            if px[x, y] != 0:
                                on += 1
                        vals.append(on / float(size[0]))
                thr = 0.10
                active = [v >= thr for v in vals]
                min_len = max(2, int((size[0 if is_col else 1]) * 0.08))
                runs = []
                start = None
                for idx, a in enumerate(active + [False]):
                    if a and start is None:
                        start = idx
                    elif not a and start is not None:
                        if idx - start >= min_len:
                            runs.append((start, idx))
                        start = None
                if not runs:
                    runs = [(0, size[0 if is_col else 1])]
                return runs

            col_runs = projection_runs((crop_w, crop_h), True)
            row_runs = projection_runs((crop_w, crop_h), False)

            cells = []
            for r_idx, (ry1, ry2) in enumerate(row_runs):
                for c_idx, (cx1, cx2) in enumerate(col_runs):
                    wx1 = int(cx1 + 0.1 * (cx2 - cx1))
                    wx2 = int(cx2 - 0.1 * (cx2 - cx1))
                    wy1 = int(ry1 + 0.1 * (ry2 - ry1))
                    wy2 = int(ry2 - 0.1 * (ry2 - ry1))
                    total = max(1, (wx2 - wx1) * (wy2 - wy1))
                    on = 0
                    for yy in range(wy1, wy2):
                        for xx in range(wx1, wx2):
                            if px[xx, yy] != 0:
                                on += 1
                    if (on / float(total)) >= 0.5:
                        cells.append((r_idx, c_idx))
            if not cells:
                return None, None
            name, rot = match_block_name(cells, allow_mirror=False)
            if name is not None:
                return name, rot
            # No match; return normalized cells for fallback
            return None, normalize_shape(cells)
        except Exception as e:
            print("[vision] detect error:", e)
            return None, None

    def open_tuning(self):
        win = tk.Toplevel(self.root)
        win.title("Vision Tuning")
        # Block thresholds
        tk.Label(win, text="Block Saturation Threshold").grid(row=0, column=0, sticky="w")
        bs = tk.Scale(win, from_=0, to=255, orient="horizontal")
        bs.set(self.block_s_thr)
        bs.grid(row=0, column=1, sticky="we")
        tk.Label(win, text="Block Value Threshold").grid(row=1, column=0, sticky="w")
        bv = tk.Scale(win, from_=0, to=255, orient="horizontal")
        bv.set(self.block_v_thr)
        bv.grid(row=1, column=1, sticky="we")
        # Board offsets
        tk.Label(win, text="Board S offset").grid(row=2, column=0, sticky="w")
        bso = tk.Scale(win, from_=-50, to=100, orient="horizontal")
        bso.set(self.board_s_off)
        bso.grid(row=2, column=1, sticky="we")
        tk.Label(win, text="Board V offset").grid(row=3, column=0, sticky="w")
        bvo = tk.Scale(win, from_=-50, to=100, orient="horizontal")
        bvo.set(self.board_v_off)
        bvo.grid(row=3, column=1, sticky="we")
        tk.Label(win, text="Board fill ratio (0.05-0.3)").grid(row=4, column=0, sticky="w")
        bfr = tk.Scale(win, from_=5, to=30, orient="horizontal")
        bfr.set(int(self.board_fill_ratio * 100))
        bfr.grid(row=4, column=1, sticky="we")

        # Apply on change
        def apply_changes(*_):
            self.block_s_thr = bs.get()
            self.block_v_thr = bv.get()
            self.board_s_off = bso.get()
            self.board_v_off = bvo.get()
            self.board_fill_ratio = max(0.01, min(0.4, bfr.get() / 100.0))

        for s in (bs, bv, bso, bvo):
            s.configure(command=lambda _val: apply_changes())
        bfr.configure(command=lambda _val: apply_changes())
        win.grid_columnconfigure(1, weight=1)

    # --------- Block palette ---------
    def build_palette(self, parent):
        tk.Label(parent, text="Blocks").grid(row=0, column=0, columnspan=PALETTE_COLS, sticky="w")
        PREVIEW_SIZE = 84
        PREVIEW_CELL = 14
        for idx, (name, shape) in enumerate(BLOCKS.items()):
            row = 1 + idx // PALETTE_COLS
            col = idx % PALETTE_COLS
            frame = tk.Frame(parent, cursor="hand2")
            frame.grid(row=row, column=col, sticky="w", padx=6, pady=6)
            cv = tk.Canvas(
                frame,
                width=PREVIEW_SIZE,
                height=PREVIEW_SIZE,
                highlightthickness=1,
                highlightbackground="#ccc",
            )
            cv.pack(side="left")
            self._draw_block_preview(cv, shape, PREVIEW_SIZE, PREVIEW_CELL)
            lbl = tk.Label(frame, text=name)
            lbl.pack(side="left", padx=6)
            # Bind click on either canvas or label to add the block
            cv.bind("<Button-1>", lambda e, n=name: self.add_block(n))
            lbl.bind("<Button-1>", lambda e, n=name: self.add_block(n))
            frame.bind("<Button-1>", lambda e, n=name: self.add_block(n))

    def _draw_block_preview(self, canvas, block, size, cell):
        # Compute bounds
        min_r = min(r for r, _ in block)
        min_c = min(c for _, c in block)
        max_r = max(r for r, _ in block)
        max_c = max(c for _, c in block)
        h = (max_r - min_r + 1) * cell
        w = (max_c - min_c + 1) * cell
        # Center within preview
        off_y = (size - h) // 2
        off_x = (size - w) // 2
        for r, c in block:
            rr = r - min_r
            cc = c - min_c
            x1 = off_x + cc * cell
            y1 = off_y + rr * cell
            x2 = x1 + cell - 1
            y2 = y1 + cell - 1
            canvas.create_rectangle(x1, y1, x2, y2, fill="#4da3ff", outline="#1d6fd6")

    def _fit_palette_to_content(self):
        try:
            req_h = self.palette_inner.winfo_reqheight()
            req_w = self.palette_inner.winfo_reqwidth()
            self.palette_canvas.configure(height=req_h, width=req_w)
            # Hide scrollbar if not needed
            self.palette_scrollbar.pack_forget()
        except Exception:
            pass

    def _maximize_window(self):
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            # Leave tiny margin
            self.root.geometry(f"{sw}x{sh}+0+0")
        except Exception:
            pass

    # --------- Board utilities ---------
    def clear_completed(self):
        new_grid, cleared = clear_lines(self.grid)
        self.grid = new_grid
        self.draw_grid()
        if cleared:
            self.root.title(f"BlockBlast Solver - Cleared {cleared} line(s)")
        else:
            self.root.title("BlockBlast Solver - No lines to clear")

    def apply_solution(self):
        if not self.last_moves:
            print("No solution computed yet. Click Solve first.")
            return
        g = self.grid
        for _, shape, r, c in self.last_moves:
            g = place_block(g, shape, r, c)
        self.grid = g
        self.draw_grid()
        # reset selection and last moves
        self.blocks = []
        self.selected_var.set("Selected: 0/3")
        self.last_moves = None
        self.last_score = None
        self.root.title("BlockBlast Solver - Applied solution")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("BlockBlast Solver")
    app = BlockBlastSolver(root)
    root.mainloop()
