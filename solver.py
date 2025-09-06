import tkinter as tk
from itertools import permutations
import copy

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
    "WL2": [(0, 0), (0, 1)],
    "WL3": [(0, 0), (0, 1), (0, 2)],
    "WL4": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "WL5": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "WH2": [(0, 0), (1, 0)],
    "WH3": [(0, 0), (1, 0), (2, 0)],
    "WH4": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "WH5": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
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
    "T4": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)],
    "S3": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "X3": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "W3": [(0, 1), (1, 0), (1, 1), (2, 0)],
    "Z3": [(0, 0), (0, 1), (1, 1), (1, 2)],
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
