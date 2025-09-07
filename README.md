BlockBlast Solver
=================

A Tkinter GUI to plan BlockBlast-like moves:
- Paint the current board by dragging the mouse
- Pick 3 blocks from a visual palette (shape previews)
- Compute the exact placement order (1 → 2 → 3) using a heuristic search
- Visualize suggested placements with distinct colors and step numbers
- Apply the solution to update the board and clear completed lines
 - Optional computer-vision to auto-detect the board and the 3 offered blocks

Quick Start
-----------

Requirements: Python 3.9+ with Tkinter available.

Run:

```bash
python solver.py
```

Tip: If you use the provided virtualenv, activate it first:

```bash
source venv/bin/activate
```

UI & Controls
-------------

- Board editing
  - Left click + drag: fill cells
  - Right click + drag: erase cells
- Blocks palette (right side)
  - Click any preview or label to select a block
  - You can select exactly 3 blocks (see “Selected: X/3”)
- Controls (bottom bar)
  - Solve: compute the best 3-step placement order for the selected blocks
  - Apply Solution: place the suggested blocks on the board (lines auto-clear)
  - Clear Blocks: reset the selection to 0/3
  - Reset Board: clear the entire board
  - Clear Completed: clear any currently complete rows/columns
  - Calibrate: open a full-screen screenshot to drag 4 rectangles: Board, Block A, Block B, Block C
  - Detect Blocks: take one screenshot, detect A/B/C, run Solve once and lock the plan
  - Auto On/Off: continuous detection loop that updates the board and blocks
  - Vision View: opens a debug window with overlays for board and blocks
  - Tune Vision: sliders to tweak thresholds for robust detection

How It Solves
-------------

The solver performs a depth-first search over placements for the selected 3 blocks, exploring the best-scoring child states first and capping branching per block. The heuristic:
- Rewards more empty cells and longer empty runs
- Rewards near-complete lines (0–2 empty cells)
- Penalizes isolated empty holes

You can tune performance via `MAX_BRANCHING_PER_BLOCK` in `solver.py`.

Computer Vision (Auto-detect)
-----------------------------

Overview
- Calibration: press Calibrate and drag 4 boxes on the full-screen preview in this order: the Board, then Block A, Block B, Block C.
- Detection: either press Detect Blocks once, or use Auto On for continuous updates.
- Vision View shows:
  - Board: current occupied cells highlighted.
  - Blocks: each block region with a light-blue bounding box and green squares for detected tiles.

Plan Lock
- When Detect Blocks (or Auto) finds 3 blocks, the solver computes a plan once and locks it until applied.
- While locked:
  - The plan persists even if the on-screen board changes.
  - The UI does not re-solve.
  - The lock releases automatically when either:
    - The three block slots are empty (new set arrived), or
    - The observed board matches the solver’s expected end-state after applying the plan.

Tuning
- Block thresholds: Saturation/Value sliders control block tile extraction.
- Board offsets: S/V offsets bias the board fill thresholds against background.
- Board fill ratio: minimum fraction of interior pixels that must exceed thresholds to mark a cell as filled (useful for decorated/numbered tiles).

Implementation Notes
- Shapes are recognized by normalizing to a compact grid and comparing against all rotations of dictionary templates (mirrors optional).
- The palette reflects the `BLOCKS` mapping. Add/rename shapes and the UI updates automatically.

Customization
-------------

Edit `solver.py`:
- `GRID_SIZE`: board dimensions (default 8)
- `CELL_SIZE`: size of each grid cell in pixels
- `PALETTE_COLS`: number of columns in the blocks palette grid (default 5)
- `MAX_BRANCHING_PER_BLOCK`: exploration cap per block

Blocks live in the `BLOCKS` dictionary as lists of `(row, col)` offsets. Add or rename shapes; the palette updates automatically.

Troubleshooting
---------------

- If solving feels slow, reduce `MAX_BRANCHING_PER_BLOCK`.
- If Tkinter import fails, install your Python’s Tk bindings (varies by OS).
 - If detection misses decorated tiles, lower Board fill ratio a little and/or nudge Board S/V offsets.
 - If blocks are misnamed, use Vision View to verify green tiles align; adjust Block S/V thresholds.

