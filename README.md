BlockBlast Solver
=================

A Tkinter GUI to plan BlockBlast-like moves:
- Paint the current board by dragging the mouse
- Pick 3 blocks from a visual palette (shape previews)
- Compute the exact placement order (1 → 2 → 3) using a heuristic search
- Visualize suggested placements with distinct colors and step numbers
- Apply the solution to update the board and clear completed lines

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

How It Solves
-------------

The solver performs a depth-first search over placements for the selected 3 blocks, exploring the best-scoring child states first and capping branching per block. The heuristic:
- Rewards more empty cells and longer empty runs
- Rewards near-complete lines (0–2 empty cells)
- Penalizes isolated empty holes

You can tune performance via `MAX_BRANCHING_PER_BLOCK` in `solver.py`.

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

