"""
Depth-First Search (DFS) for the Eight-Queen Problem
AMCS2104 Fundamentals of Artificial Intelligence

------------------------------------------------------------------
Formulation (see Report Section 3.2 "Depth-First Search (DFS)")
------------------------------------------------------------------
State       : a list of column indices, one per row, built up row by
              row. e.g. state = [3, 7, 0] means a queen has been
              placed in column 3 of row 0 and column 7 of row 1 and
              column 0 of row 2; rows 3-7 are still empty.
Initial state: an empty board -> state = []
Goal test   : len(state) == 8 AND the state matches GOAL_STATE exactly
              (see below for why the goal is one fixed target rather
              than "any" complete arrangement).
Frontier    : a Last-In-First-Out (LIFO) stack, so DFS always expands
              the deepest node generated so far, and backtracks
              automatically whenever a row has no safe column left.

------------------------------------------------------------------
Why one shared GOAL_STATE, instead of stopping at any solution
------------------------------------------------------------------
For BFS, DFS, and Greedy Search to be compared fairly, all three need
to be doing the same job. If each algorithm simply stopped at the
first complete arrangement it happened to reach, differences in nodes
expanded / time / memory would partly reflect "which of the 92
solutions did this search order bump into first" rather than purely
the algorithm's search strategy. So instead, all three target the
SAME specific solution:

    GOAL_STATE = [3, 7, 0, 4, 6, 1, 5, 2]

This is exactly Set 1 from Report Section 4.1 "Data Set", which is
already a conflict-free arrangement, so it is guaranteed reachable.
Meaning: row 0's queen is in column 3, row 1's queen is in column 7,
row 2's queen is in column 0, and so on. If a teammate's BFS or Greedy
code represents a state the other way around (column index -> row
value), use the transposed version [0, 6, 4, 7, 1, 3, 5, 2] instead --
same physical board, just written for that indexing convention.

------------------------------------------------------------------
How the 10 datasets (Report Section 4.1 "Data Set") are used
------------------------------------------------------------------
Each dataset is a full permutation of 8 columns. That already rules
out row/column conflicts, but most sets still have one or more queens
sharing a diagonal (only Set 1 happens to have zero conflicts). DFS
uses each dataset to set the COLUMN TRIAL ORDER it follows at every
row (instead of always trying columns 0, 1, 2, ... 7, the run for Set
N tries columns in the order Set N lists them). This is still standard
DFS -- the order successors are generated in does not change the
algorithm -- it just gives each of the 10 runs a different path
through the same search tree on the way to the same GOAL_STATE, so the
results are meaningfully different and comparable across runs.

------------------------------------------------------------------
Metrics
------------------------------------------------------------------
nodes_expanded    - TIME complexity metric (total search effort,
                     corresponds to the theoretical O(b^m)).
max_stack_size    - SPACE complexity metric, node-count version (peak
                     number of states held in memory at once,
                     corresponds to the theoretical O(bm)).
peak_memory_kb    - SPACE complexity metric, actual measured version
                     (real bytes allocated by the search, via Python's
                     built-in tracemalloc module, converted to KB).
                     max_stack_size and peak_memory_kb should rise and
                     fall together, since more states held in memory
                     is exactly what uses more actual memory -- they
                     are just two different units for the same idea.
------------------------------------------------------------------
"""

import time
import tracemalloc

BOARD_SIZE = 8

# The 10 datasets exactly as listed in Report Section 4.1 "Data Set".
DATA_SETS = {
    1:  [3, 7, 0, 4, 6, 1, 5, 2],
    2:  [6, 2, 5, 1, 7, 4, 0, 3],
    3:  [1, 5, 7, 3, 0, 6, 2, 4],
    4:  [4, 0, 6, 2, 5, 7, 3, 1],
    5:  [7, 3, 1, 6, 4, 0, 2, 5],
    6:  [2, 6, 4, 0, 3, 5, 7, 1],
    7:  [5, 1, 3, 7, 2, 6, 4, 0],
    8:  [0, 6, 2, 5, 1, 7, 3, 4],
    9:  [4, 7, 5, 1, 3, 0, 6, 2],
    10: [7, 2, 0, 5, 4, 1, 3, 6],
}

# Shared target for BFS, DFS, and Greedy Search -- see note above.
GOAL_STATE = [3, 7, 0, 4, 6, 1, 5, 2]


def count_conflicts(perm):
    """Number of diagonally-attacking queen pairs in a full permutation
    (used only to report how 'conflicted' each dataset starts out)."""
    n = len(perm)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(perm[i] - perm[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def is_safe(state, col):
    """True if placing a queen in `col` of the next row conflicts with
    none of the queens already placed in `state`."""
    row = len(state)
    for r, c in enumerate(state):
        if c == col or abs(c - col) == abs(r - row):
            return False
    return True


def dfs_solve(column_order=None, target=None):
    """
    Depth-First (backtracking) search for the Eight-Queen problem.

    column_order : the order in which columns are tried at every row.
                    Defaults to ascending order [0, 1, ..., 7].
    target        : the exact solution to search for. If None, DFS
                    stops at the first complete, conflict-free state
                    it reaches (the old "any solution" behaviour). If
                    given (e.g. GOAL_STATE), DFS keeps searching past
                    any OTHER complete state until it finds this exact
                    one -- so node counts will generally be higher
                    than "any solution" mode, since the search can no
                    longer stop early at a different valid solution.

    Returns (solution, nodes_expanded, max_stack_size, peak_memory_kb).
    """
    if column_order is None:
        column_order = list(range(BOARD_SIZE))

    tracemalloc.start()

    stack = [[]]             # LIFO frontier, starts with the empty board
    nodes_expanded = 0
    max_stack_size = len(stack)

    while stack:
        state = stack.pop()              # expand the deepest (newest) node
        nodes_expanded += 1

        if len(state) == BOARD_SIZE:
            if target is None or state == target:
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                return state, nodes_expanded, max_stack_size, peak_bytes / 1024
            continue  # complete, but not the target -- dead end, backtrack

        # Generate children for the next row in the given column order,
        # pushing only ones that don't conflict (pruned at generation time).
        for col in column_order:
            if is_safe(state, col):
                stack.append(state + [col])

        max_stack_size = max(max_stack_size, len(stack))

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return None, nodes_expanded, max_stack_size, peak_bytes / 1024


def conflicting_rows(perm):
    """Set of row indices that take part in at least one diagonal conflict
    (used only to flag those queens when drawing a dataset's own board)."""
    bad = set()
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(perm[i] - perm[j]) == abs(i - j):
                bad.add(i)
                bad.add(j)
    return bad


def print_board(state, flagged_rows=None):
    """Chess-style board rendering, files a-h and ranks 1-8 labelled the
    same way as Report Section 4.1 (row 0 = rank 8 at the top, row 7 =
    rank 1 at the bottom). Queens that take part in a diagonal conflict
    are shown as *Q* instead of Q, so a dataset's own conflicts are
    visible at a glance."""
    flagged_rows = flagged_rows or set()
    files_header = "    " + "   ".join("abcdefgh")
    border = "  +" + "+".join(["---"] * BOARD_SIZE) + "+"

    print(files_header)
    print(border)
    for row in range(BOARD_SIZE):
        rank = BOARD_SIZE - row
        cells = []
        for col in range(BOARD_SIZE):
            if state[row] == col:
                cells.append("*Q*" if row in flagged_rows else " Q ")
            elif (row + col) % 2 == 0:
                cells.append("...")
            else:
                cells.append("   ")
        print(f"{rank} |" + "|".join(cells) + "|")
        print(border)
    print(files_header)


def main():
    print("=" * 60)
    print("DEPTH-FIRST SEARCH (DFS) - EIGHT-QUEEN PROBLEM")
    print("=" * 60)

    print(f"\nShared target GOAL_STATE (same one BFS and Greedy aim for): {GOAL_STATE}")
    print_board(GOAL_STATE)

    results = []

    for set_id, dataset in DATA_SETS.items():
        initial_conflicts = count_conflicts(dataset)
        bad_rows = conflicting_rows(dataset)

        print(f"\n{'-' * 60}")
        print(f"SET {set_id}")
        print(f"{'-' * 60}")
        print(f"\nDataset (column order used)  : {dataset}")
        print(f"Initial conflicts in dataset : {initial_conflicts} (queens involved marked *Q*)")
        print(f"\nInitial state board:")
        print_board(dataset, flagged_rows=bad_rows)

        start_time = time.perf_counter()
        solution, nodes_expanded, max_stack_size, peak_memory_kb = dfs_solve(
            column_order=dataset, target=GOAL_STATE
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"\nResult board:")
        print_board(solution)
        print(f"\nReached target GOAL_STATE      : {solution == GOAL_STATE}")
        print(f"Nodes expanded   (time metric) : {nodes_expanded}")
        print(f"Time taken       (time metric) : {elapsed_ms:.4f} ms")
        print(f"Max stack size  (space metric) : {max_stack_size}")
        print(f"Peak memory     (space metric) : {peak_memory_kb:.2f} KB")

        results.append((set_id, initial_conflicts, nodes_expanded, elapsed_ms, max_stack_size, peak_memory_kb))

    # Summary table, handy for the report's "Results" section.
    # Nodes Expanded / Time (ms)       -> empirical TIME complexity
    # Max Stack Size / Peak Memory KB  -> empirical SPACE complexity
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Set':<5}{'Init.Conflicts':<16}{'Nodes':<10}{'Time(ms)':<12}{'MaxStack':<12}{'PeakMem(KB)':<12}")
    for set_id, initial_conflicts, nodes_expanded, elapsed_ms, max_stack_size, peak_memory_kb in results:
        print(f"{set_id:<5}{initial_conflicts:<16}{nodes_expanded:<10}{elapsed_ms:<12.4f}{max_stack_size:<12}{peak_memory_kb:<12.2f}")


if __name__ == "__main__":
    main()
