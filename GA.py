"""
Greedy Search for the Eight-Queen Problem
AACS3273 Fundamentals of Artificial Intelligence

------------------------------------------------------------------
Formulation (see Report Section 3.3 "Greedy Search")
------------------------------------------------------------------
State       : a list of column indices, one per row, built up row by
              row. e.g. state = [3, 7, 0] means a queen has been
              placed in column 3 of row 0 and column 7 of row 1 and
              column 0 of row 2; rows 3-7 are still empty.
Initial state: an empty board -> state = []
Goal test   : len(state) == 8 (a state is only ever pushed onto the
              priority queue if it is already conflict-free, so reaching
              length 8 always means a valid, complete solution).
Frontier    : a priority queue (min-heap) ordered by heuristic value
              h(n) = number of remaining rows to fill. This makes Greedy
              Search always expand the node that appears closest to the
              goal (i.e., the deepest non-conflicting state).

------------------------------------------------------------------
How the 10 datasets (Report Section 3.0 "Data Set") are used
------------------------------------------------------------------
Each dataset is a full permutation of 8 columns (e.g. [3, 7, 0, 4, 6,
1, 5, 2]). That already rules out row/column conflicts, but most sets
still have one or more queens sharing a diagonal (only Set 1 happens
to have zero conflicts) -- this is the "complete-state" formulation,
normally used to seed local-search algorithms such as Hill-Climbing
or a Genetic Algorithm, which start from a fully-placed board and
repair conflicts from there.

Greedy Search instead uses the "incremental" formulation described above,
and always starts from an EMPTY board, so a conflicting full permutation
can never appear as a node in Greedy Search's own search tree. To still
put each of the 10 datasets to genuine use, this program uses each one
to set the COLUMN TRIAL ORDER that Greedy Search follows at every row
of that run (instead of always trying columns 0, 1, 2, ... 7, the run
for Set N tries columns in the order Set N lists them). This is still
standard Greedy Search -- the order successors are generated in does not
change the algorithm -- it just gives each of the 10 runs a different
path through the same search tree, so the node counts and timings
reported below are meaningfully different and comparable across runs.
------------------------------------------------------------------
"""

import time
import heapq

BOARD_SIZE = 8

# The 10 datasets exactly as listed in Report Section 3.0 "Data Set".
DATA_SETS = {
    1: [3, 7, 0, 4, 6, 1, 5, 2],
    2: [6, 2, 5, 1, 7, 4, 0, 3],
    3: [1, 5, 7, 3, 0, 6, 2, 4],
    4: [4, 0, 6, 2, 5, 7, 3, 1],
    5: [7, 3, 1, 6, 4, 0, 2, 5],
    6: [2, 6, 4, 0, 3, 5, 7, 1],
    7: [5, 1, 3, 7, 2, 6, 4, 0],
    8: [0, 6, 2, 5, 1, 7, 3, 4],
    9: [4, 7, 5, 1, 3, 0, 6, 2],
    10: [7, 2, 0, 5, 4, 1, 3, 6],
}


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


def greedy_solve(column_order=None):
    """
    Greedy Search for a single Eight-Queen solution.

    column_order : the order in which columns are tried at every row.
                   Defaults to ascending order [0, 1, ..., 7].

    Returns (solution, nodes_expanded):
        solution       - a list of 8 column indices (one per row).
        nodes_expanded - number of states popped off the priority queue,
                         used as Greedy Search's search-effort metric.
    """
    if column_order is None:
        column_order = list(range(BOARD_SIZE))

    # Priority queue: (heuristic_value, tie_breaker_counter, state)
    # Heuristic: h(n) = -len(state) (negative because we want deeper states first)
    # This makes Greedy Search always expand the deepest non-conflicting state.
    counter = 0
    priority_queue = [(-0, counter, [])]  # h(empty) = 0
    nodes_expanded = 0

    while priority_queue:
        neg_depth, _, state = heapq.heappop(priority_queue)
        nodes_expanded += 1

        if len(state) == BOARD_SIZE:
            return state, nodes_expanded

        # Generate children for the next row in the given column order
        for col in column_order:
            if is_safe(state, col):
                new_state = state + [col]
                counter += 1
                # Heuristic: deeper states are preferred (closer to goal)
                # Using negative depth as priority value (higher depth = higher priority)
                heapq.heappush(priority_queue, (-len(new_state), counter, new_state))

    return None, nodes_expanded


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
    same way as Report Section 3.0 (row 0 = rank 8 at the top, row 7 =
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
    print("GREEDY SEARCH - EIGHT-QUEEN PROBLEM")
    print("=" * 60)

    results = []

    for set_id, dataset in DATA_SETS.items():
        initial_conflicts = count_conflicts(dataset)
        bad_rows = conflicting_rows(dataset)

        print(f"\n{'-' * 60}")
        print(f"SET {set_id}")
        print(f"{'-' * 60}")
        print(f"\nDataset (column order used)  : {dataset}")
        print(
            f"Initial conflicts in dataset : {initial_conflicts} (queens involved marked *Q*)")
        print_board(dataset, flagged_rows=bad_rows)

        start_time = time.perf_counter()
        solution, nodes_expanded = greedy_solve(column_order=dataset)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"\nGreedy Search solution        : {solution}")
        print(f"Nodes expanded                : {nodes_expanded}")
        print(f"Time taken                    : {elapsed_ms:.4f} ms")
        print_board(solution)

        results.append((set_id, initial_conflicts, nodes_expanded, elapsed_ms))

    # Summary table, handy for the report's "Results" section.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Set':<5}{'Initial Conflicts':<20}{'Nodes Expanded':<18}{'Time (ms)':<12}")
    for set_id, initial_conflicts, nodes_expanded, elapsed_ms in results:
        print(
            f"{set_id:<5}{initial_conflicts:<20}{nodes_expanded:<18}{elapsed_ms:<12.4f}")


if __name__ == "__main__":
    main()
