"""
Hill-Climbing Search for the Eight-Queen Problem
AACS3273 Fundamentals of Artificial Intelligence

------------------------------------------------------------------
Formulation (see Report Section 3.4 "Hill-Climbing Search")
------------------------------------------------------------------
State       : a full permutation of 8 column indices, one per row,
              representing a complete board configuration. e.g.
              [3, 7, 0, 4, 6, 1, 5, 2] means a queen in row 0 at
              column 3, row 1 at column 7, etc.
Initial state: a randomly generated permutation of 0..7, or one of
              the 10 datasets provided in Report Section 3.0.
Goal test   : number of conflicts == 0 (no two queens share a row,
              column, or diagonal). Note: since we use permutations,
              row and column conflicts are already impossible.
Successor   : generate all neighbors by swapping two rows' columns.
              The best neighbor (with fewest conflicts) is chosen.
Frontier    : Hill-Climbing is a local search algorithm that does
              not maintain a frontier of multiple paths; it only
              keeps track of the current state and its neighbors.

------------------------------------------------------------------
How the 10 datasets (Report Section 3.0 "Data Set") are used
------------------------------------------------------------------
Each dataset is a full permutation of 8 columns. This matches
Hill-Climbing's complete-state formulation perfectly. The algorithm
starts from each dataset as its initial state and attempts to repair
it to a conflict-free solution by making incremental improvements.

Unlike DFS/BFS/Greedy Search which start from an empty board, Hill-
Climbing uses the dataset as the actual starting configuration. This
is appropriate because Hill-Climbing is a local search algorithm
designed to work with complete states. The starting conflicts of each
dataset determine the difficulty of finding a solution and influence
the number of steps and time required.

Note: Hill-Climbing may get stuck in local optima, and may not find
a solution for some starting configurations. The program will report
whether a solution was found or if it terminated at a local optimum.
------------------------------------------------------------------
"""

import time
import random

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
    """
    Count the number of diagonal conflicts in a permutation-based state.
    Since we use permutations, row and column conflicts are zero by
    construction; only diagonal conflicts matter.
    """
    n = len(perm)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(perm[i] - perm[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def generate_neighbors(state):
    """
    Generate all neighbor states by swapping two rows' columns.
    This creates all possible single-swap moves from the current state.
    Returns a list of (neighbor_state, conflicts_count) tuples.
    """
    neighbors = []
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            # Create a copy and swap two positions
            neighbor = state[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            conflicts = count_conflicts(neighbor)
            neighbors.append((neighbor, conflicts))
    return neighbors


def hill_climbing_solve(initial_state, max_iterations=10000):
    """
    Hill-Climbing Search for a conflict-free Eight-Queen solution.

    initial_state   : a full permutation of 0..7 representing the
                      starting board configuration.
    max_iterations  : maximum number of steps to prevent infinite loops.

    Returns (solution, nodes_expanded, steps_taken, local_optimum):
        solution       - the final state (could be a solution or local optimum)
        nodes_expanded - number of states evaluated (neighbors generated)
        steps_taken    - number of moves made from initial state
        local_optimum  - True if terminated at a local optimum (no improving move)
    """
    current_state = initial_state[:]
    current_conflicts = count_conflicts(current_state)
    steps_taken = 0
    nodes_expanded = 0

    while steps_taken < max_iterations:
        if current_conflicts == 0:
            return current_state, nodes_expanded, steps_taken, False

        # Generate all neighbors
        neighbors = generate_neighbors(current_state)
        nodes_expanded += len(neighbors)

        # Find the best neighbor (minimum conflicts)
        best_neighbor = None
        best_conflicts = current_conflicts

        for neighbor, conflicts in neighbors:
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_neighbor = neighbor

        # If no improving move, we're at a local optimum
        if best_neighbor is None or best_conflicts >= current_conflicts:
            return current_state, nodes_expanded, steps_taken, True

        # Move to the best neighbor
        current_state = best_neighbor
        current_conflicts = best_conflicts
        steps_taken += 1

    return current_state, nodes_expanded, steps_taken, True


def conflicting_rows(perm):
    """Set of row indices that take part in at least one diagonal conflict."""
    bad = set()
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(perm[i] - perm[j]) == abs(i - j):
                bad.add(i)
                bad.add(j)
    return bad


def print_board(state, flagged_rows=None):
    """Chess-style board rendering with conflict highlighting."""
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
    print("HILL-CLIMBING SEARCH - EIGHT-QUEEN PROBLEM")
    print("=" * 60)

    results = []

    for set_id, dataset in DATA_SETS.items():
        initial_conflicts = count_conflicts(dataset)
        bad_rows = conflicting_rows(dataset)

        print(f"\n{'-' * 60}")
        print(f"SET {set_id}")
        print(f"{'-' * 60}")
        print(f"\nInitial state (dataset)       : {dataset}")
        print(f"Initial conflicts             : {initial_conflicts} (queens involved marked *Q*)")
        print_board(dataset, flagged_rows=bad_rows)

        start_time = time.perf_counter()
        solution, nodes_expanded, steps_taken, local_optimum = hill_climbing_solve(
            dataset)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        final_conflicts = count_conflicts(solution)

        print(f"\nHill-Climbing result:")
        print(f"  Final state                 : {solution}")
        print(f"  Final conflicts             : {final_conflicts}")
        print(f"  Nodes expanded (evaluated)  : {nodes_expanded}")
        print(f"  Steps taken                 : {steps_taken}")
        print(f"  Time taken                  : {elapsed_ms:.4f} ms")
        print(f"  Status                      : {'SOLUTION FOUND' if final_conflicts == 0 else 'LOCAL OPTIMUM'}")

        if final_conflicts > 0:
            bad_rows_final = conflicting_rows(solution)
            print("  (Queens in conflict marked *Q*)")
            print_board(solution, flagged_rows=bad_rows_final)
        else:
            print("  (All queens safe!)")
            print_board(solution)

        results.append((set_id, initial_conflicts, final_conflicts,
                       nodes_expanded, steps_taken, elapsed_ms))

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"{'Set':<5}{'Init Conf':<12}{'Final Conf':<12}{'Nodes Eval':<14}{'Steps':<10}{'Time (ms)':<12}{'Status':<15}")
    for set_id, init_conf, final_conf, nodes, steps, elapsed in results:
        status = "SOLVED" if final_conf == 0 else "LOCAL OPT"
        print(
            f"{set_id:<5}{init_conf:<12}{final_conf:<12}{nodes:<14}{steps:<10}{elapsed:<12.4f}{status:<15}")


if __name__ == "__main__":
    main()
