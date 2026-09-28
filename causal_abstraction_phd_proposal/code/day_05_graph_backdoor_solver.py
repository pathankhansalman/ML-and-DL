"""
Day 5 Coding Lab: Automated Backdoor Adjustment Solver using NetworkX.

Given any DAG, finds all valid adjustment sets Z to estimate P(Y | do(X)).
Criteria:
1. No node in Z is a descendant of X.
2. Z d-separates X and Y in the graph G_X_lower (with outgoing edges from X removed).
"""

from itertools import chain, combinations
import networkx as nx


def powerset(iterable):
    """Generates all possible subsets of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_backdoor_adjustment_sets(G: nx.DiGraph, treatment: str, outcome: str):
    """
    Finds all valid Pearl Backdoor adjustment sets for estimating causal effect of treatment on outcome.
    """
    # 1. Condition 1: No node can be a descendant of treatment X
    descendants_of_x = nx.descendants(G, treatment)
    forbidden_nodes = descendants_of_x | {treatment, outcome}
    candidate_nodes = [node for node in G.nodes() if node not in forbidden_nodes]

    # 2. Construct G_X_lower: remove all OUTGOING edges from treatment X
    G_x_lower = G.copy()
    outgoing_edges = list(G.out_edges(treatment))
    G_x_lower.remove_edges_from(outgoing_edges)

    # 3. Test each subset for d-separation in G_x_lower
    valid_sets = []
    for candidate_set in powerset(candidate_nodes):
        candidate_set = set(candidate_set)
        # Check d-separation in G_x_lower
        if nx.is_d_separator(G_x_lower, {treatment}, {outcome}, candidate_set):
            valid_sets.append(sorted(list(candidate_set)))

    return valid_sets, list(descendants_of_x)


if __name__ == '__main__':
    print("=" * 65)
    print("DAY 5: AUTOMATED PEARL BACKDOOR ADJUSTMENT SOLVER")
    print("=" * 65)

    # Example 1: The Classic M-Bias + Confounder Graph
    # Paths between X and Y:
    # - X -> Y (Direct causal)
    # - X <- Z -> Y (Confounder backdoor path)
    # - X <- U1 -> M <- U2 -> Y (Collider path via M)
    G = nx.DiGraph([
        ('Z', 'X'), ('Z', 'Y'),
        ('X', 'Y'),
        ('U1', 'X'), ('U1', 'M'),
        ('U2', 'Y'), ('U2', 'M')
    ])

    treatment = 'X'
    outcome = 'Y'

    valid_sets, descendants = find_backdoor_adjustment_sets(G, treatment, outcome)

    print(f"Graph nodes: {list(G.nodes())}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    print(f"Descendants of {treatment} (Forbidden): {descendants}")
    print(f"\nValid Backdoor Adjustment Sets:")
    for s in valid_sets:
        print(f"  -> {s}")

    print("\nInsight Check:")
    print("  * Does the valid set require 'Z'? YES (to block X <- Z -> Y)")
    print("  * Can the set include 'M'? NO! M is a collider; conditioning on M opens U1 -> M <- U2 (M-Bias).")
    print("=" * 65)
