def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in original order."""
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_sorted: list[tuple[int, float]] = []
    running_max = 0.0
    for rank, (original_idx, p_value) in enumerate(indexed):
        adjusted = min((m - rank) * p_value, 1.0)
        running_max = max(running_max, adjusted)
        adjusted_sorted.append((original_idx, running_max))

    adjusted_by_original = [1.0] * m
    for original_idx, adjusted in adjusted_sorted:
        adjusted_by_original[original_idx] = adjusted
    return adjusted_by_original
