"""Submission interface. Implement your own selection method here."""


"""MP2 - E0 Baseline: Always Choose Candidate 0."""


def predict(bank, arrays, checkpoint):
    """Return one candidate index per question."""
    return [0 for _ in bank]