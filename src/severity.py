"""Turn a lesion count into a severity level.

The thresholds follow the same idea as the Hayashi grading scale that
dermatologists use: severity is judged from how many lesions are present.
Documented in docs/PLAN.md and in the training notebook (notebooks/Acno.ipynb).
"""

SEVERITY_LEVELS = ["clear", "mild", "moderate", "severe"]

# upper bound of lesion count for each level; the last level has no bound
THRESHOLDS = [
    (1, "clear"),
    (10, "mild"),
    (25, "moderate"),
]


def severity_from_count(count: int) -> str:
    for bound, level in THRESHOLDS:
        if count <= bound:
            return level
    return "severe"


def needs_dermatologist(severity: str, acne_types: list[str]) -> bool:
    """Severe findings and cystic acne always get a see-a-dermatologist note."""
    return severity == "severe" or "Cyst" in acne_types
