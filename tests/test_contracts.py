import pytest
from pydantic import ValidationError

from pr_review.contracts import Diff, DiffFile, Finding


def test_finding_round_trip():
    finding = Finding(
        path="app/auth.py",
        line=42,
        side="RIGHT",
        category="security",
        severity="high",
        confidence=0.91,
        rationale="Hardcoded credential in source.",
        fingerprint="abc123",
    )
    assert finding.suggested_patch is None
    assert Finding.model_validate(finding.model_dump()) == finding


def test_finding_rejects_invalid_category():
    with pytest.raises(ValidationError):
        Finding(
            path="app/auth.py",
            line=1,
            side="RIGHT",
            category="performance",  # not one of the four lenses
            severity="low",
            confidence=0.1,
            rationale="x",
            fingerprint="x",
        )


def test_diff_holds_files():
    diff = Diff(
        head_sha="deadbeef",
        files=[DiffFile(path="app/auth.py", status="modified", patch="@@ -1 +1 @@")],
    )
    assert diff.files[0].path == "app/auth.py"
