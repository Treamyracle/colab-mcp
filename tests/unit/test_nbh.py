"""
Unit tests for Notebook Hash (nbh) Derivation Algorithm.
Requirement: TECH_SPEC §4.2, PROJECT.md Interface Contract 1.
Authoritative source: COLAB_MCP_TECHNICAL_SPECIFICATION.md lines 674-695.
Algorithm:
1. Generates UUID v4 (36 chars: e.g. 'c9a646d3-9c61-4cc9-bc77-b22e11e0563e')
2. Replaces hyphens '-' with underscores '_'
3. Right-pads with '.' up to exactly 44 characters (8 dots)
"""

import re
import uuid
import pytest

# Feature 19 (NBH derivation) is defined in PROJECT.md for M2 runtime management
# Support progressive testability across M1 and M2
try:
    import colab_mcp.utils.nbh as nbh_mod
    generate_nbh = getattr(nbh_mod, "generate_nbh", getattr(nbh_mod, "generate_notebook_hash", None))
except ImportError:
    nbh_mod = None
    generate_nbh = None


@pytest.fixture(autouse=True)
def require_nbh_implementation():
    if generate_nbh is None:
        pytest.skip("colab_mcp.utils.nbh is scheduled for M2 (Feature 19)")


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

def test_generate_nbh_returns_string_with_exact_length_44():
    """Verify generate_nbh() produces a string of exactly 44 characters."""
    nbh = generate_nbh()
    assert isinstance(nbh, str)
    assert len(nbh) == 44, f"Expected 44 characters, got {len(nbh)}: {nbh}"


def test_generate_nbh_contains_only_valid_charset():
    """Verify generate_nbh() contains only alphanumeric characters, underscores, and dots."""
    nbh = generate_nbh()
    valid_pattern = re.compile(r"^[a-zA-Z0-9_\.]+$")
    assert valid_pattern.match(nbh) is not None, f"NBH contains invalid characters: {nbh}"


def test_generate_nbh_deterministic_with_known_uuid():
    """
    Verify generate_nbh() with a fixed UUID derives the exact expected golden string.
    Authoritative vector:
      UUID: c9a646d3-9c61-4cc9-bc77-b22e11e0563e
      Expected: c9a646d3_9c61_4cc9_bc77_b22e11e0563e........ (8 dots)
    """
    fixed_uuid = uuid.UUID("c9a646d3-9c61-4cc9-bc77-b22e11e0563e")
    expected = "c9a646d3_9c61_4cc9_bc77_b22e11e0563e........"
    derived = generate_nbh(fixed_uuid)

    assert derived == expected
    assert len(derived) == 44


def test_generate_nbh_exact_eight_trailing_dots():
    """Verify that a standard 36-character UUID is right-padded with exactly 8 dots."""
    u = uuid.uuid4()
    nbh = generate_nbh(u)
    assert nbh.endswith("........")
    assert not nbh.endswith(".........")  # Must not have 9 dots


def test_generate_nbh_random_uniqueness():
    """Verify repeated invocations with session_id=None produce distinct hashes."""
    hashes = {generate_nbh() for _ in range(50)}
    assert len(hashes) == 50, "Generated duplicate NBHs from random UUIDs"


# ============================================================================
# Tier 2: Boundary, Edge Cases, & Corner Conditions (>= 5 tests)
# ============================================================================

def test_generate_nbh_zero_uuid_boundary():
    """Verify edge case with nil UUID (all zeros: 00000000-0000-0000-0000-000000000000)."""
    zero_uuid = uuid.UUID(int=0)
    expected = "00000000_0000_0000_0000_000000000000........"
    derived = generate_nbh(zero_uuid)

    assert derived == expected
    assert len(derived) == 44


def test_generate_nbh_max_uuid_boundary():
    """Verify edge case with max integer UUID (all f's: ffffffff-ffff-ffff-ffff-ffffffffffff)."""
    max_uuid = uuid.UUID(int=(1 << 128) - 1)
    expected = "ffffffff_ffff_ffff_ffff_ffffffffffff........"
    derived = generate_nbh(max_uuid)

    assert derived == expected
    assert len(derived) == 44


def test_generate_nbh_replaces_all_hyphens():
    """Verify that no hyphens exist in the generated NBH string."""
    for _ in range(20):
        nbh = generate_nbh()
        assert "-" not in nbh, f"Found hyphen in NBH: {nbh}"
        assert nbh.count("_") == 4, f"Standard UUID v4 has 4 hyphens, expected 4 underscores: {nbh}"


def test_generate_nbh_web_safe_compliance():
    """Verify NBH contains no URI-unsafe or RFC 3986 reserved characters."""
    unsafe_chars = {"/", "+", "=", "%", "?", "&", "#", " ", "\t", "\n"}
    for _ in range(20):
        nbh = generate_nbh()
        assert not any(c in nbh for c in unsafe_chars), f"Found URI-unsafe char in {nbh}"


def test_generate_nbh_scale_batch_consistency():
    """Verify generating 1000 NBHs retains 100% length-44 validity and uniqueness."""
    batch_size = 1000
    generated = [generate_nbh() for _ in range(batch_size)]

    assert len(set(generated)) == batch_size
    assert all(len(h) == 44 for h in generated)
    assert all(h.endswith("........") for h in generated)


def test_generate_nbh_invalid_type_handling():
    """Verify passing an invalid type (e.g. non-UUID object) raises a TypeError or handles appropriately."""
    with pytest.raises((TypeError, AttributeError)):
        generate_nbh(12345)  # type: ignore
