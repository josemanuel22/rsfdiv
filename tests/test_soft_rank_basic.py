# tests/test_soft_rank_basic.py
import torch
from ranks.ranks_torch import sliced_rank_fdiv

def test_zero_divergence_identical():
    """D_f(X||X) should be ~0 for identical distributions."""
    torch.manual_seed(0)
    X = torch.randn(10000, 2)
    D = sliced_rank_fdiv(X, X, L=10, K=64, f_name="js")
    assert D.item() < 1e-2, f"Divergence too large: {D.item()}"

def test_positive_definite():
    """D_f(X||Y) ≥ 0 for all valid f-divergences."""
    torch.manual_seed(1)
    X = torch.randn(512, 2)
    Y = torch.randn(512, 2) + 1.0
    D = sliced_rank_fdiv(X, Y, L=16, K=32, f_name="kl")
    assert D.item() >= 0.0, "KL divergence became negative!"

def test_increasing_with_shift():
    """D_f increases as mean shift grows."""
    torch.manual_seed(2)
    base = torch.randn(512, 2)
    D1 = sliced_rank_fdiv(base, base + 0.2, L=16, K=32, f_name="js")
    D2 = sliced_rank_fdiv(base, base + 1.0, L=16, K=32, f_name="js")
    assert D2 > D1, f"Expected D(shift=1) > D(shift=0.2), got {D2} ≤ {D1}"
    
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "--disable-warnings"]))
