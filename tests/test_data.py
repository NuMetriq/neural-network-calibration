from calrep.data import _deterministic_split_indices


def test_split_no_overlap_and_complete():
    n = 1000
    train_idx, val_idx = _deterministic_split_indices(n=n, val_fraction=0.2, seed=123)
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(train_idx) + len(val_idx) == n


def test_split_deterministic():
    n = 1000
    a_train, a_val = _deterministic_split_indices(n=n, val_fraction=0.2, seed=123)
    b_train, b_val = _deterministic_split_indices(n=n, val_fraction=0.2, seed=123)
    assert (a_train == b_train).all()
    assert (a_val == b_val).all()
