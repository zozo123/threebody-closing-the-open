from threebody_atlas.topology import canonical_conjugacy_word, cyclic_reduce, free_reduce


def test_free_reduction():
    assert free_reduce("aAbB") == ""
    assert free_reduce("abBA") == ""


def test_cyclic_reduction():
    assert cyclic_reduce("abA") == "b"


def test_conjugacy_is_rotation_invariant():
    assert canonical_conjugacy_word("abAB") == canonical_conjugacy_word("bABa")


def test_inverse_not_silently_identified():
    assert canonical_conjugacy_word("aab") != canonical_conjugacy_word("BAA")
