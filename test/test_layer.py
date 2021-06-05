def test_import_layer():
    from pinkfrog.layer import Layer
    assert Layer is not None


def test_import_add():
    from pinkfrog.layer import Add
    add_one = Add(1)
    assert 2 == add_one(1)
