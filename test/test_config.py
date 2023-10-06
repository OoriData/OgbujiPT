from ogbujipt import config

def test_attr_dict():
    d = config.attr_dict({'a': 1, 'b': 2})
    assert d.__class__.__bases__ == (dict,)
    assert d.a == 1
    assert d.b == 2
    assert d['a'] == 1
    assert d['b'] == 2
