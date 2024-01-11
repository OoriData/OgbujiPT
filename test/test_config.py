# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# test/test_config.py
from ogbujipt import config

def test_attr_dict():
    d = config.attr_dict({'a': 1, 'b': 2})
    assert d.__class__.__bases__ == (dict,)
    assert d.a == 1
    assert d.b == 2
    assert d['a'] == 1
    assert d['b'] == 2


if __name__ == '__main__':
    raise SystemExit("Attention! Run with pytest")
