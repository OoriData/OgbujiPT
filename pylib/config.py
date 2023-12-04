# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# ogbujipt.config

# Given the nature of a config module, we don't use __all__
# & avoid top-level imports where possible, and at the end of the module
# del any symbols we don't want exposed
# pylint: disable=wrong-import-position

'''
Configuration & globally-relevant values
'''

# Really just a bogus name for cases when OpenAPI is being emulated
# OpenAI API requires the model be specified, but many compaitble APIs
# have a model predetermined by the host
HOST_DEFAULT_MODEL = HOST_DEFAULT = 'HOST-DEFAULT'
OPENAI_KEY_DUMMY = 'OPENAI_DUMMY'


class attr_dict(dict):
    # XXX: Should unknown attr access return None rather than raise?
    # If so, can just do: __getattr__ = dict.get
    # __getattr__ = dict.__getitem__
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            # Substitute with more normally expected exception
            raise AttributeError(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
