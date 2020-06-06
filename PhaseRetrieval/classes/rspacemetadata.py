#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for object-domain metadata

"""

from __future__ import division

import pandas as pd


class RSpaceMetadata(pd.Series):
    """
    Class for object-domain metadata.
    """
    def __init__(self, filename=None):
        """
        Initializes object-domain metadata class.

        Parameters
        ----------
        filename : str, optional
            Path used to load metadata.
            If None, no metadata will be loaded.
            Default is None.
        """
        super().__init__()

        if filename is not None:
            self['Name'] = filename
            self.read_from_csv(filename)

    def __repr__(self):
        return "Metadata of an object-domain image"