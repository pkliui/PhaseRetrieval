#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for Fourier-domain metadata

"""

from __future__ import division

import pandas as pd
import os


class KSpaceMetadata(pd.Series):
    """
    Class for Fourier-domain metadata.
    """
    def __init__(self, filename=None):
        """
        Initializes Fourier-domain metadata class.

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
        return "Metadata of a Fourier-domain image"

    def read_from_csv(self, filename):
        """
        Reads Fourier-domain metadata from a csv file.

        Parameters
        ----------
        filename : str
            The path used to read the object-domain metadata.
        """
        if os.path.exists(filename):
            f = open(filename)
            st = f.readlines()
            f.close()
            if len(st) > 0:
                data = pd.read_csv(filename, sep='\t', index_col=0, header=None).transpose()
                if len(data.columns == 1):
                    data = data.iloc[0]
                else:
                    data = data.iloc[0].T.squeeze()

                for c in data.index:
                    try:
                        self[c] = float(data[c])
                    except ValueError:
                        self[c] = data[c]