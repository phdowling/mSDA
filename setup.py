#!/usr/bin/env python

from distutils.core import setup

setup(name='linear_msda',
      version='0.9',
      description='(Linear) Marginalized Stacked Denoising Autoencoder',
      author='Philipp Dowling',
      author_email='philipp@dowling.io',
      url='https://github.com/phdowling/mSDA',
      py_modules=['linear_msda', 'mda_layer'],
     )