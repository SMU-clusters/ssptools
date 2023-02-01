# SSPTools

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/pjs902/ssptools/blob/master/LICENSE)
[![Tests](https://github.com/SMU-clusters/ssptools/actions/workflows/tests.yml/badge.svg)](https://github.com/SMU-clusters/ssptools/actions/workflows/tests.yml)
### Simple Stellar Population Tools

Provides access to the `EvolvedMF` class, which evolves an arbitrary 3-component
power law initial mass function (IMF) to a binned present-day mass function
(PDMF) at any given set of ages, and computes the numbers and masses of stars
and remnants in each mass bin.

To be used for populating mass models and other such simulations based on
an IMF and various other initial population parameters.

Can optionally account for all of the evolution of stars off the main-sequence,
the loss of low-mass stars to a host tidal field, the ejection of
black holes due to both dynamical ejections and natal kicks.


### Note
This is a fork of [SSPTools](https://github.com/balbinot/ssptools) which has been updated to use a 3-component
mass function, updated remnant initial-final mass relations and implements
further black hole retention calculations, among other changes.

