# SSPTools

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/pjs902/ssptools/blob/master/LICENSE)
[![Tests](https://github.com/SMU-clusters/ssptools/actions/workflows/tests.yml/badge.svg)](https://github.com/SMU-clusters/ssptools/actions/workflows/tests.yml)
### Simple Stellar Population Tools

Provides access to the `EvolvedMF` class (and similar subclasses), which
evolves an arbitrary N-component power law initial mass function (IMF) to a
binned present-day mass function (PDMF) at any given set of ages, and computes
the numbers and masses of stars and remnants in each mass bin.

To be used for populating mass models and other such simulations based on
an IMF and various other initial population parameters.

Can optionally account for all of the evolution of stars off the main-sequence,
the loss of low-mass stars to a host tidal field, the ejection of
black holes due to both dynamical ejections and natal kicks.


### Note
This is a fork of [SSPTools](https://github.com/balbinot/ssptools) which has been updated to use an N-component
mass function, updated remnant initial-final mass relations and implements
further black hole retention calculations, among other changes.


### Quickstart

SSPTools can be installed from [PyPI](https://pypi.org/project/astro-ssptools/) using:

```
pip install astro-ssptools
```

An evolved mass function can be computed using the `EvolvedMF` class:

```python
import ssptools

m_break, a_slopes, nbins = [0.08, 0.5, 150.], [+1.3, -2.3], [5, 30]
pdmf = ssptools.EvolvedMF.from_powerlaw(m_break, a_slopes, nbins,
                                        FeH=-1.5, tout=13000, esc_rate=0, N0=1e6)
```

Alternatively, an IMF class can be instantiated and used directly:

```python
imf = ssptools.masses.PowerLawIMF(m_break, a_slopes, N0=1e6)
pdmf = ssptools.EvolvedMF(imf, nbins, FeH=-1.5, tout=[0, 1000, 13000], esc_rate=0)
```

See the documentation of each class for more details on all possible parameters.

The final element of `tout` (in Myr) defines the age to which the mass function
is evolved to, where the masses and numbers of stars and remnants (together)
can then be accessed easily:

```python

pdmf.N  # Total number of stars in each bin
pdmf.M  # Total mass of stars in each bin
pdmf.m  # Mean mass of stars in each bin

pdmf.M[pdmf.types == 'MS']  # Star bins only
pdmf.M[pdmf.types == 'BH']  # Black hole bins only

```

All other outputted times can also be seen in the underlying attributes, which
generally have shape (`len(tout)`, `sum(nbins)`). Note that these arrays will
also contain bins with basically no objects in them (i.e. N<0.1), which are not
valid and are typically filtered out in the output arrays above.

```python
pdmf.Ns[0]  # Initial star bin amounts
pdmf.Ns[-1]  # Final star bin amounts

pdmf.Mr  # Named tuple with fields ('WD', 'NS', 'BH')
pdmf.Mr.BH[-1]  # Final BH bin masses
```

```python
import matplotlib.pyplot as plt

for i in range(pdmf.nout):
    mes = pdmf.massbins.turned_off_bins(pdmf.tout[i])  # mass bins at t_i
    plt.step(pdmf.ms[i], pdmf.Ns[i] / (mes.upper - mes.lower),
             ls='-', label=f"Stars @ t={pdmf.tout[i]} Myr")
#
    mebh = pdmf.massbins.bins.BH  # BH mass bins
    plt.step(pdmf.mr.BH[i], pdmf.Nr.BH[i] / (mebh.upper - mebh.lower),
             ls='--', c=str(0.8 * 1 - (i / pdmf.nout)),
             label=f" BHs  @ t={pdmf.tout[i]} Myr")

plt.ylabel(r"$\frac{\mathrm{d}N}{\mathrm{d}m}$", rotation=0)
plt.xlabel(r"$m\ [M_\odot]$")
plt.yscale('log'); plt.xscale('log')
plt.ylim(bottom=1)

plt.legend(); plt.show()
```

![evolve_mf_example1](docs/source/_static/evolve_mf_example1.png)
