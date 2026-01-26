=========
Citations
=========

This library is a fork based on the original algorithms of
`Balbinot and Gieles (2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.2479B>`_.

The package was then updated and described in
`Dickson et al. (2023)`_, `Dickson et al. (2024)`_ (version <1.0.0),
and Dickson et al. (in prep) (version >2.0.0).


The stellar evolution lifetimes are interpolated from the
Dartmouth Stellar Evolution Program models
(`Dotter et al. 2007`_, `2008`_).
The White Dwarf IFMR is interpolated from the MIST 2018 isochrones
(`Choi et al. 2016`_, `Dotter 2016`_).
The default "Maxwellian" IFMR is interpolated from an updated grid of SSE models
(`Banerjee et al. 2020`_), with the "rapid" and "delayed" supernovae schemes
from `Fryer et al. (2012)`_.
The "COSMIC" IFMRs use the COSMIC-popsynth library (`Breivik et al. 2020`_).

If you find this package useful in your research, please consider citing the
relevant papers.


.. _Dickson et al. (2023): https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.5320D
.. _Dickson et al. (2024): https://ui.adsabs.harvard.edu/abs/2024MNRAS.529..331D

.. _Dotter et al. 2007: https://ui.adsabs.harvard.edu/abs/2007AJ....134..376D
.. _2008: https://ui.adsabs.harvard.edu/abs/2008ApJS..178...89D
.. _Choi et al. 2016: https://ui.adsabs.harvard.edu/abs/2016ApJ...823..102C
.. _Dotter 2016: https://ui.adsabs.harvard.edu/abs/2016ApJS..222....8D
.. _Banerjee et al. 2020: https://ui.adsabs.harvard.edu/abs/2020A&A...639A..41B
.. _Fryer et al. (2012): https://ui.adsabs.harvard.edu/abs/2012ApJ...749...91F
.. _Breivik et al. 2020: https://ui.adsabs.harvard.edu/abs/2020ApJ...898...71B
