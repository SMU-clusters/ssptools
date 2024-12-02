# Initial-Final Mass Relation Grids

This folder contains various grids of black hole IFMR prescriptions (initial
and final black hole masses between ZAMS 15-150 Msun), to be used in fast
IFMR functions.

The updated versions of both COSMIC and SSE are very similar in most aspects,
with the most noticeable difference in the below prescriptions being the
choice made in the treatment of PP SNe.
Custom IFMRs with different parameters can also be created in SSPtools, using
COSMIC (but will be slightly slower). See `ifmr` for more details.


## `uSSE_rapid/`

Updated version of SSE (Banerjee et al. 2020;
https://github.com/sambaranb/updated-BSE) IFMR, assuming the Fryer+2012 rapid
supernovae prescription. Also included are the fallback fractions `fbac`, to
be used in computing natal kick distributions. The models assume `zsun=0.02`.
Models are computed using the following SSE parameters:

| Parameter | Value |
| ----------| ----- |
| ifflag    | 0     |
| wdflag    | 1     |
| nsflag    | 3     |
| bhflag    | 2     |
| sigma     | 265   |
| idum      | 111   |
| psflag    | 0     |
| kmech     | 1     |
| ecflag    | 1     |
| pts1      | 0.001 |
| pts2      | 0.02  |
| pts3      | 0.02  |
| neta      | 0.5   |
| bwind     | 0.0   |
| hewind    | 1.0   |
| mxns      | 2.5   |


## `uSSE_delayed/`

Updated version of SSE (Banerjee et al. 2020;
https://github.com/sambaranb/updated-BSE) IFMR, assuming the Fryer+2012 delayed
supernovae prescription. Also included are the fallback fractions `fbac`, to
be used in computing natal kick distributions. The models assume `zsun=0.02`.
Models are computed using the following SSE parameters:

| Parameter | Value |
| ----------| ----- |
| ifflag    | 0     |
| wdflag    | 1     |
| nsflag    | 4     |
| bhflag    | 2     |
| sigma     | 265   |
| idum      | 111   |
| psflag    | 0     |
| kmech     | 1     |
| ecflag    | 1     |
| pts1      | 0.001 |
| pts2      | 0.02  |
| pts3      | 0.02  |
| neta      | 0.5   |
| bwind     | 0.0   |
| hewind    | 1.0   |
| mxns      | 2.5   |


## `COSMIC_rapid/`

COSMIC (Breivik et al. 2020; version 3.4.17) IFMR, assuming the Fryer+2012
rapid supernovae prescription.
Models are computed using the following BSE parameters:

| Parameter   | Value |
| ----------- | ----- |
| pts1        | 0.001 |
| pts2        | 0.02  |
| pts3        | 0.02  |
| zsun        | 0.02  |
| windflag    | 3     |
| eddlimflag  | 0     |
| pisn        | 45    |
| remnantflag | 3     |

All other parameters are left as defaults (see `ifmr._DEFAULT_BSEDICT`).


## `COSMIC_delayed/`


COSMIC (Breivik et al. 2020; version 3.4.17) IFMR, assuming the Fryer+2012
delayed supernovae prescription.
Models are computed using the following BSE parameters:

| Parameter   | Value |
| ----------- | ----- |
| pts1        | 0.001 |
| pts2        | 0.02  |
| pts3        | 0.02  |
| zsun        | 0.02  |
| windflag    | 3     |
| eddlimflag  | 0     |
| pisn        | 45    |
| remnantflag | 4     |

All other parameters are left as defaults (see `ifmr._DEFAULT_BSEDICT`).

