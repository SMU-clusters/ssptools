#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from matplotlib import pyplot as p
from Kroupa import Kroupa

## Slopes and bin edges
a = np.r_[np.zeros(6) + 1.3, np.zeros(6) + 2.3]
mlim = np.r_[
    np.logspace(np.log10(0.08), np.log10(0.5), 7),
    np.logspace(np.log10(0.5), np.log10(3), 7)[1:],
]

for A, MMIN, MMAX in zip(a, mlim[:-1], mlim[1:]):
    print("Piece {:.2f}, Limits [{:.2f}, {:.2f}]".format(A, MMIN, MMAX))

K = Kroupa(a=a, mlim=mlim)
mlim = K._mlim

# Read model
## MIST uses base natural for luminosity and temperature...
mfile = "/home/balbinot/ssptools/ssptools/data/models/MIST_iso_5b0533fab00b3.iso.cmd"
eep, m, logT, logL, g, r, i = np.loadtxt(
    mfile, usecols=(0, 2, 4, 6, 10, 11, 12), unpack=True
)

## Select only MS, SGB and RGB
v = eep < 601
eep = eep[v]
m = m[v]
logT = logT[v]
logL = logL[v]
g = g[v]
r = r[v]
i = i[v]

# Interpolate interesting quantities
kind = "cubic"
berr = False
ilogL = interp1d(m, logL, bounds_error=berr, kind=kind)
ilogT = interp1d(m, logT, bounds_error=berr, kind=kind)
ig = interp1d(m, g, bounds_error=berr, kind=kind)
ir = interp1d(m, r, bounds_error=berr, kind=kind)
ii = interp1d(m, i, bounds_error=berr, kind=kind)

# Convenience function for integration of L(m)
def evalf(m):
    return K.eval(m) * (np.exp(ilogL(m)))


## Define mass bins from minimum and maximum mass available in the model
## CAUTION: last mass bin must be the maximum mass in the isochrone. Otherwise interpolation
##          will return rubish
mm = np.linspace(np.min(m), np.max(m), 10)

# Test range and interpolation
# p.plot(mm, ilogL(mm))
# for M in mm:
#    p.axvline(M)
# p.show()

for mmin, mmax in zip(mm[0:-1], mm[1:]):
    I, I2 = K.integral(mmin, mmax)

# Get mass and total luminosity in bins
p.figure(figsize=(8, 8))
mc = 0.5 * (mm[1:] + mm[:-1])
I, I2, L = [], [], []
for mmin, mmax in zip(mm[0:-1], mm[1:]):
    res = K.integral(mmin, mmax)
    Lres = quad(evalf, mmin, mmax)
    I.append(res[0])
    I2.append(res[1])
    L.append(Lres[0])
    # print(mmin, mmax, res[0], res[1], Lres[0])
I = np.array(I)
I2 = np.array(I2)
L = np.array(L)

# print(ilogT(mc), ilogL(mc))
p.plot(logT, logL, "k.")
p.scatter(ilogT(I2 / I), ilogL(I2 / I), c=I2 / L, s=90, zorder=99)
p.xlim(p.xlim()[::-1])
cb = p.colorbar()
cb.set_label("M/L")

for M in mm:
    p.axhline(ilogL(M), c="k", ls="--")

for M in mlim:
    p.axhline(ilogL(M), c="r", ls="-", lw=2)

p.xlabel("logTeff")
p.ylabel("logLum")

## Final test, sample MF, use masses in interpolated isocrhone and check
## against analytic prediction
p.figure()
N = np.int(1e5)
mass = K.sample(N)  # sample some masses
for mmin, mmax in zip(mm[0:-1], mm[1:]):
    j = (mass < mmax) * (mass >= mmin)
    Lum = np.exp(ilogL(mass[j]))
    totLum = np.sum(Lum)
    totMass = np.sum(mass[j])
    mMass = np.mean(mass[j])
    p.plot(mMass, totMass / totLum, "ko")

p.plot(mMass, totMass / totLum, "ko", label="Simulated")
p.plot(I2 / I, I2 / L, "o", label="Analytic")
p.legend()

## Artificial CMD with mock errors
p.figure()
sg = ig(mass)
sr = ir(mass)


def err(x):
    ee = 0.002 + np.exp((x - 12) / 1.3)
    ee[ee > 0.5] = 0.5
    return ee


sg += err(sg) * np.random.randn(len(sg))
sr += err(sr) * np.random.randn(len(sr))

p.plot(sg - sr, sg, "k.", ms=1, alpha=0.8)
p.ylim(14, -2)
p.xlim(-1.2, 3)

p.show()
