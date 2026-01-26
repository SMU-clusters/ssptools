==========
Algorithms
==========

This package provides a number of classes designed to represent the evolution
of a given stellar mass function over time in an extremely rapid fashion. This
is accomplished not by directly sampling and evolving a system of stars, but
by solving simpler semi-analytical relations to derive the rate of change of
various quantities over time.

These algorithms are described in a simple fashion below. You may also see the
relevant papers noted in ":doc:`citations`" for more.

As this class is meant to be used in populating mass-models, the actual
implementation of these algorithms in the code is done on mass bins of stars
and remnants, defined by a total mass and number :math:`M_j`, :math:`N_j` and
thus :math:`m_j=M_j/N_j`.
However, most of these prescriptions could also be adapted to a continuous
mass function (see :class:`ssptools.StellarEvMassLoss` for an example).


Firstly, for convenience, we first introduce the helpful integral:

.. math::
    P_k(\alpha_j,\ m_{j,l},\ m_{j,u}) = \int_{m_{j,l}}^{m_{j,u}} m^{\alpha_j + k - 1}\ \mathrm{d}m


Initial Mass Function
---------------------

An initial mass function (IMF; :class:`ssptools.masses.PowerLawIMF`) is defined
by a broken power-law distribution function, composed of :math:`n` components
defined between break masses:

.. math::

    \xi (m) = \begin{cases}
        A_1\ m^{-\alpha_1}, & m_{b,0} < m \leq m_{b,1} \\
        &\vdots \\
        A_{n}\ m^{-\alpha_N}, & m_{b,n-1} < m \leq m_{b,n} \\
    \end{cases}

where the :math:`\alpha_i` exponents are the mass function (log-)slopes,
:math:`A_{i}` are the normalization constants,
:math:`m_{b,i}` are the break masses defining each component domain,
and :math:`\xi(m) \Delta m` is the number of stars with masses within the
interval :math:`m + \Delta m`.
The normalization constants are defined such that the total number of stars
is :math:`N_{tot}`:

.. math::
    A_i^{-1} = N_{tot}\ \sum_{j=1}^{i} P_1(a_j) \prod_{k=j+1}^{i} m_{k}^{a_k-a_{k-1}}

The IMF determines the initial distribution of stars within a given population,
can be evaluated to compute the total number or mass of stars of a given mass,
and sets the initial conditions of the mass function evolution algorithms.


Mass Function Evolution
-----------------------

The evolution of a mass function over time is modelled as two distinct systems
of differential equations (stellar evolution and stellar escapes),
which together define the overall rate of change of:

1) The number of stars :math:`\dot{N}_{s,j}`
2) The stellar mass function slopes :math:`\dot{\alpha}_{s,j}`
3) The number of remnants :math:`\dot{N}_{r,j}`
4) The mass of remnants :math:`\dot{M}_{r,j}`

These four quantities described everything necessary to model the evolution of
the mass function, as the mass of stars can also be computed as
:math:`M_{s,j} = {N}_{s,j}\ P_2(\alpha_{s,j}) / P_1(\alpha_{s,j})`.


Stellar Evolution
^^^^^^^^^^^^^^^^^

Stellar evolution mass loss refers to the loss of mass of stars over the
lifetime of a star from processes such as stellar winds, and supernovae.
In our prescriptions, this means a loss of mass and number of stars, and a
corresponding gain of mass and number of remnants.

This evolution is modelled by, at a given time, determining the number of stars
at the current "turn-off" phase of their lifetime, and computing the overall
loss of mass they experienced, as they are turned into remnants.


First, the lifetime of main-sequence stars are approximated by a function of
their initial mass:

.. math::

    t_{ms} = a_0 \exp(a_1 m^{a_2})

where the :math:`a_i` coefficients are interpolated from a fit to the Dartmouth
Stellar Evolution Program models (Dotter et al. 2007, 2008).
This equation can then be inverted and differentiated to find the rate of
change of the turn-off mass:

.. math::

    \frac{\mathrm{d}m_{to}}{\mathrm{d}t} = \frac{1}{a_1 a_2} \frac{1}{t}
        \left(\frac{\log(t/a_0)}{a_1}\right)^{1/a_2 - 1}


Thus the rate of change of stars, at a given turn off mass :math:`m_{to}` is
given by:

.. math::

    \dot{N}_s(m_{to}) = - \left.\frac{\mathrm{d}N}{\mathrm{d}m}\right|_{m_{to}}
        \left|{\frac{\mathrm{d}m_{to}}{\mathrm{d}t}}\right|

where the amount of stars at a given mass (:math:`\mathrm{d}N / \mathrm{d}m`,
evaluated at :math:`m_{to}`) is given by the current mass function slopes
(as defined by :math:`\alpha_j` and :math:`N_{s,j}`).


The corresponding change in the number and mass of remnants are then given by:

.. math::
    \dot{N}_{r} (m_r) = -\dot{N}_s(m_{to})\ f_{ret}

.. math::
    \dot{M}_r (m_r) = -m_r\ \dot{N}_s(m_{to})\ f_{ret} = m_r\ \dot{N}_r(m_r)

where :math:`m_r` is the mass of the individual remnants, and :math:`f_{ret}`
is the *retention fraction* of the remnants. This fraction is assumed to be
100% for White Dwarfs (WD). The initial retention fraction of Neutron Stars (NS)
after supernovae natal kicks is, by default, 10%. The retention fraction of BHs
is discussed in more detail below.

The type of remnant which will form from a given (initial) stellar mass, and its
final mass :math:`m_r`, is defined based on grids of stellar evolution models
and their initial-final mass relations (IFMR).

By default, the maximum initial mass which will form a WD, and
the WD IFMR, are interpolated, based on metallicity, from the MIST 2018
isochrones (Choi et al. 2016; Dotter 2016).
The Black Hole (BH) IFMR, as well as the minimum initial mass required to form
a BH, is by default interpolated directly from a grid of updated stellar
evolution library (SSE) models (Banerjee et al. 2020), using the "rapid"
supernovae scheme from Fryer et al. (2012).
Other IFMR models and supernovae schemes are also available.
All stars between the bounding WD and BH precursor masses are assumed to form
NS with a mass of :math:`1.4\ M_\odot`.


Black Hole Natal Kicks
""""""""""""""""""""""

The supernovae natal kicks experienced by BHs, and the resulting rate of
immediately ejected BHs from a system (defining :math:`f_{ret}`) is a
complicated function of uncertain supernova physics, and dependent on both
the properties of the BHs and host system.

By default, if BH natal kicks are applied, the "Maxwellian" method is used.
First, the assumption is made that the kick velocity is drawn from a Maxwellian
distribution with a given dispersion. By default this is
:math:`265\ \mathrm{km}\, \mathrm{s}^{-1}`, as has been found for NS
(Hobbs et al. 2005), but see also Disberg & Mandel (2025).
This velocity is then scaled down linearly by the "fallback fraction"
:math:`f_b`, the fraction of the precursor stellar envelope which falls back
on to the BH after the initial supernova explosion.
This fraction is, by default, interpolated from the same grid of SSE models
used to define the BH IFMR, and is a function of the individual BH mass,
though other fallback prescriptions are also available.

The fraction of BHs retained in each mass bin (:math:`f_{ret,j}`) is then
found by integrating the Maxwellian kick velocity distribution from 0 to the
escape velocity of the system, which must be given explicitly.


Other analytical natal kick retention fraction prescriptions are also available.
In order to still match the general mass-dependence of the physically-defined
kicks (i.e. a preferential loss of low-mass BHs), the retention fraction can
be represented more flexibly by a sigmoid function, such as:

.. math::
    f_{ret}(m) = \frac{1}{2} \left(
        \tanh\left(\mathrm{u_k}\ (m - m_k)\right) + 1
    \right)

where :math:`u_k` and :math:`m_k` are the slope and characteristic mass of the
curve.

For convenience, users may also supply a total kick fraction parameter
:math:`f_k` which is defined as the total fraction of initially formed BH mass
which is ejected from the cluster due to natal kicks, such that:

.. math::
        f_{k} = \sum_j \frac{M_{\mathrm{BH},j}}{M_{\mathrm{BH,tot}}}\
                        \left(1 - f_{ret}(m_{\mathrm{BH},j})\right)

in which case a root-finding algorithm will be used to determine the
:math:`m_k` which results in this :math:`f_k`.





Stellar Escapes
^^^^^^^^^^^^^^^

Next the loss of objects which escape from the system directly is modelled.
This includes both the loss of stars over the tidal boundary of a system, and
the dynamical ejection of BHs from the core.

The amount of stars lost to tides is a more complicated function of the
structure, dynamics and energy balance of both the system and a host galaxy, and
are out of scope of this package. Instead, the losses must be defined directly
by an input *escape rate* :code:`esc_rate`, which defines either
:math:`\dot{N}_{esc}` or :math:`\dot{M}_{esc}`, and may be either a constant
rate, or a function of time (in which case :math:`B` simply becomes
:math:`B(t)` below).

Next, it is assumed that the *mass function* (:math:`f`) changes due to the
tidal losses as:

.. math::
    \dot{f}(m) = -B\ f(m)\ h(m)

where :math:`B` is a normalization constant and :math:`h(m)` is a function
which can account for the preferential loss of stars of a given mass.
To model systems like star clusters which undergo mass segregation, and after
some time preferentially lose low-mass stars,
following Balbinot & Gieles (2018):

.. math::

    h(m) = \begin{cases}
        1 - \left(\frac{m}{m_d}\right)^{1/2} &\ ,\ m < m_d \\
        0  & \ ,\ m > m_d
    \end{cases}

where :math:`m_d` is the depletion mass, before which there is no biased
mass loss. This is set to :math:`1.2\ M_\odot` by default.

To determine the normalization constant, and the rate of change of the mass
function slopes :math:`\dot{\alpha}_{s,j}`, based on the input total escape
rate, the sum of escapes of both stars and remnants must be computed.

The integral over the mass function :math:`f(m)h(m)` (by number), in each
stellar mass bin :math:`j` is given by:

.. math::
    I_{s,j} = N_j \left(1 - m_d^{-1/2} \frac{P_{3/2}}{P_1}\right)

and by mass:

.. math::
    J_{s,j} = M_j \left(1 - m_d^{-1/2} \frac{P_{5/2}}{P_2}\right)


While for the remnants, :math:`P_{3/2}/P_1` cannot be computed directly, and
is instead approximated as
:math:`P_{3/2} / P_1 = \langle m_r^{1/2}\rangle \approx m_r^{1/2}`
and thus:

.. math::
    I_{r,j} = N_j \left(1 - m_d^{-1/2} m^{1/2}\right)

and similarly for the integral over mass:

.. math::
    J_{r,j} = M_j \left(1 - m_d^{-1/2} m^{1/2}\right)

With this, the normalization constant can be computed as either:

.. math::

    B = \frac{\dot{N}_{esc}}{\sum\limits{I_s + I_r}}

if the given total escape rate is defined as :math:`\dot{N}_{esc}`, or:

.. math::

    B = \frac{\dot{M}_{esc}}{\sum\limits{J_s + J_r}}


if it is defined as :math:`\dot{M}_{esc}`.


And finally, the mass and number derivatives are given by:

.. math::
    \begin{align*}
        \dot{N}_{s,j} &= B\ I_{s,j} \\
        \dot{N}_{r,j} &= B\ I_{r,j} \\
        \dot{M}_{r,j} &= B\ J_{r,j} \\
    \end{align*}

and the mass function slopes in each bin (defined by the bounds
:math:`m_{j1},\ m_{j2}`) are given by:

.. math::
    \dot{\alpha}_{j,s} = \frac{B}{\ln{(m_{j2}/m_{j1})}}\ \left[\left(\frac{m_{j1}}{m_d}\right)^{1/2} - \left(\frac{m_{j2}}{m_d}\right)^{1/2}\right] \\


BHs are also ejected over time from the core of most systems like star
clusters due to dynamical interactions with one another
(see e.g. Breen & Heggie 2013).
Here this process is simulated separately from the differential
equations defined above, through the direct removal of BHs, beginning with
the heaviest mass bins (with larger gravitational interaction cross-sections)
through to the lightest, until the combination of mass in BHs lost through both
the natal kicks and these dynamical ejections leads to a retained mass in BHs
which equals an inputted overall BH mass retention parameter (either
:code:`BH_ret_dyn` in :class:`ssptools.EvolvedMF` or :code:`f_BH` in
:class:`ssptools.EvolvedMFWithBH`).

----


These differential equations representing the stellar evolution and escapes
are solved, together, to determine the
final evolved mass function at a given age, and the distributions of stars
and remnants of different masses.
