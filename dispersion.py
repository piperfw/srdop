#!/usr/bin/env python

from mpmath import polylog
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, h, e, hbar
import sys, os

if not os.path.exists('figures'):
    os.makedirs('figures')
#print(plt.rcParams.keys())
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 16,
    "lines.linewidth": 2.5,
})
lambdap = 660 # nm
k0 = 2*np.pi/lambdap * 1e09 # resonance wavevector, SI units
w0 = k0 * c # resonance frequency, SI units
w0 = (hbar/e) * k0 * c # in eV
print(f'wp = {w0:.4g}')
w02 = (hbar/e) * c *  2 * np.pi /665 * 1e09
print(f'wp2 = {w02:.4g}')
#E0 = (h * c/e) / (lambdap * 1e-09) # same as w0 in eV
#print(w0, E0)
eta = -2 # longitudinal mode
#eta = 1 # transverse mode
a = 40 # nanoparticle radius nm
print('k0a = {:.2g}'.format(k0*a*1e-09))
d = 2*a + 1.0 # nanoparticle spacing, centre-centre (1nm gap)
Omega = (w0/2) * (a/d)**3 # long-range dipolar coupling strength between LSPs
Q0 = 150
Nk = 2*Q0+1
L = round(Nk * d, 4)
qs = np.array([2*np.pi*n/L for n in range(-Q0,Q0+1)])
fs = np.zeros_like(qs)

for i,q in enumerate(qs):
    exp = np.exp(1j*q*d)
    fs[i] = eta * float(polylog(3, exp).real+polylog(3,np.conj(exp)).real)
omegas_0 = w0 * np.sqrt(1 + 2 * (Omega/w0) * fs)
omegas_nn = w0 * np.sqrt(1 + 4 * eta * (Omega/w0) * np.cos(qs * d))
#cutoff = (hbar/e) / a # ultraviolet cutoff, a in nm, no factor of c
cutoff = 1/a # cutoff in inverse nm
print('Cutoff', cutoff)
# SI_units
EV_TO_HZ = e/hbar
omegas_0_HZ_NM = EV_TO_HZ * omegas_0 / 1e09
#print(np.max(omegas_0_HZ_NM))
cutoff_HZ_NM = c/a #* 1e09
qs_M = qs # * 1e09 # SI units
#print(omegas_0_HZ_NM/(c*qs_M))
#sys.exit()

#deltas = (eta/2) * (w0**2/omegas_0) * (qs**2 * a**3/d) * np.heaviside(cutoff - np.abs(qs), 0.0)\
# * (np.log(cutoff/qs)
#    + 0.5*(1+np.sign(eta)*(omegas_0_HZ_NM/(c*qs_M))**2)\
#    * np.log(np.abs(c**2*qs_M**2-omegas_0_HZ_NM**2)/(cutoff_HZ_NM**2-omegas_0_HZ_NM**2))
#    )
# #print(deltas)
def omega_full(q):
    if np.isclose(q, 0):
        return omega_full((qs[1]-qs[0])/10)
    if q < 0.0:
        return omega_full(-q)
    exp = np.exp(1j*q*d)
    f = eta * float(polylog(3, exp).real+polylog(3,np.conj(exp)).real)
    omega_0 =  w0 * np.sqrt(1 + 2 * (Omega/w0) * f)
    omega_0_HZ_NM = EV_TO_HZ * omega_0 / 1e09
    prefactor = (eta/2) * (w0**2/omega_0) * (q**2 * a**3/d) * np.heaviside(cutoff - np.abs(q), 0.0)
    term1 = np.log(cutoff/q)
    term2 = 0.5*(1 + np.sign(eta)*(omega_0_HZ_NM/(c*q))**2)\
    * np.log(np.abs(c**2*q**2-omega_0_HZ_NM**2)/(cutoff_HZ_NM**2-omega_0_HZ_NM**2))
    return omega_0 + prefactor * (term1 + term2)

omegas_full = [omega_full(q) for q in qs]

fig, ax = plt.subplots()
#ax.plot(qs, omegas_0, label='quasistatic')
#ax.plot(qs, omegas_nn, label='quasistatic n.n.')
##ax.plot(qs, omegas_0+deltas, label='2nd order')
ax.plot(qs, omegas_full, label='2nd order')
#ax.legend()
ax.set_xlabel(r'$q\ \text{\rm{(nm}}^{-1}\text{\rm{)}}$ ')
ax.set_ylabel(r'$\omega_q\  \text{\rm{(eV)}}$')
fig.savefig('figures/chain-dispersion.png', dpi=450, bbox_inches='tight')


