#!/usr/bin/env python
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import logging, pickle, os, sys
from opt_einsum import contract
from time import time
from pprint import pprint, pformat
from copy import copy
from scipy.signal import butter, filtfilt, lfilter, freqz, argrelextrema
from scipy.optimize import curve_fit
from scipy import constants
from scipy.fft import fft, ifft, fftshift, ifftshift # recommended over numpy.fft
from scipy.integrate import RK45, DOP853, solve_ivp
#SOLVER = RK45 # Runge-Kutta 4th order
SOLVER = DOP853
from mpmath import polylog
try:
    import pretty_traceback
    pretty_traceback.install()
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

try:
    from gmanp import pBasis, Pauli, Boson
except ModuleNotFoundError as e:
    logger.critical(e)
    #logger.critical('gmanp.py not found in current working directory')
    sys.exit(1)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 11,
    #"lines.linewidth": 2.5,
})


class HTC:
    COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    EV_TO_FS = (constants.hbar/constants.e)*1e15 # convert time in electronvolts to time in fs
    DEFAULT_DIRS = {'data':'./data', 'figures':'./figures'} # output directories
    DEFAULT_PARAMS = {
            'Q0': 61, # N_k = 2*Q0+1 nanoparticles (123)
            'NE': 4, # Number of emitters per gap
            'w': 1, # Gap width, nm (Emitter spacing Delta_r = 2a+w = 81nm)
            'a': 40, # Nanoparticle radius, nm (Chain length L = N_k * Delta_r = 10.0 nm)
            'omega_p': 1.88, # Plasmon resonance, eV
            'omega_0': 1.86, # Dye resonance, eV
            't': 0.2, # Hopping strength, eV
            'g': 0.095, # Individual light-matter coupling, eV (collective gSqrt(NE))
            'kappa': 1e-01, # photon loss
            'dephase': 0.0, # Emitter pure dephasing
            'pump_strength': 1e-01, # Emitter pump magnitude
            'pump_width': 500, # Emitter pump spot width, nm (approx. 6 nanoparticles)
            'decay': 1e-01, # Emitter non-resonant decay
            'Nnu': 2, # Number of vibrational levels for each emitter
            'S': 0.01, # Huang-Rhys parameter
            'omega_nu': 0.19, # Vibrational mode resonance, eV
            'T':0.026, # k_B T in eV (.026=302K)
            'gam_nu': 1e-03, # vibrational damping rate
            'gam_as': 0.0, # Antistokes decay
            'as_coherent': False, # True to turn on g(a . sigma^+ . b^- + H.C.) terms
            'dt': 0.5, # interval at which solution is sampled. Does not affect accuracy of solution 
            'model': 'plasmonic', # or 'tight-binding', 'two-node' - sets the dispersion ('two-node' is tight-binding but only coupling points at k= \pm \pi/(2 Delta r) [N.B. assumes all but those entires are zero in the initial state]
            }

    @classmethod
    def default_params(self):
        return copy(self.DEFAULT_PARAMS)

    @classmethod
    def butter_lowpass_filter(cls, data, cutoff, fs, order=5, axis=-1):
        data = np.array(data)
        b, a = butter(order, cutoff, fs=fs, btype='low')
        padlen = 3 * max(len(a), len(b)) # default filtfilt padlength
        if padlen >= data.shape[-1] - 1:#but must be less than data.shape[-1]-1
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
            padlen = max(0, data.shape[-1]-2) 
        return filtfilt(b, a, data, axis=axis, padlen=padlen)

    def __init__(self, params=None):
        for name, dp in self.DEFAULT_DIRS.items():
            if not os.path.exists(dp):
                os.makedirs(dp)
        self.params = self.parse_params(params)
        self.gp = pBasis(self.params['Nnu'], verbose=False, tests=False)
        self.add_useful_params(self.params) # requires self.gp. Otherwise must be run before all other functions
        self.boson = Boson(self.Nnu)
        self.rates = self.get_rates(self.params)
        self.Ks, self.ks = self.get_modes()
        self.make_state_dic()
        self.ns = np.arange(self.Nk)
        self.rs = self.ns * self.params['delta_r'] * 1e-3 # micrometers
        self.wrapit(self.make_coeffs, f'Constructing EoM coefficients...', timeit=False)
        self.create_initial_state()
        self.labels = self.get_labels()

    def parse_params(self, params):
        if params is None:
            return self.default_params()
        default_params = self.default_params()
        parsed_params = {}
        used_defaults = {}
        unused = {}
        for name, val in default_params.items():
            if name not in params:
                used_defaults[name] = val
                parsed_params[name] = val
                continue
            try:
                parsed_params[name] = type(val)(params[name])
            except TypeError or ValueError:
                logger.warning(f'Param {name} should be {type(val)}, using default {val}')
                used_defaults[name] = val
                unused[name] = params[name]
                parsed_params[name] = val
        for name, val in params.items():
            if name not in default_params:
                unused[name] = val
        if len(used_defaults) > 0:
            logger.info('The following parameters were not specified hence assigned default values:')
            logger.info(pformat(used_defaults, sort_dicts=False))
        if len(unused) > 0:
            logger.warning('The following parameters were not recognised and have not been used:')
            logger.warning(pformat(unused, sort_dicts=False))
        return parsed_params

    def add_useful_params(self, params):
        params['nb'] = self.nb(params['T'])
        self.Q0, self.N0, self.N1 = params['Q0'], self.gp.indices_nums[0], self.gp.indices_nums[1]
        params['Nk'] = 2*self.Q0+1 # total number of modes / nanoparticles
        params['delta_r'] = round(2 * params['a'] + params['w'], 5) # grid spacing in micrometers
        params['L'] = round(params['Nk'] * params['delta_r'], 5) # chain length
        self.c = constants.c # speed of light in material
        self.K_to_eV = (constants.h * self.c) / (constants.e * params['L'] * 1e-06) # factor 1/e for Joules to eV
        self.wp = self.params['omega_p']
        self.rescale_int = 1 # rescale variables according to prescription in thesis
        self.NE, self.Nk, self.Nnu = params['NE'], params['Nk'], params['Nnu']
        params['Nm'] = self.NE * self.Nk # Total number of emitters
        self.Nm = params['Nm']
        params['gSqrtNE'] = params['g'] * np.sqrt(self.NE)
        self.off_diag_indices_Nk = np.where(~np.eye(self.Nk, dtype=bool))
        self.diag_indices_Nk = np.diag_indices(self.Nk)

    def get_rates(self, params):
        rates = {}
        rates['pump_strength'] = params['pump_strength']
        rates['decay'] = params['decay']
        rates['dephase'] = params['dephase']
        rates['gam_nu'] = params['gam_nu']
        rates['gam_up'] = rates['gam_nu'] * params['nb']
        rates['gam_down'] = rates['gam_nu'] * (params['nb'] + 1)
        rates['gam_delta'] = rates['gam_up'] - rates['gam_down']
        params['gam_up'] = rates['gam_up']
        params['gam_down'] = rates['gam_down']
        rates['gam_as'] = params['gam_as']
        return rates

    def get_modes(self):
        Qs = np.array([Q for Q in self.get_Q_range()])
        return Qs, (2*np.pi/self.params['L']) * Qs # integers, inverse microns

    def get_Q_range(self, sign=1, offset=0, reverse=False, start=None):
        if start == None:
            shift = 0
        else:
            shift = start + self.Q0 # e.g. start=0 makes [0, 2Q0] rather than [-Q0,Q0], for indexing arrays
        min_Q = max(-self.Q0, -self.Q0 - sign*offset)
        max_Q = min(self.Q0, self.Q0 - sign*offset)
        if reverse:
            return range(max_Q+shift, min_Q-1+shift, -1)
        # Q in [-Q0, Q0] such that offset + sign*Q is also in this interval
        return range(min_Q+shift, max_Q+1+shift)

    def make_state_dic(self):
        Nk, Nnu = self.Nk, self.Nnu
        slices = {}
        names = ['ada', 'l', 'al', 'll']
        self.state_dic = {
            'ada': {'shape': (Nk, Nk)}, 
            'l': {'shape': (2*Nnu**2-1, Nk)}, 
            'al': {'shape': (Nnu**2, Nk, Nk)}, 
            'll': {'shape': (Nnu**2, Nnu**2, Nk, Nk)}, 
        }
        self.state_split_list, self.state_reshape_list = [], []
        tot = 0
        for name in self.state_dic:
            shape = self.state_dic[name]['shape']
            self.state_reshape_list.append(shape)
            num = np.prod(shape)
            self.state_dic[name]['num'] = num
            self.state_dic[name]['slice'] = slice(tot, tot+num)
            self.state_split_list.append(tot+num)
            tot += num
        #self.state_split_list.pop() 
        self.added_state_length = 1# EXTRA entry appending to state to indicate if has been rescaled or not
        self.state_length = np.sum([self.state_dic[name]['num'] for name in self.state_dic]) + self.added_state_length
        self.state_dic_mf = {
                'a': {'shape': Nk},
                'lp': {'shape': (Nnu**2,Nk)},
                'l0': {'shape': (2*Nnu**2-1, Nk)},
                }
        self.state_split_list_mf, self.state_reshape_list_mf = [], []
        tot = 0
        for name in self.state_dic_mf:
            shape = self.state_dic_mf[name]['shape']
            self.state_reshape_list_mf.append(shape)
            num = np.prod(shape)
            self.state_dic_mf[name]['num'] = num
            self.state_dic_mf[name]['slice'] = slice(tot, tot+num)
            self.state_split_list_mf.append(tot+num)
            tot += num
        self.state_length_mf = np.sum([self.state_dic_mf[name]['num'] for name in self.state_dic_mf]) 
        correct_state_length_mf = Nk + (Nnu**2*Nk) + (2*Nnu**2-1)*Nk
        correct_state_length = Nk**2 + (2*Nnu**2-1)*Nk + Nnu**2*Nk**2 + Nnu**4*Nk**2 + self.added_state_length
        self.state_split_list_mf.pop()
        assert self.state_length == correct_state_length,\
                f'state length is {self.state_length} but should be {correct_state_length}'
        assert self.state_length_mf == correct_state_length_mf,\
                f'mean-field state length is {self.state_length_mf} but should be {correct_state_length_mf}'

    def wrapit(self, meth, msg='', timeit=True):
        if msg:
            logger.info(msg)
        t0 = time()
        ret = meth()
        if timeit:
            logger.info('...done ({:.2f}s)'.format(time()-t0))
        return ret

    def make_coeffs(self):
        # dictionary for equation coefficients and constants used in their construction
        coeffs, consts = {}, {}
        gp = self.gp
        params = self.params
        rates = self.rates
        Nm, Nk, Nnu, Q0 = self.Nm, self.Nk, self.Nnu, self.Q0
        Hvib = Boson(Nnu)
        b, bd, bn, bi = Hvib.b, Hvib.bd, Hvib.n, Hvib.i
        sm, sp, sz, si = Pauli.m, Pauli.p, Pauli.z, Pauli.i
        A = 0.5*params['omega_0']*np.kron(sz, bi) +\
                params['omega_nu']*np.kron(si, bn) +\
                params['omega_nu']*np.sqrt(params['S'])*np.kron(sz, b+bd) + 0j
        A += 0.25 * (-1j * rates['gam_delta']) * np.kron(si, (bd @ bd - b @ b))
        # 2024-06-22 : BUG, replace gSqrtN with gSqrtNE * (Sqrt(Nm)/Sqrt(NE)=Sqrt(Nk)) NOT JUST gSqrtNE
        # Correction: add sNk factors here (also definition of zeta_k - but not currently used?)
        # sNk = np.sqrt(Nk) B = sNk * params['gSqrtNE'] * np.kron(sp, bi+bd)
        # Better: just use gSqrtNm
        # 2024-06-26: NOT a bug
        if params['as_coherent']:
            B =  params['gSqrtNE'] * np.kron(sp, bi+bd)
        else:
            B =  params['gSqrtNE'] * np.kron(sp, bi)
        A0_base, _discard = gp.get_coefficients(A, sgn=0, eye=True) # discard part proportional to identity
        consts['A0_n'] = np.outer(A0_base, np.ones(Nk)) # currently no spatial dependence 
        consts['Bp'] = gp.get_coefficients(B, sgn=1) # N.B. gets i_+ coefficients i.e. traces against lambda_{i_-}
        C1_base = gp.get_coefficients(np.kron(sz, bi), sgn=0)
        C2_base = gp.get_coefficients(np.kron(si, bd), sgn=0)
        C3_base = gp.get_coefficients(np.kron(si, b), sgn=0)
        Dp_base = gp.get_coefficients(np.kron(sp, bi), sgn=1)
        Dm_base = gp.get_coefficients(np.kron(sm, bi), sgn=-1)
        DAS_base = gp.get_coefficients(np.kron(sm, bd), sgn=-1)
        consts['gam0_n'] = np.array([np.outer(base, np.sqrt(Xs)) for Xs, base in
                                     zip([self.dephase(self.ns), rates['gam_up'] * np.ones(Nk),
                                          rates['gam_down'] * np.ones(Nk)],[C1_base, C2_base, C3_base])])
        consts['gamp_n'] = np.outer(Dp_base, np.sqrt(self.pump(self.ns)))
        consts['gamm_n'] = np.outer(Dm_base, np.sqrt(self.decay(self.ns)))
        consts['gamas_n'] = np.outer(DAS_base, np.sqrt(rates['gam_as'] * np.ones(Nk)))
        #
        consts['gam00_n'] = contract('arn,apn->rpn', consts['gam0_n'].conj(), consts['gam0_n'])
        consts['gampp_n'] = contract('in,jn->ijn', consts['gamp_n'].conj(), consts['gamp_n'])
        consts['gammm_n'] = contract('in,jn->ijn', consts['gamm_n'].conj(), consts['gamm_n'])
        consts['gamas_n'] = contract('in,jn->ijn', consts['gamas_n'].conj(), consts['gamas_n'])
        f000 = gp.f_tensor((0,0,0))
        f011 = gp.f_tensor((0,1,1))
        z000 = gp.z_tensor((0,0,0))
        z011 = gp.z_tensor((0,1,1))
        z011_swap = np.swapaxes(z011, 1, 2)
        assert np.allclose(z011, np.conj(z011_swap))
        zm011 = gp.z_tensor((0,-1,-1))
        consts['xi_n'] = 2 * contract('ipj,pn->ijn', f000, consts['A0_n']) \
                + 2 * contract('irq,rpn,qpj->ijn', f000, consts['gam00_n'], z000).imag \
                + 2 * contract('ijn,aip,bpj->abn', consts['gampp_n'], f011.conj(), zm011).imag \
                + 2 * contract('ijn,aip,bpj->abn', consts['gammm_n'], f011, z011).imag \
                + 2 * contract('ijn,aip,bpj->abn', consts['gamas_n'], f011, z011).imag
        consts['phi0_n'] = (2/params['Nnu']) * contract('ajq,jqn->an', f000, consts['gam00_n'].imag) \
                + (2/params['Nnu']) * contract('ijn,aij->an', consts['gampp_n'], f011.conj()).imag \
                + (2/params['Nnu']) * contract('ijn,aij->an', consts['gammm_n'], f011).imag \
                + (2/params['Nnu']) * contract('ijn,aij->an', consts['gamas_n'], f011).imag 
                  # Note gamm00 index order reversed below (conjugate)
        consts['xip_n'] = - 2 * contract('aij,an->ijn', f011, consts['A0_n']) \
                + 1j * contract('aip,abn,bpj->ijn', f011, consts['gam00_n'], zm011.conj()) \
                - 1j * contract('aip,ban,bpj->ijn', f011, consts['gam00_n'], z011) \
                + 1j * contract('aip,qpn,aqj->ijn', f011, consts['gammm_n'], zm011.conj()) \
                + 1j * contract('aip,qpn,aqj->ijn', f011, consts['gamas_n'], zm011.conj()) \
                - 1j * contract('aip,pqn,aqj->ijn', f011, consts['gampp_n'], z011)
        consts['xim_n'] = - 2 * contract('aij,an->ijn', f011.conj(), consts['A0_n']) \
                + 1j * contract('aip,abn,bpj->ijn', f011.conj(), consts['gam00_n'], z011.conj()) \
                - 1j * contract('aip,ban,bpj->ijn', f011.conj(), consts['gam00_n'], zm011) \
                + 1j * contract('aip,qpn,aqj->ijn', f011.conj(), consts['gampp_n'], z011.conj()) \
                - 1j * contract('aip,pqn,aqj->ijn', f011.conj(), consts['gammm_n'], zm011) \
                - 1j * contract('aip,pqn,aqj->ijn', f011.conj(), consts['gamas_n'], zm011)
        shifted_Ks = np.fft.ifftshift(self.Ks) # ALL COMPUTATIONS DONE WITH numpy order of modes
        #rolled_Ks = np.roll(self.Ks, -self.Q0) # equivalent to shifted_Ks
        consts['kappa'] = self.kappa(shifted_Ks)
        consts['omega'] = self.omega(shifted_Ks)
        # CIJ means Jth coeff. of Ith eqn.
        def expikr(K, n):
            return np.exp(2j * np.pi / Nk * K * n)
        #coeffs['expi_kn'] = np.zeros((Nk, Nk), dtype=complex)
        coeffs['expi_kn'] = np.fromfunction(np.vectorize(expikr), (Nk, Nk)) # because of periodicity, this is sufficient
        # EQ 1 , here p->k'
        coeffs['11_pk'] = np.zeros((Nk, Nk), dtype=complex) # But MUST construct this manually!
        for i, K in enumerate(shifted_Ks):
            for j, P in enumerate(shifted_Ks):
                # EDIT 2023-10-26: Fixed sign of omega term!
                coeffs['11_pk'][j,i] = (1j * self.omega_diff(P,K) - 0.5 * self.kappa_sum(P,K))[0]
        # 2024-04-02 elements now with n dependence: 21_00n, (22_0n,) 31_11kn, 41_11n, 42_11n
        coeffs['12_1'] = 1j * consts['Bp']
        coeffs['13_1'] = coeffs['12_1'].conj()
        # EQ 2
        coeffs['21_00n'] = 1 * consts['xi_n'] # (make a copy)
        coeffs['22_0n'] = 1 * consts['phi0_n']
        coeffs['23_01'] = 4 * contract('i,aij->aj', consts['Bp'], f011)
        # EQ 3
        coeffs['31_11kn'] = contract('ijn,k->ijkn', consts['xip_n'], np.ones(Nk)) \
                - contract('ij,n,k->ijkn', np.eye(gp.indices_nums[1]), np.ones(Nk),
                           1j * consts['omega'] + 0.5 * consts['kappa'])
        coeffs['32_1'] = coeffs['12_1'].conj()
        coeffs['33_1kn'] = 1j * contract('i,kn->ikn', consts['Bp'].conj(), coeffs['expi_kn']) / Nm
        coeffs['34_1kn0'] = - contract('jkn,aij->ikna', coeffs['33_1kn'], z011)
        coeffs['35_1kn'] = - coeffs['33_1kn'] / Nnu
        coeffs['36_01'] = 2 * contract('aij,j->ai', f011, consts['Bp'].conj())
        # EQ 4
        coeffs['41_11n'] = 1 * consts['xip_n']
        coeffs['42_11n'] = 1 * consts['xim_n']
        coeffs['43_01'] = coeffs['36_01'].conj() 
        coeffs['44_01'] = 1 * coeffs['36_01']
        if self.rescale_int==1:
            coeffs['23_01'] *= (1/Nm)
            coeffs['34_1kn0'] *= Nm
            coeffs['35_1kn'] *= Nm
        elif self.rescale_int==2:
            sNm = np.sqrt(Nm)
            coeffs['23_01'] *= (1/Nm) * (1/sNm)
            coeffs['34_1kn0'] *= Nm * sNm
            coeffs['35_1kn'] *= Nm * sNm
        # HOPFIELD coefficients (in shifted basis i.e. K=0,1,2,...,Q0,-Q0,-Q0+1,....-1
        #consts['zeta_k'] = 0.5 * np.sqrt( (params['omega_0'] - consts['omega'])**2 + 4 * params['gSqrtN']**2 )
        #coeffs['X_k'] = np.sqrt(0.5  + 0.5**2 * (params['omega_0'] - consts['omega'])/consts['zeta_k'])
        #coeffs['Y_k'] = np.sqrt(0.5  - 0.5**2 * (params['omega_0'] - consts['omega'])/consts['zeta_k'])
        #assert np.allclose(coeffs['X_k']**2+coeffs['Y_k']**2, 1.0), 'Hopfield coeffs. not normalised'
        consts['vsigma'] = gp.get_coefficients(np.kron(sp, bi), sgn=1, eye=False)
        assert np.allclose(consts['vsigma'].imag, 0.0)
        consts['vvsigma'] = self.Nnu/2
        assert np.allclose(consts['vvsigma'], contract('i,i', consts['vsigma'], consts['vsigma']))
        assert np.allclose(contract('i,inm->nm', consts['vsigma'], gp.basis[gp.indices[1]]), np.kron(sp,bi))
        # Coefficients used to calculate dynamics
        ocoeffs = {}
        # 'pup_l' and 'pup_I' are C^0_{i_0} and D^0 in thesis
        ocoeffs['pup_l'], ocoeffs['pup_I'] = \
                self.gp.get_coefficients(np.kron(Pauli.p1, bi), sgn=0, eye=True)
        assert np.allclose(contract('a,anm->nm', ocoeffs['pup_l'], gp.basis[gp.indices[0]]),
                           np.kron(Pauli.p1, bi)-ocoeffs['pup_I']*np.eye(2*self.Nnu))
        assert np.allclose(ocoeffs['pup_I'], 0.5)
        ocoeffs['sp_l'] = consts['vsigma']
        ocoeffs['rn'] = params['delta_r'] * self.ns
        ocoeffs['rn2'] = ocoeffs['rn']**2
        ocoeffs['msrn'] = (ocoeffs['rn'] - 0.5 * params['L'])**2
        v_ops = [np.kron(si, # si = identity i.e. no condition on electronic state
                         np.diag([1.0 if i==j else 0.0 for i in range(Nnu)])) for j in range(Nnu)]
        ev_ops = [np.kron(Pauli.p1, # vibrational populations conditioned on excited electronic state
                          np.diag([1.0 if i==j else 0.0 for i in range(Nnu)])) for j in range(Nnu)]
        ocoeffs['vpops'] = [self.gp.get_coefficients(v_op, sgn=0, eye=True) for v_op in v_ops]
        ocoeffs['evpops'] = [self.gp.get_coefficients(ev_op, sgn=0, eye=True) for ev_op in ev_ops]
        # assign to instance variables
        self.consts, self.coeffs, self.ocoeffs = consts, coeffs, ocoeffs
        # Construct masks used in two-node model
        if self.params['model'] == 'two-node':
            assert Q0%2==0, 'For two-node mode QO must be even'
            Qhalf = Q0//2
            QL, QR = Q0-Qhalf, Q0+Qhalf
            logger.info(f'Quarter mode at {QL} compared to N_k/4 = {Nk/4} has omega_k = '\
                    f'{self.omega(self.Ks[QL]):.2g}')
            mask_1d = np.ones(Nk, dtype=bool)
            self.MORE_MODES = False
            num_about = 0
            if num_about >= Qhalf:
                more_modes = self.Ks
                self.num_modes = self.Nk
            else:
                self.num_modes = 4*num_about+2
                more_modes = [QL + x for x in range(-num_about,num_about+1)] + [QR + x for x in range(-num_about,num_about+1)]
            if self.MORE_MODES:
                for Q in more_modes:
                    mask_1d[Q] = False
            else:
                num_about = 0
                mask_1d[QL], mask_1d[QR] = False, False
            mask_2d = np.ones((Nk, Nk), dtype=bool)
            if self.MORE_MODES:
                for Q in more_modes:
                    mask_2d[Q] = mask_1d
            else:
                mask_2d[QL] = mask_1d
                mask_2d[QR] = mask_1d
            self.mode_mask = {}
            self.mode_mask['ada'] = ifftshift(mask_2d)
            self.mode_mask['ada_args'] = np.where(self.mode_mask['ada'])
            #print(self.mode_mask['ada'].astype(int))
            #print(self.mode_mask['ada_args'])
            mask_kn = (np.zeros((Nk,Nk), dtype=bool) + mask_1d).T # only if Nnu=1
            mask_ikn = np.zeros(self.state_dic['al']['shape'], dtype=bool)
            mask_ikn[:] = mask_kn # at each i mask everything except rows K=QL, K=QR
            self.mode_mask['al'] = ifftshift(mask_ikn, axes=1)
            self.mode_mask['al_args'] = np.where(self.mode_mask['al'])
            self.QL, self.QR = QL, QR
        else:
            self.QL, self.QR, self.mode_mask = None, None, None
        # coefficients for mean-field equations
        M = {}
        M['iB_1'] = 1j * consts['Bp']
        M['11_k'] = -(1j * consts['omega'] + 0.5 * consts['kappa'])
        M['21_11n'] =  1 * consts['xip_n']
        M['22_10'] = 2 * contract('j,aij->ia', consts['Bp'], f011)
        M['31_00n'] = 1 * consts['xi_n']
        M['32_0n'] = 1 * consts['phi0_n']
        M['33_01'] = 4 * contract('i,aij->aj', consts['Bp'], f011) # DON'T copy coeffs['23_01'] due to rescaling
        self.coeffs_mf = M

    def omega_single(self, K):
        if K == 0:
            return self.omega_single(1/10)
        if K < 0:
            return self.omega_single(-K)
        k = 2*np.pi*K/self.params['L']
        a = self.params['a'] # nanoparticle radius
        d = self.params['delta_r'] # gap spacing
        c = self.c
        w0 = self.params['omega_p'] # plasmon resonance
        eta = -2 # longitudinal mode
        Omega = (w0/2) * (a/d)**3 # long-range dipolar coupling strength between LSPs
        cutoff = 1/a # momentum cutoff
        EV_TO_HZ = constants.e/constants.hbar
        exp = np.exp(1j*k*d)
        f = eta * float(polylog(3, exp).real+polylog(3,np.conj(exp)).real)
        omega_0 =  w0 * np.sqrt(1 + 2 * (Omega/w0) * f)
        omega_0_HZ_NM = EV_TO_HZ * omega_0 / 1e09
        cutoff_HZ_NM = c/a
        prefactor = (eta/2) * (w0**2/omega_0) * (k**2 * a**3/d) * np.heaviside(cutoff - np.abs(k), 0.0)
        term1 = np.log(cutoff/k)
        term2 = 0.5*(1 + np.sign(eta)*(omega_0_HZ_NM/(c*k))**2)\
        * np.log(np.abs(c**2*k**2-omega_0_HZ_NM**2)/(cutoff_HZ_NM**2-omega_0_HZ_NM**2))
        return omega_0 + prefactor * (term1 + term2)

    def omega_hop(self, K):
        kdr = K * (2*np.pi/self.Nk)
        return - 2 * self.params['t'] * np.cos(kdr)

    def omega(self, Ks):
        if self.params['model'] in ['tight-binding', 'two-node']:
            return self.omega_hop(Ks)
        Ks = np.atleast_1d(Ks)
        return np.array([self.omega_single(K) for K in Ks])

    def kappa(self, K):
        K = np.atleast_1d(K)
        return self.params['kappa'] * np.ones_like(K)

    def gaussian(self, r, _max, _width, _offset=0):
        r0 = self.Q0 * self.params['delta_r'] + _offset
        #return self.params['pump_strength'] * np.ones_like(n) # uniform
        return _max * np.exp(- 0.5 * ( (r-r0)/_width )**2 )

    def pump(self, n):
        r = self.params['delta_r'] * n
        #return self.params['pump_strength'] * np.ones_like(n)
        return self.gaussian(r, self.params['pump_strength'], self.params['pump_width'])

    def decay(self, n):
        n = np.atleast_1d(n)
        return self.params['decay'] * np.ones_like(n)

    def dephase(self, n):
        n = np.atleast_1d(n)
        return self.params['dephase'] * np.ones_like(n)

    def deltak(self, k=0):
        return np.array([1.0 if q==k else 0.0 for q in self.get_Q_range()])

    def cg(self, K):
        return self.c * self.K_factor * K / np.sqrt(1 + self.K_factor**2 * K**2)

    def velocity(self, t, D, alpha):
        return alpha * D * t**(alpha-1)

    def omega_diff(self, K, P):
        return self.omega(K) - self.omega(P)

    def kappa_sum(self, K, P):
        return self.kappa(K) + self.kappa(P)

    def partition(self, T):
        if T==0.0:
            return 1.0
        if T==np.inf:
            return self.params['Nnu']
        Nnu, omnu = self.params['Nnu'], self.params['omega_nu']
        return (1-np.exp(-Nnu * omnu / T))/(1-np.exp(-omnu/T))

    def thermal_rho_vib(self, T):
        if T==0.0:
            ps = [1.0 if n==0 else 0.0 for n in range(self.Nnu)]
            return np.diag(ps)
        if T==np.inf:
            return (1/self.Nnu) * np.eye(self.Nnu)
        Z = self.partition(T)
        ps = [np.exp(-n * self.params['omega_nu'] / T) for n in range(self.Nnu)]
        return (1/Z) * np.diag(ps)

    def nb(self, T):
        if T==0.0:
            return 0.0
        return 1/(np.exp(self.params['omega_nu']/self.params['T'])-1)

    def create_initial_state(self):
        self.initial_state = self.ground_state()

    def ground_state(self, mf=False):
        if mf:
            logger.info(f'Creating mean-field initial state (ground state) with small symmetry breaking')
            #pex = 2.2208628880360237e-07
            pex = 0.0
        else:
            logger.info(f'Creating initial state with 0 photons/excitons + thermal vibrational populations')
            pex = 0.0 # initial excited state population of emitters
        rho0_ele = np.diag([pex,1.0-pex])
        rho0_vib = self.thermal_rho_vib(self.params['T']) # molecular vibrational density matrix
        rho0 = np.kron(rho0_ele, rho0_vib)
        coeffs0, eye0 = self.gp.get_coefficients(rho0, sgn=0, eye=True, warn=False)
        l = [2 * coeffs0 for n in range(self.Nk)]
        l = np.real(l).T # i_0 index first, then ensemble index n
        self.all_eye0s = [eye0 for n in range(self.Nk)] # only needed if want to recreate density matrix on a site
        if mf:
            #coeffs1 = (-0.0004712602672110967) * self.gp.get_coefficients(Pauli.p, sgn=1)
            #coeffs1 = (-0.0004712602672110967) * self.gp.get_coefficients(Pauli.p, sgn=1)
            #lp = [2 * coeffs1 for n in range(self.Nk)]
            state = np.zeros(self.state_length_mf, dtype=complex)
            #state[:self.Nk] = 0.13338429247292602/np.sqrt(self.Nm)
            state[:self.Nk] = 0.1
            state[self.Nk * (1 + self.Nnu**2):] = l.flatten()
            #state[:self.Nk] = 0.1/np.sqrt(self.Nm) # symmetry-breaking
            #state[self.Nk:self.Nk*(1+self.Nnu**2)] = np.array(lp).T.flatten()
            #pprint(state)
            return state
        state = np.zeros(self.state_length, dtype=complex)
        state[self.state_dic['l']['slice']] = l.flatten()
        state[-1] = -1 # indicates state has NOT been rescaled
        return state

    def evolve_mf(self, tend=100.0, atol=1e-8, rtol=1e-6, y0=None):
        dt_fs = self.params['dt']
        t_fs = np.arange(0.0, tend+dt_fs/2, step=dt_fs)
        t = t_fs / self.EV_TO_FS
        dt = dt_fs / self.EV_TO_FS
        Nt = len(t)
        #y0 = self.ground_state(mf=True)
        if y0 is None:
            y0 = self.ground_state(mf=True)
        #if ak0s is not None:
        #    y0[:self.Nk] = ak0s
        #print(len(y0))
        #with open('init.pkl', 'rb') as fb:
        #    y0 = pickle.load(fb)
        ##pprint(y0)
        #print(len(y0))
        logger.info(f'Integrating mean-field EoMs to tend={tend} using solve_ivp')
        t0 = time()
        soln = solve_ivp(self.eoms_mf,
                         (t[0], t[-1]),
                         y0,
                         t_eval=t,
                         method='DOP853',
                         atol=atol,
                         rtol=rtol)
        ys = soln.y
        aKs = np.zeros((Nt, self.Nk), dtype=complex)
        nPs = np.zeros((Nt, self.Nk), dtype=float)
        nMs = np.zeros((Nt, self.Nk), dtype=float)
        logger.info(soln.message + ' (runtime {:.1f}s)'.format(time()-t0))
        for i, y in enumerate(ys.T):
            #print(len(y))
            a, lp, l0 = self.split_reshape_return_mf(y)
            a *= np.sqrt(self.Nm)
            aKs[i,:] = a
            ar = ifft(a, norm='ortho')
            nP =  ar.conj() * ar 
            self.check_real(i, nP, 'Photon number')
            nPs[i,:] = nP.real
            nM = self.NE * (contract('a,an->n', self.ocoeffs['pup_l'], l0) + self.ocoeffs['pup_I'])
            self.check_real(i, nM, 'Photon number')
            nMs[i,:] = nM.real
        self.dynamics_mf = {'ak':aKs, 'nP': nPs, 'nM': nMs}
        #with open('results.pkl', 'rb') as fb:
        #    other_ts, other_ys = pickle.load(fb)
        #aKs = np.zeros((Nt, self.Nk), dtype=complex)
        #nPs = np.zeros((Nt, self.Nk), dtype=float)
        #nMs = np.zeros((Nt, self.Nk), dtype=float)
        #logger.info(soln.message + ' (runtime {:.1f}s)'.format(time()-t0))
        #for i, y in enumerate(other_ys.T):
        #    a, lp, l0 = self.split_reshape_return_mf(y)
        #    a *= np.sqrt(self.Nm)
        #    aKs[i,:] = a
        #    ar = ifft(a, norm='ortho')
        #    nP =  ar.conj() * ar 
        #    self.check_real(i, nP, 'Photon number')
        #    nPs[i,:] = nP.real
        #    nM = self.NE * (contract('a,an->n', self.ocoeffs['pup_l'], l0) + self.ocoeffs['pup_I'])
        #    #print(nM[self.Q0])
        #    self.check_real(i, nM, 'Photon number')
        #    nMs[i,:] = nM.real
        #self.dynamics_mf2 = {'ak':aKs, 'nP': nPs, 'nM': nMs}

    
    def evolve(self, tend=100.0, atol=1e-8, rtol=1e-6):
        """Integrate second-order cumulants equations of motion from t=0 to  t=tend (femptoseconds)"""
        dt_fs = self.params['dt']
        self.t_fs = np.arange(0.0, tend+dt_fs/2, step=dt_fs)
        self.t = self.t_fs / self.EV_TO_FS
        dt = dt_fs / self.EV_TO_FS
        self.num_t = len(self.t)
        state_MB = sys.getsizeof(self.initial_state) / 1024**2
        #logger.info(f'Number of variables {len(self.initial_state):.2e}, '\
        #        f'estimate ~{8*state_MB:.0f} MB to propagate dynamics')
        logger.info(f'Integrating 2nd-order EoMs to {tend:.0f}fs with interpolation'\
                f' to fixed grid with dt={dt_fs:.3g}fs...')
        self.setup_dynamics_storage() # creates self.dynamics data dictionary
        #
        t_index = 0 # indicates current position in output grid of times
        num_checkpoints = 5 # checkpoints at 0, 25%,...
        checkpoint_spacing = int(round(self.num_t/num_checkpoints))
        checkpoints = np.linspace(0, self.num_t-1, num=num_checkpoints, dtype=int)
        next_check_i = 1
        last_solver_i = 0
        solver_t = [] # keep track of solver times too (not fixed grid)
        tic = time() # time the computation
        rk45 = SOLVER(self.eoms,
                    t0=0.0,
                    y0=self.initial_state,
                    t_bound=self.t[-1],
                    atol=atol,
                    rtol=rtol,
                    )
        # Save initial state
        assert rk45.t == self.t[t_index], 'Solver initial time incorrect'
        self.record_dynamics(t_index, rk45.y) # record physical dynamics for initial state
        solver_t.append(rk45.t)
        t_index += 1
        next_t = self.t[t_index]
        while rk45.status == 'running':
            end = False # flag to break integration loop
            step_message = rk45.step() # perform one step (necessary before call to dense_output())
            solver_t.append(rk45.t)
            if rk45.t >= next_t: # solver has gone past one (or more) of our grid points, so now evaluate soln
                soln = rk45.dense_output() # interpolation function for the last timestep
                while rk45.t >= next_t: # until soln has been evaluated at all grid points up to solver time
                    y = soln(next_t)
                    self.record_dynamics(t_index, y) # extract relevant dynamics from state y 
                    t_index += 1
                    if t_index >= self.num_t: # reached the end of our grid, stop solver
                        end = True
                        break
                    next_t = self.t[t_index]
            if next_check_i < num_checkpoints and t_index >= checkpoints[next_check_i]:
                solver_diffs = np.diff(solver_t[last_solver_i:])
                logger.info('{:.0f}% ({:.0f}s)'.format(100*(checkpoints[next_check_i]+1)/self.num_t, time()-tic))
                solver_dt = np.mean(solver_diffs)
                if not np.isclose(solver_dt, dt, atol=0.0, rtol=1.0):
                    if solver_dt < dt:
                        logger.warning('Average solver step size {:.2g}fs is far smaller'\
                            ' than target grid spacing {}fs. Consider decreasing parameter dt.'.format(
                                solver_dt * self.EV_TO_FS, dt_fs))
                    #elif solver_dt > 10.0 * dt:
                    #    logger.warning('Average solver step size {:.2g}fs is far larger'\
                    #        ' than target grid spacing {}fs. Consider increasing parameter dt.'.format(
                    #            solver_dt * self.EV_TO_FS, dt_fs))
                next_check_i += 1
                last_solver_i = len(solver_t) - 1
            if end:
                break # safety, stop solver if we have already calculated state at self.t[-1]
        toc = time()
        self.compute_time = toc-tic # ptoc-ptic
        if rk45.status == 'failed':
            logger.warning(f'RK45 solver failed at t={rk45.t:.1f} with message "{step_message}"')
        self.solver_info = {'method': 'RK45', 't0': 0.0, 'tend': tend, 'atol': atol, 'rtol': rtol,
                            'solver_tend': rk45.t, 'status': rk45.status, 
                            'runtime': self.compute_time}
        #logger.info('...done ({:.0f}s)'.format(self.compute_time))
        self.rescale_state(y)
        self.results = {'parameters': self.params,
                        'dynamics': self.dynamics,
                        'solver_info': self.solver_info,
                        'final_state': y, # save entire final state
                        }
        return self.results


    def setup_dynamics_storage(self):
        """Prepare dictionary self.dynamics to store values of relevant dynamics
        These arrays (or arrays in dictionaries) are zero initialised and then assigned
        non-zero values in place by self.record_dynamics during the computation
        """
        Nt = self.num_t
        aTMks = np.zeros((Nt, 2,2), dtype=complex) # two-mode model
        nPs = np.zeros((Nt, self.Nk), dtype=float)
        nKs = np.zeros((Nt, self.Nk), dtype=float)
        nMs = np.zeros((Nt, self.Nk), dtype=float)
        nBs = np.zeros((Nt, self.Nk), dtype=float)
        g1s = np.zeros((Nt, self.Nk), dtype=complex)
        #g1s = np.zeros((Nt, self.Nk, self.Nk), dtype=complex)
        Vs = np.zeros((Nt, self.Q0+1), dtype=float)
        vpops = np.zeros((Nt, self.Nnu, self.Nk), dtype=float) # 2024-05-05 r_n index now last
        ph_dic = self.blank_density_dic() # 2024-05-30
        #vpops = np.zeros((Nt, self.Nk, self.Nnu), dtype=float)
        self.dynamics = {'t': self.t_fs,
                         'r': self.rs, 
                         'aTMk': aTMks,
                         'nP': nPs,
                         'nK': nKs,
                         'nM': nMs,
                         'nB': nBs,
                         'g1': g1s,
                         'V': Vs, 
                         'vpop': vpops,
                         'ph_dic':ph_dic,
                         }

    def blank_density_dic(self, dtype=float):
        Ns = self.num_t
        return {'vals': np.zeros((Ns, self.Nk), dtype=dtype),
                'mean': np.zeros((Ns,), dtype=dtype),
                'var': np.zeros((Ns,), dtype=dtype),
                'msd': np.zeros((Ns,), dtype=dtype),
                }

    def record_dynamics(self, t_index, y):
        """Calculates and saves observable values from state y at timestep t_index
        To add additional dynamics, add a key-empty array to self.dynamics e.g.
        self.dynamics['my_obs'] in self.setup_storage_dynamics and then write a
        function to take state, calculate value of observable and assign to
        self.dynamics['my_obs'][t_index]
        """
        # This is only copy of entire state we make. Essential -otherwise we would be modifying solver's state!
        # To avoid this copy overhead, don't rescale until AFTER calculating each observable (significant rewrite)
        state = y.copy()
        t = self.t[t_index]
        self.rescale_state(state) # correct scale of variables to calculate physical quantities (state modified in-place)
        ada, l, al, ll = self.split_reshape_return(state, check_rescaled=True) # VIEWS of original state i.e. modifying state will change ada, l, al and ll and vice versa
        # The following directly update the instance variable self.dynamics 
        self.calculate_photonic(t_index, ada) # Photon exciton densities
        self.calculate_electronic(t_index, l, ll) # Electronic and bright state densities 
        self.calculate_vibronic(t_index, l) # vibrational populations for emitters in each gap
        #if t_index % 100 == 0:
        #    print(' - '.join(['{:.2g}'.format(np.max(np.abs(X))) for X in [ada, l, al, ll]]))

    def calculate_photonic(self, t_index, ada):
        nk = fftshift(np.diag(ada))
        self.check_real(t_index, nk, 'Photon number (k-space)')
        alpha = ifft(ada, axis=0) # including 1/N_k normalisation!
        dft2 = fft(alpha, axis=-1) # real space so no fftshift... (start with position 0...)
        if self.params['model'] == 'two-node':
            aTMk = np.zeros((2,2), dtype=complex)
            # Better: precalculate the four-indices and do in one assignment 
            ada_shift = fftshift(ada)
            aTMk[0,0] = ada_shift[self.QL, self.QL]
            aTMk[0,1] = ada_shift[self.QL, self.QR]
            aTMk[1,0] = ada_shift[self.QR, self.QL]
            aTMk[1,1] = ada_shift[self.QR, self.QR]
            self.dynamics['aTMk'][t_index] = aTMk
        nph = np.diag(dft2) # n(r_n) when n=m
        self.check_real(t_index, nph, 'Photon density')
        mid_n = self.Q0
        g1 = np.zeros(self.Nk, dtype=complex)
        for n in self.ns:
            numer = dft2[n, mid_n]
            demon = np.sqrt(np.abs(np.real(dft2[n,n]) * np.real(dft2[mid_n, mid_n])))
            if np.isclose(demon, 0.0, atol=1e-8):
                #logger.info('Nearly zero! t_index', t_index, '   n=', n)
                g1[n] = np.zeros_like(numer)
            else:
                g1[n] = numer/demon
        # 2024-04-05 - calculate g^(1)(r,r')
        #g1 = np.zeros((self.Nk, self.Nk), dtype=complex)
        #for n1 in self.ns:
        #    for n2 in self.ns:
        #        numer = dft2[n1, n2]
        #        denom = np.sqrt(np.abs(np.real(dft2[n1,n1]) * np.real(dft2[n2, n2])))
        #        if np.isclose(denom, 0.0, atol=1e-8):
        #            g1[n1,n2] = np.zeros_like(numer)
        #        else:
        #            g1[n1,n2] = numer/denom
        # Visibility
        V = np.zeros(self.Q0+1, dtype=float)
        for n in range(self.Q0+1):
            numerV = 2 * np.abs(dft2[self.Q0+n, self.Q0-n])
            denomV = np.real(dft2[self.Q0+n,self.Q0+n]) +  np.real(dft2[self.Q0-n,self.Q0-n]) 
            if np.isclose(denomV, 0.0, atol=1e-8):
                V[n] = 0.0
            else:
                V[n] = numerV/denomV
        #self.check_real(t_index, g1s, 'First-order coherence')
        self.dynamics['nP'][t_index] = np.real(nph)
        self.dynamics['nK'][t_index] = np.real(nk)
        self.dynamics['g1'][t_index] = g1
        self.dynamics['V'][t_index] = V
        # Moments for photon density dic
        self.dynamics['ph_dic']['vals'][t_index] = np.real(nph)
        self.calculate_moments('ph_dic', t_index)

    def calculate_moments(self, name, t_index):
        weights = np.abs(self.dynamics[name]['vals'][t_index])
        if np.isclose(np.sum(weights), 0.0):
            return 3 * [np.nan]
        avg = lambda x: np.average(x, weights=weights)
        mean = avg(self.ocoeffs['rn'])
        self.dynamics[name]['mean'][t_index] = mean
        self.dynamics[name]['var'][t_index] = avg(self.ocoeffs['rn2']) - mean**2
        self.dynamics[name]['msd'][t_index] = avg(self.ocoeffs['msrn'])

    def calculate_electronic(self, t_index, l, ll):
        # N.B. we do not need to use the coefficients of the initial density
        # matrix for the identity matrix (self.coeff_eyes), that is only
        # relevant if we want to create the density matrix; instead just note if
        # OP = A lambda0 + B lambda1 + C I then <OP> = A <lambda0> + B<lambda1>
        # + C since <I>=1 should always be true, i.e. Tr[rho]==1
        nM = self.NE * (contract('a,an->n', self.ocoeffs['pup_l'], l) + self.ocoeffs['pup_I'])
        self.check_real(t_index, nM, 'Electronic excitation density')
        self.dynamics['nM'][t_index] = np.real(nM) # Electronic excitation per gap
        # Bright state calculation
        diag_ind = self.diag_indices_Nk
        ll_diag = ll[:,:,*diag_ind] # diagonal entries at EACH i,j N.B. returns a new array
        ssll = contract('i,j,ijn->n',
                        self.consts['vsigma'],
                        self.consts['vsigma'],
                        ll_diag)
        nB = (self.NE-1) * ssll + nM/self.NE
        self.check_real(t_index, nB, 'Bright state')
        self.dynamics['nB'][t_index] = np.real(nB)

    def calculate_vibronic(self, t_index, l):
        vpops = [contract('a,an->n', self.ocoeffs['vpops'][V][0], l) + self.ocoeffs['vpops'][V][1]
                       for V in range(len(self.ocoeffs['vpops']))]
        self.check_real(t_index, vpops, 'Vibrational populations')
        #self.dynamics['vpop'][t_index] = np.real(vpops).T # transpose -> r_n index first
        self.dynamics['vpop'][t_index] = np.real(vpops)

    def rescale_state(self, state):
        """Rescale state before calculating physical dynamics
        N.B. state is modified in place and so does not need to be returned"""
        assert len(state.shape) == 1, 'state must be a 1-d array (single state)'
        ada, l, al, ll, rescaled_arr = np.split(state, self.state_split_list)
        #print('Max ada | l | al | ll = {:.0e} | {:.0e} | {:.0e} | {:.0e}'.format(*[np.max(np.abs(X)) for X in [ada, l, al, ll]]))
        assert rescaled_arr[0]==-1, 'State has already been rescaled!'
        if self.rescale_int == 0:
            ada *= self.Nm
            al *= np.sqrt(self.Nm)
        elif self.rescale_int == 1:
            al /= np.sqrt(self.Nm)
            ll /= self.Nm
        elif self.rescale_int == 2:
            al /= self.Nm
            ll /= (self.Nm * np.sqrt(self.Nm))
            ada /= np.sqrt(self.Nm)
        rescaled_arr[0] = self.rescale_int  # non-negative value indicates state has been rescaled

    def split_reshape_return(self, state, check_rescaled=False, copy=False):
        """Return views of variables <a^dag_k'a_k>, <lambda_n^i>, <a_k lambda_n^i>, <lambda_n^i lamnbda_m^j>
        in state reshaped into conventional multidimensional arrays."""
        split = np.split(state, self.state_split_list) # ada, l, al, ll, rescaled_arr as flattened arrays
        if check_rescaled:
            assert not np.isnan(split[-1][0]), 'State must be rescaled'
        # reshape each array except the 'rescaled_arr' (contains int indicating rescaling factor used)
        reshaped = [split[i].reshape(self.state_reshape_list[i]) for i in range(len(self.state_reshape_list))]
        # N.B. np.split, reshape returns VIEWS of original array; BEWARE mutations 
        # - copy if want to change return without mutating original state variable
        if copy:
            reshaped = [np.copy(X) for X in reshaped]
        return reshaped

    def split_reshape_return_mf(self, state, copy=False):
        split = np.split(state, self.state_split_list_mf)
        reshaped = [split[i].reshape(self.state_reshape_list_mf[i]) for \
                    i in range(len(self.state_reshape_list_mf))]
        #split = np.split(state, [self.Nk, self.Nk + self.Nnu**2*self.Nk]) # a, lp, l0
        #split[1] = split[1].reshape((self.Nnu**2, self.Nk))
        #split[2] = split[2].reshape((2*self.Nnu**2-1, self.Nk))
        #if copy:
        #    split = [np.copy(X) for X in split]
        #return split
        if copy:
            reshaped = [np.copy(X) for X in reshaped]
        return reshaped

    #def eoms_mf(self, t, state):
    #    #2024-06-21 mf eoms - no spatial dependence in coefficients (untested)
    #    C = self.mf_coeffs
    #    a, lp, l0 = self.split_reshape_return_mf(state)
    #    pre_c = C['iB_1'] @ lp
    #    c = fft(pre_c, norm='forward')
    #    alpha = fft(a, norm='backward')
    #    dy_a = C['11_k'] @ a + c.conj()
    #    dy_lp = C['21_11'] @ lp + C['22_10'] @ (l0 * alpha.conj())
    #    dy_l0 = C['31_00'] @ l0 + C['32_0'] + (C['33_01'] @ (lp * alpha)).real
    #    dy_state = np.concatenate((dy_a, dy_lp, dy_l0), axis=None)
    #    return dy_state
    
    def eoms_mf(self, t, state):
        #2024-06-21 mf eoms - with spatially dependence pump (untested)
        C = self.coeffs_mf
        a, lp, l0 = self.split_reshape_return_mf(state)
        pre_c = C['iB_1'] @ lp
        c = fft(pre_c, norm='forward')
        alpha = fft(a, norm='backward')
        dy_a = C['11_k'] @ a + c.conj()
        dy_lp = contract('ijn,jn->in', C['21_11n'], lp) + C['22_10'] @ (l0 * alpha.conj())
        dy_l0 = contract('abn,bn->an', C['31_00n'], l0) + C['32_0n'] + (C['33_01'] @ (lp * alpha)).real
        dy_state = np.concatenate((dy_a, dy_lp, dy_l0), axis=None)
        return dy_state

    def eoms(self, t, state):
        # 2024-04-02 elements now with n dependence: 21_00n, 31_11kn, 41_11n, 42_11n
        """Equations of motion as in cumulant_in_code.pdf"""
        C = self.coeffs
        ada, l, al, ll = self.split_reshape_return(state)
        # If two-node model check relevant elements are absolutely zero
        if self.params['model'] == 'two-node':
            assert np.allclose(ada[self.mode_mask['ada_args']], 0.0)
            assert np.allclose(al[self.mode_mask['al_args']], 0.0)
#        if np.isclose(t%40, 0.0, atol=0.25):
#            print('Maxes: ada {:.1g} l {:.1g} al {:.1g} ll {:.1g}'.format(*[np.max(np.abs(X)) for X in [ada, l, al, ll]]))
        # Calculate DFTs
        alpha = ifft(ada, axis=0, norm='forward')
        d = fft(al, axis=1, norm='backward') # backward default
        # EQ 1 # N.B. kp->pk ordering
        pre_c = contract('i,ikn->kn', C['12_1'], al)
        post_c = fft(pre_c, axis=1, norm='forward')
        dy_ada2 = C['11_pk'] * ada + np.transpose(post_c) + np.conj(post_c)
        # EQ 2
        dy_l = contract('abn,bn->an', C['21_00n'], l) + C['22_0n'] + contract('aj,jnn->an', C['23_01'], d).real
        # EQ 3
        pre_beta = contract('j,ijnm->imn', C['32_1'], ll) # N.B. swapped axes
        post_beta = ifft(pre_beta, axis=-2, norm='backward') # See Eqs.
        dy_al2 = contract('ijkn,jkn->ikn', C['31_11kn'], al) \
                + post_beta \
                + contract('jkn,ijnn->ikn', C['33_1kn'], ll) \
                + contract('ikna,an->ikn', C['34_1kn0'], l) \
                + C['35_1kn'] \
                + contract('ai,an,nk->ikn', C['36_01'], l, alpha) 
        # EQ 4
        dy_ll = contract('ipn,pjnm->ijnm', C['41_11n'], ll) \
                + contract('jpn,ipnm->ijnm', C['42_11n'], ll) \
                + contract('aj,am,imn->ijnm', C['43_01'], l, d) \
                + contract('ai,an,jnm->ijnm', C['44_01'], l, d.conj())
        dy_rescale_int = np.zeros(1)
        # if model two-mode, mask all but two photon modes
        if self.params['model'] == 'two-node':
            dy_ada2[self.mode_mask['ada_args']] = 0.0
            dy_al2[self.mode_mask['al_args']] = 0.0 
        # flatten and concatenate to match input state structure (1d array)
        dy_state = np.concatenate((dy_ada2, dy_l, dy_al2, dy_ll, dy_rescale_int), axis=None)
        return dy_state

    WARN_REAL = {}
    def check_real(self, step, arr, name):
        if name not in self.WARN_REAL:
            self.WARN_REAL[name] = True
        if not self.WARN_REAL[name]:
            return
        if not np.allclose(np.imag(arr), 0.0):
            t = self.t[step]
            logger.warning(f'{name} at t={t} has non-zero imaginary part (further warnings suppressed)')
            self.WARN_REAL[name] = False


    def export_data(self, fp=None):
        if fp is None:
            fp = self.gen_fp()
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'wb') as fb:
            pickle.dump(self.results, fb)
        logger.info(f'Wrote parameters & dynamics data to {fp}')

    def import_data(self, fp=None):
        if fp is None:
            fp = self.generate_fp()
        with open(fp, 'rb') as fb:
            self.dynamics = pickle.load(fb)
        logger.info('Loaded parameters and dynamics data from {fp}')

    def gen_fp(self):
        fname = 'Nnu{Nnu}/Nk{Nk}/gn{gSqrtN}S{S}Gamz{Gam_z}.pkl'.format(**self.params)
        return os.path.join(self.DEFAULT_DIRS['data'], fname)


    def get_labels(self):
        return {'K': r'\(K\)',
                'k': r'\(k\ \rm{\text{(}}\mu\text{\rm{m}}^{-1}\text{\rm{)}}\)',
                't': r'\(t\)',
                't_fs': r'\(t\) \rm{(fs)}',
                #'rn': r'\(r_n\) \rm{(nm)}',
                'rn': r'\(r_n\) \rm{(}\(\mu\)\rm{m)}',
                'ph_rn': r'\(n_{\rm{\text{\rm{ph}}}}(t, r_n)\)',
                'ph_rn0': r'\(n_{\rm{\text{\rm{ph}}}}(t, r_n)-n_{\rm{\text{ph}}}(0, r_n) \)',
                'ph_rms': r'\(\sqrt{\text{\rm{MSD}}[n_{\text{ph}}]}\) \rm{(}\(\mu\)\rm{m}\({}^2\)\rm{)}',
                'mol_rn': r'\(n_{M}(t, r_n)\)',
                'mol_rn0': r'\(n_{M}(t, r_n)-n_M(0,r_n)\)',
                'mol_rms': r'\(\sqrt{\text{\rm{MSD}}[n_{M}]}\) \rm{(}\(\mu\)\rm{m}\({}^2\)\rm{)}',
                'ph0nM0': r'\(\Delta n_{\rm{\text{ph}}}+ \Delta n_{M}\)',
                'coh': r'\(\lvert\langle a \sigma^+\rangle\rvert(t,r_n)\)',
                'nB': r'\(n_{\mathcal{B}}(t,r_n)\)',
                'nB0': r'\(n_{\mathcal{B}}(t,r_n)-n_{\mathcal{B}}(0,r_n)\)',
                'nD': r'\(n_{\mathcal{D}}(t,r_n)\)',
                'nD0': r'\(n_{\mathcal{D}}(t,r_n)-n_{\mathcal{B}}(0,r_n)\)',
                'Dph': r'\(\Delta n_{\rm{\text{ph}}}\)',
                'DnM': r'\(\Delta n_{M}\)',
                'DnB': r'\(\Delta n_{\mathcal{B}}\)',
                'DnD': r'\(\Delta n_{\mathcal{D}}\)',
                'D': r'\(\Delta n_X(t) = \sum_{n} \left(n_X(t, r_n) - n_X(0, r_n)\right)\)',
                'Eph': r'\(n_{\text{\rm{ph}}}\)',
                'EnM': r'\( n_{M}\)',
                'EnB': r'\( n_{\mathcal{B}}\)',
                'EnD': r'\( n_{\mathcal{D}}\)',
                'E': r'\(n_X(t) = \sum_{n} n_X(t, r_n)\)',
                }

    def plot_final_state(self, normalise=False):
        fig = plt.figure(figsize=(8,8), constrained_layout=True)
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        #ax3 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        fig.suptitle(r'Final $t={}$ fs'.format(self.dynamics['t'][-1]))
        nP = np.copy(self.dynamics['nP'][-1])
        nM = np.copy(self.dynamics['nM'][-1])
        if normalise:
            nP /= np.max(nP)
            nM /= np.max(nM)
        else:
            nM /= self.NE
        ax1.plot(self.ns, nP, label=self.labels['Eph'])
        # Analytical calculation ignoring off-diagonal terms - not currently useful 
        #n_analytic = self.n_profile()
        #ax1.plot(self.ns, n_analytic, label=r'$n_{\text{\rm{ph}}}$ \rm{(analytic)}', ls='--')
        #ax1.set_xlabel(self.labels['rn'])
        ax1.legend()
        final_vpops = self.dynamics['vpop'][-1]
        for i in range(self.Nnu):
            # 2024-05-05: Now vpop first index is level, second r_n
            ax2.plot(self.rs, final_vpops[i,:], label=r'$i={}$'.format(i))
        ax2.legend(title='level $i$')
        ax2.set_xlabel(self.labels['rn'])
        ax2.set_title(r'Vibrational populations')
        ax3.plot(self.rs, np.abs(self.dynamics['g1'][-1]), label=r'$\lvert g^{(1)}(r_n) \rvert$')
        ax4.plot(self.rs[self.Q0:], self.dynamics['V'][-1], label=r'$V(r_n)$')
        #ax3.plot(self.rs[:self.Q0+1], self.dynamics['V'][-1], label=r'$V(R)$')
        #ax3.set_title(r'$\lvert g^{(1)}(r_n) \rvert$')
        all_ns = np.linspace(0, self.Nk, 250)
        all_rs = self.params['delta_r'] * all_ns * 1e-03
        all_pumps = self.pump(all_ns)
        ax3.plot(all_rs, all_pumps/np.max(all_pumps), ls='--', c='k',
                 label=r'$\Gamma_\uparrow(r_n)/\Gamma_\uparrow(0)$')
        ax4.plot(all_rs[len(all_rs)//2:], all_pumps[len(all_pumps)//2:]/np.max(all_pumps), ls='--', c='k',
                 label=r'$\Gamma_\uparrow(r_n)/\Gamma_\uparrow(0)$')
        ax3.set_xlabel(self.labels['rn'])
        ax4.set_xlabel(self.labels['rn'])
        ax3.legend()
        ax4.legend()
        fp = os.path.join(self.DEFAULT_DIRS['figures'], 'final_state.png')
        fig.savefig(fp, bbox_inches='tight', dpi=350)
        plt.close(fig)
        fig2, ax2 = plt.subplots(figsize=(4,4), constrained_layout=True)
        Sz = 2 * self.dynamics['nM'][-1]/self.params['NE'] - 1
        Sa = self.S_steady()
        ax2.plot(self.ns, Sz, label=r'$\langle \sigma^z_n \rangle$')
        ax2.plot(self.ns, Sa, label=r'$\Gamma_\uparrow(n)/\Gamma_\downarrow$', ls='--')
        #ax2.plot(self.ns, Sa, label=r'\rm{analytical}')
        ax2.legend()
        fp2 = os.path.join(self.DEFAULT_DIRS['figures'], 'polarisation.png')
        fig2.savefig(fp2, dpi=350, bbox_inches='tight')
        logger.info(f'Final state plots saved to {fp} and {fp2}.')
        plt.close(fig2)

    def fit_early(self, name, cutoff=0.01, early_i1=1, early_i2=None):
        valid_i = 1 # ignore first datum
        dense_dic = self.dynamics[name]
        avgP, errorP, msdP = dense_dic['mean'], np.sqrt(dense_dic['var']), dense_dic['msd']
        plot_tP = self.dynamics['t'][valid_i:]
        fs = 1/(plot_tP[1]-plot_tP[0])
        cutoffFS = cutoff * fs
        msdP = self.butter_lowpass_filter(msdP[valid_i:], cutoffFS, fs)
        to_plot = np.sqrt(msdP)
        early_i1 -= valid_i 
        def f(t, D, po):
            return D * (t-plot_tP[early_i1])**po + to_plot[early_i1]
        if early_i2 is None:
            early_i2 = -1
        else:
            early_i2 -= valid_i
        if len(plot_tP) > 10: # need reasonable number of points
            popt, pcov = curve_fit(f, plot_tP[early_i1:early_i2],
                                   to_plot[early_i1:early_i2], 
                                   bounds=([0,0],[np.inf, 2]))
            fit_data = f(plot_tP[early_i1:], *popt)
            fit = [fit_data, popt]
            return plot_tP, to_plot, fit
        return plot_tP, to_plot, None
        logger.warning('Too few data points to generate fit')

    def plot_dynamics(self):
        fig, axes = plt.subplots(2, 2, figsize=(8,8), constrained_layout=True)
        t_fs = self.dynamics['t']
        extent = [0, 1.01*self.params['L']*1e-3, t_fs[0], t_fs[-1]]
        cm = colormaps['coolwarm'] 
        my_im = lambda axis, vals: axis.imshow(vals, origin='lower', aspect='auto',
                                           interpolation='none', extent=extent, cmap=cm)
        if self.params['model'] == 'two-node':
            # N.B. nK was already fftshifted to -Q0,...0,...Q0 ordering shifted
            axes[0,0].plot(t_fs, self.dynamics['nK'][:,self.QL], label=r'$1=-Q_{\text{half}}$') # NON-ZERO
            axes[0,0].plot(t_fs, self.dynamics['nK'][:,self.QR], label=r'$2=+Q_{\text{half}}$') # NON-ZERO
            axes[0,0].legend(title=r'$k$')
            axes[0,0].set_xlabel(self.labels['t_fs'])
            axes[0,0].set_title(r'$n_k$')
            axes[0,1].plot(t_fs, self.dynamics['aTMk'][:,0,1].real, label=r'Re')
            axes[0,1].plot(t_fs, self.dynamics['aTMk'][:,1,0].imag, label=r'Im')
            axes[0,1].legend()
            axes[0,1].set_xlabel(self.labels['t_fs'])
            axes[0,1].set_title(r'$\langle a^\dagger_{12} a_{21}^{\vphantom{\dagger}} \rangle$')
        else:
            axes[0,1].set_title(self.labels['mol_rn0'])
            im0 = my_im(axes[0,0], self.dynamics['nP'])
            cbar0 = fig.colorbar(im0, ax=axes[0,0], aspect=20)
            axes[0,0].set_xlabel(self.labels['rn'])
            axes[0,0].set_ylabel(self.labels['t_fs'])
            axes[0,0].set_title(self.labels['ph_rn'])
            axes[0,1].set_xlabel(self.labels['rn'])
            axes[0,1].set_ylabel(self.labels['t_fs'])
            #zeroed_nM = self.dynamics['nM'] - self.dynamics['nM'][0,:] # subtract initial pops
            #im1 = my_im(axes[0,1], zeroed_nM)
            im1 = my_im(axes[0,1], self.dynamics['nM'])
            cbar1 = fig.colorbar(im1, ax=axes[0,1], aspect=20)
        nPh_tots = np.sum(self.dynamics['nP'], axis=-1)
        nM_tots = np.sum(self.dynamics['nM'], axis=-1)
        nB_tots = np.sum(self.dynamics['nB'], axis=-1)
        nD_tots = nM_tots - nB_tots
        axes[1,0].plot(t_fs, nPh_tots, label=self.labels['Eph'])
        axes[1,0].plot(t_fs, nM_tots, label=self.labels['EnM'])
        axes[1,0].plot(t_fs, nB_tots, label=self.labels['EnB'])
        axes[1,0].plot(t_fs, nD_tots, label=self.labels['EnD'])
        axes[1,0].set_title(self.labels['E'])
        axes[1,0].set_xlabel(self.labels['t_fs'])
        axes[1,0].legend()
        self.plot_parameters(axes[1,1])
        fig3, ax3 = plt.subplots(figsize=(4,4), constrained_layout=True)
        #ax3.plot(smooth_t, smooth_msd, label=self.labels['Eph'])
        cutoff = 0.01
        cutoffFS = 0.01/self.EV_TO_FS
        rn_scale = 1e-3 # nm to mu.m
        early_t1 = 0
        early_t2 = 20
        early_i1 = next((j for j, t in enumerate(t_fs) if t > early_t1), None)
        early_i2 = next((j for j, t in enumerate(t_fs) if t > early_t2), None)
        smooth_t, smooth_msd, fit = self.fit_early('ph_dic', cutoff, early_i1, early_i2)
        ax3.plot(smooth_t, smooth_msd*rn_scale,
                 label=r'\rm{{Lowpass (}}\({:.1g} \text{{\rm{{fs}}}}^{{-1}}\)\rm{{)}}'.format(cutoffFS))
        popt = fit[1]
        if self.params['model'] != 'two-node':
            axes[0,0].plot(smooth_msd*rn_scale+self.rs[self.Q0], smooth_t, ls='--', color='lime')
        offset = max(0, early_i1-1)
        end = offset + early_i2-early_i1
        ax3.plot(smooth_t[offset:end], fit[0][:early_i2-early_i1]*rn_scale, ls='--',
                 label=r'\(D,\alpha=({:.2g},{:.2g})\)'.format(popt[0]*rn_scale, popt[1])\
                         +'\n'+ r'\rm{{fit on ({},{})}}'.format(early_t1,early_t2))
        #early_t1b = 20
        #early_t2b = 60
        #early_i1b = next((j for j, t in enumerate(t_fs) if t > early_t1b), None)
        #early_i2b = next((j for j, t in enumerate(t_fs) if t > early_t2b), None)
        #smooth_tb, smooth_msdb, fitb = self.fit_early('ph_dic', cutoff, early_i1b, early_i2b)
        #poptb = fitb[1]
        #ax3.plot(smooth_tb[early_i1b:early_i2b], fitb[0][:early_i2b-early_i1b]*rn_scale, ls='--',
        #         label=r'\(D,\alpha=({:.2g},{:.2g})\)'.format(poptb[0]*rn_scale, poptb[1])\
        #                 +'\n'+ r'\rm{{fit on ({},{})}}'.format(early_t1b,early_t2b))
        #ax3.set_title(r'\rm{RMSD[}'+self.labels['Eph']+r'\rm{] (}$\mu$\rm{m), fit} $Dt^\alpha$')
        ax3.set_title(r'$t={}\quad \kappa={}$'.format(self.params['t'], self.params['kappa']))
        ax3.set_xlabel(self.labels['t_fs'])
        ax3.legend()
        fp = os.path.join(self.DEFAULT_DIRS['figures'], 'dynamics.png')
        fig.savefig(fp, bbox_inches='tight', dpi=350)
        plt.close(fig)
        fp3 = os.path.join(self.DEFAULT_DIRS['figures'], 'fit_dynamics.png')
        fig3.savefig(fp3, bbox_inches='tight', dpi=350)
        logger.info(f'Dynamics plots saved to {fp} and {fp3}.')
        plt.close(fig3)
        #fig2, ax2 = plt.subplots(figsize=(4,4), constrained_layout=True)
        #ax2.set_xlabel(self.labels['t_fs'])
        #ax2.set_ylabel(r'$\langle \sigma^z_n \rangle$', rotation=0)
        #choseni = [self.Q0, self.Q0+5, self.Q0+10]
        #for i in choseni:
        #    szi = 2 * self.dynamics['nM'][:,i]/self.params['NE'] - 1
        #    sai = self.S_decay(i, self.t)
        #    p0 = ax2.plot(self.t_fs, szi, label=r'${}$'.format(i))
        #    ax2.plot(self.t_fs, sai, ls='--', color='k')
        #    #ax2.plot(self.t_fs, sai, ls='--', color=p0[0].get_color())
        #ax2.legend(title=r'Site index')
        #fp2 = os.path.join(self.DEFAULT_DIRS['figures'], 'decay_dynamics.png')
        #fig2.savefig(fp2, bbox_inches='tight', dpi=350)
        #logger.info(f'Dynamics plots saved to {fp} and {fp2}.')
        #plt.close(fig2)

    def S_decay(self, n, t):
        GT = self.pump(n) + self.params['decay']
        DG = GT - 2*self.params['decay']
        return (1/GT) * (DG - 2 * self.pump(n) * np.exp(- GT * t))

    def plot_parameters(self, ax):
        params = self.params
        L = params['L'] * 1e-3
        size_params = \
            [#r'\rm{Size}',
             r'\(N_E={}\quad N_k={}\quad N_\nu={}\)'.format(self.NE, self.Nk, self.Nnu),
             r'\(a={:.0f} \text{{\rm{{nm}}}}\quad w={:.0f} \text{{\rm{{nm}}}}\quad L={:.0f} \mu\text{{\rm{{m}}}}\)'.format(params['a'], params['w'], L),
            ]
        sys_params = \
            [
             #'\n',
             r'\rm{System (eV)}',
             r'\(\omega_p={}\)'.format(params['omega_p']),
             r'\(\omega_0={}\)'.format(params['omega_0']),
             r'\(g={:.3g}\)'.format(params['g']),
             r'\(g\sqrt{{N_E}}={:.3g}\)'.format(params['gSqrtNE']),
             r'\(t={:.3g}\)'.format(params['t']),
            ]
        rate_params = \
            [
             #'\n',
             r'\rm{Rates (eV)}',
             r'\(\kappa={}\)'.format(params['kappa']),
             r'\(\Gamma_\uparrow(0)={}\)'.format(params['pump_strength']),
             r'\(\Gamma_\downarrow={}\)'.format(params['decay']),
             r'\(\Gamma_z={}\)'.format(params['dephase']),
            ]
        def pow_str(flo, prec=1):
            flo = float(flo)
            r0a = '{:.{prec}e}'.format(flo, prec=prec).split('e')
            r0a[1] = r0a[1].replace('0','')
            return r'\({}\!\times\!10^{{{}}}\)'.format(*r0a)
        bath_params = \
            [
             #'\n',
             r'\rm{Bath (eV)}',
             r'\(S={}\)'.format(params['S']),
             r'\(\omega_\nu={}\)'.format(params['omega_nu']),
             r'\(T={}\)'.format(params['T']),
             r'\(\gamma_\nu={}\)'.format(params['gam_nu']),
             r'\(\gamma_\uparrow= \) {}'.format(pow_str(self.rates['gam_up'])),
             r'\(\gamma_\downarrow= \) {}'.format(pow_str(self.rates['gam_down'])),
             ]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.text(0.5, 0.975, '\n'.join(size_params),
                ha='center', va='top', transform=ax.transAxes, size='medium') # axis coords
        ax.text(0.25, 0.8, '\n'.join(sys_params),
                ha='center', va='top', transform=ax.transAxes, size='medium')
        ax.text(0.75, 0.8, '\n'.join(rate_params),
                ha='center', va='top', transform=ax.transAxes, size='medium')
        ax.text(0.25, 0.425, '\n'.join(bath_params),
                ha='center', va='top', transform=ax.transAxes, size='medium')

    def plot_dispersion_and_pump(self):
        fig, axes = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
        fig.suptitle(f'Emitter positions / discrete modes shown in red ($N_k={self.Nk}$)')
        axes[0].set_xlabel(r'$k$ \rm{(}$\mu$\rm{m}${}^{-1}$\rm{)}')
        axes[0].set_title(r'$\hbar\omega_k$ \rm{(eV)}')
        #axes[1].set_xlabel(self.labels['rn'])
        axes[1].set_title(r'$\Gamma_\uparrow(r_n)\ (\sigma={}$\rm{{nm}}$)$'.format(
            self.params['pump_width']))
        all_Ks = np.linspace(self.Ks[0],self.Ks[-1], 250)
        #all_ks = (2*np.pi/self.params['L']) * all_Ks * 1e3
        all_y = self.omega(all_Ks)
        all_ns = np.linspace(0, self.Nk, 250)
        all_rs = self.params['delta_r'] * all_ns * 1e-03
        all_pumps = self.pump(all_ns)
        select_pumps = self.pump(self.ns)
        axes[0].plot(all_Ks, all_y)
        Q0=self.Q0
        Nk=self.Nk
        #ticks = [-Q0, -round(Q0/2), 0, round(Q0/2), Q0]
        ticks = [-(Nk/2), -(Nk/4), 0, (Nk/4), (Nk/2)]
        tick_labels = [r'$-\pi/\Delta r$', r'$-\pi/(2\Delta r)$',r'$0$', r'$\pi/2(\Delta r) $',r'$\pi/\Delta r $']
        axes[0].set_xlim([-(Nk/2), (Nk/2)])
        axes[0].set_xticks(ticks)
        axes[0].set_xticklabels(tick_labels)
        #axes[0].scatter(self.ks * 1e3, chosen_y, c='r', s=8, zorder=2)
        axes[0].axhline(self.params['omega_0']-self.params['omega_p'], c='r', label=r'$\omega_0-\omega_p$')
        if self.params['model'] == 'two-node':
            KL, KR = self.Ks[self.QL], self.Ks[self.QR]
            ws = self.omega(np.array([KL,KR]))
            axes[0].scatter([KL,KR], ws, c='m', s=80, marker='x', label=r'\rm{two-node}',zorder=2)
        axes[0].legend()
        axes[1].plot(all_ns, all_pumps)
        axes[1].scatter(self.ns, select_pumps, c='r', s=8, zorder=2)
        fp = os.path.join(self.DEFAULT_DIRS['figures'], 'dispersion_pump.png')
        fig.savefig(fp, bbox_inches='tight', dpi=350)
        logger.info(f'Dispersion and pump profile saved to {fp}.')
        plt.close(fig)
    
    @staticmethod
    def J(n_nm, t=1.0):
        J_R = 1j*t*np.concatenate(([n_nm[0,-1]], np.diag(n_nm, k=-1)))
        J_L = 1j*t*np.concatenate(([n_nm[-1,0]], np.diag(n_nm, k=+1)))
        return J_L, J_R

    def S_steady(self):
        params = self.params
        if not np.isclose(params['dephase'], 0.0):
            logger.critical('S solution for 0 pure dephasing only')
            return np.zeros_like(self.ns)
        GT = self.pump(self.ns) + params['decay']
        DG = self.pump(self.ns) - params['decay']
        k = params['kappa']
        g = params['g']
        NE = params['NE']
        w0 = params['omega_0']
        Gd = params['decay']
        return DG/GT
        #numer = GT * k * (8 * g**2 * w0 + 2 * Gd * ((GT+k)**2+4*w0**2))
        #denom = 4 * g**2 * (2 * Gd * ( GT + k) * (NE * (GT-k) + k) - 2 * GT * k * w0)
        #return numer/denom

    def n_profile(self):
        params = self.params
        if not np.isclose(params['dephase'], 0.0) or not np.isclose(params['omega_0'],0.0):
            logger.critical('n solution for 0 pure dephasing and on resonance')
            return np.zeros_like(self.ns)
        Gu = self.pump(self.ns)
        Gd = params['decay']
        GT = Gu + Gd
        DG = Gu - Gd
        kap = params['kappa']
        g = params['g']
        NE = params['NE']
        numer = 4 * g**2 * NE * GT * Gu
        denom = GT**2 * kap * (GT + kap) + 4 * DG * g**2 * (kap - NE * (GT + kap))
        return numer/denom

    def cauchy_mask(self, ada):
        delta = 1e-8
        ada = fftshift(ada)
        diags = np.diag(ada)
        Nk = len(diags)
        ada_p = np.array([x * np.ones(Nk) for x in diags])
        ada_k = ada_p.T
        #mask = np.abs(ada)**2 > ada_p * ada_k + delta # delta for numerical approx.
        diff =   ada_p * ada_k - np.abs(ada)**2
        mask = diff < - delta # delta for numerical tolerance
        return mask, diff 

def plot_input_output(parameters, pump_strengths, tend=100, cauchy_mask=True, xlims=None):
    # 2024-04-16
    params = parameters
    num_pumps = len(pump_strengths)
    ratios = np.array(pump_strengths) / params['decay']
    Nk = 2*params['Q0']+1
    nph_final = np.zeros((num_pumps, Nk), dtype=float)
    nK_final = np.zeros((num_pumps, Nk), dtype=float)
    nM_final = np.zeros((num_pumps, Nk), dtype=float)
    nvpop_final = np.zeros((num_pumps, Nk), dtype=float)
    evpops_final = np.zeros((num_pumps, params['Nnu'], Nk), dtype=float) # vibrational pops conditioned on electronic excited state
    gvpops_final = np.zeros((num_pumps, params['Nnu'], Nk), dtype=float) # electronic ground state
    adaga_final = np.zeros((num_pumps, Nk, Nk), dtype=complex) 
    adaga_final_mask = np.zeros((num_pumps, Nk, Nk), dtype=bool) 
    adaga_final_diff = np.zeros((num_pumps, Nk, Nk), dtype=bool) 
    n_nm_final = np.zeros((num_pumps, Nk, Nk), dtype=complex) 
    span_final = np.zeros((num_pumps, Nk), dtype=complex) 
    JLs_final = np.zeros((num_pumps, Nk), dtype=complex)
    JRs_final = np.zeros((num_pumps, Nk), dtype=complex)
    Js_final = np.zeros((num_pumps, Nk), dtype=complex)
    for i, pump in enumerate(pump_strengths):
        logger.info(f'On pump {i+1} of {num_pumps}')
        params['pump_strength'] = pump
        htc = HTC(parameters)
        results = htc.evolve(tend=tend)
        nph_final[i, :] = results['dynamics']['nP'][-1, :]
        nK_final[i, :] = results['dynamics']['nK'][-1, :] # already fftshifted to ascending order
        nM_final[i, :] = results['dynamics']['nM'][-1, :]
        nvpop_final[i, :] = results['dynamics']['vpop'][-1, -1,:] # highest vibrational state (irrespective of electronic state)
        ada, l, al, ll = htc.split_reshape_return(results['final_state'],
                                                   check_rescaled=True) # final state variables
        n_nm_final[i, :, :] = fft(ifft(ada, axis=0), axis=1)
        JLs_final[i, :], JRs_final[i, :] = htc.J(n_nm_final[i,:,:])
        Js_final[i, :] = JRs_final[i] - JLs_final[i]
        adaga_final[i, :, :] = ada #fftshift(ada)
        if cauchy_mask:
            adaga_final_mask[i,:,:], adaga_final_diff[i,:,:] = htc.cauchy_mask(ada)
        else:
            adaga_final_mask[i,:,:] = np.zeros(ada.shape, dtype=bool) # no masked values
            adaga_final_diff[i,:,:] = np.zeros(ada.shape, dtype=bool)
        an_l = fft(al, axis=1) # index i, then k, then n ! (see EoMs). No normalisation (choice)
        #print(contract('imn,i->mn', htc.gp.basis[htc.gp.indices[1]], htc.ocoeffs['sp_l']))
        an_spn = np.diag(contract('imn,i->mn', an_l, htc.ocoeffs['sp_l']))
        span_final[i, :] = an_spn
        v_pops = results['dynamics']['vpop'][-1, :, :]
        ev_pops = np.real(
            [contract('a,an->n', htc.ocoeffs['evpops'][V][0], l) + htc.ocoeffs['evpops'][V][1]
             for V in range(len(htc.ocoeffs['evpops']))]
            ) #.T # transpose to put r_n index first (no longer needed)
        gv_pops = np.real([v_pops[j,:] - ev_pops[j,:] for j in range(params['Nnu'])])
        evpops_final[i, :, :] = ev_pops
        gvpops_final[i, :, :] = gv_pops
    htc.plot_dispersion_and_pump()
    fig, axes = plt.subplots(3, 3, figsize=(12,12), constrained_layout=True)
    nph_tots = np.sum(nph_final, axis=1) # Sum over all lattice positions 
    nM_tots = np.sum(nM_final, axis=1) # sum over all lattice positions
    axes[0,0].set_xlabel(r'$\Gamma_\uparrow(L/2)/\Gamma_\downarrow$')
    #axes[0,1].set_xlabel(r'$r_n (\mu \text{\rm{m}})$')
    axes[0,0].set_title(r'$\sum_n n_{{\text{{ph}}}}(r_n)$')
    #axes[0,0].set_title(r'$\sum_n n_{{\text{{ph}}}}(t={}, r_n)$'.format(tend))
    axes[0,0].plot(ratios, nph_tots)
    axes[0,0].set_xscale('log') # Log x-axis
    axes[0,0].set_yscale('log') # Log y-axis
    #axes[0,2].set_yscale('log') # Log y-axis
    num_cross_sections = min(6, num_pumps)  # maximum number of cross-sections
    select_indices = np.round(np.linspace(0, num_pumps-1, num_cross_sections)).astype(int)
    for i in select_indices:
        axes[0,1].plot(htc.Ks, nph_final[i, :], label=r'${}$'.format(round(ratios[i],5)))
    #axes[0,1].set_yscale('log') # Log y-axis
    axes[0,1].set_title(r'$n_{{\text{{ph}}}}(r_n)$'.format(tend))
    #axes[0,1].set_title(r'$n_{{\text{{ph}}}}(t={}, r_n)$'.format(tend))
    axes[0,1].legend(title=r'$\Gamma_\uparrow(L/2)/\Gamma_\downarrow$')
    axes[1,0].set_xlabel(r'$\Gamma_\uparrow(L/2)/\Gamma_\downarrow$')
    #axes[1,1].set_xlabel(r'$r_n (\mu \text{\rm{m}})$')
    fig.suptitle(r'\(t_{{\text{{\rm{{end}}}}}}={}\)'.format(tend) + r' \rm{(fs)}' + r' \(N_\nu={}\)'.format(params['Nnu']))
    axes[1,0].set_title(r'\rm{average inversion (over all sites)}')
    #axes[1,0].set_title(r'$(1/N_M)\sum_n (2n_{{\text{{M}}}}(t={}, r_n)/N_E-1) (av. inversion)$'.format(tend))
    axes[1,1].set_title(r'\rm{emitter inversion}')
    #axes[1,1].set_title(r'$2n_{{\text{{M}}}}(t={}, r_n)/N_E-1$ (emitter inversion)'.format(tend))
    axes[1,0].set_xscale('log') # Log x-axis
    axes[1,0].plot(ratios, 2*nM_tots/(htc.NE*htc.Nk)-1)
    chosen_nu = htc.Nnu - 1
    for i in select_indices:
        plabel = r'${}$'.format(round(ratios[i],5))
        axes[1,1].plot(htc.Ks, 2*nM_final[i, :]/htc.NE-1, label=plabel)
        axes[2,0].plot(htc.Ks, gvpops_final[i, chosen_nu, :], label=plabel)
        axes[2,1].plot(htc.Ks, evpops_final[i, chosen_nu, :], label=plabel)
        #axes[2,2].plot(htc.ns, evpops_final[i, chosen_nu, :]+gvpops_final[i, chosen_nu, :], label=plabel)
        #axes[2,2].plot(htc.rs, evpops_final[i, -1, :], label=plabel)
        #axes[0,2].plot(htc.ns, np.abs(span_final[i]), label=plabel) 
        axes[0,2].plot(htc.Ks, np.gradient(nph_final[i]), label=plabel) 
        #axes[2,2].plot(htc.ns, np.real(Js_final[i]), label=plabel)
        #axes[2,2].plot(htc.ns, np.imag(Js_final[i]), label=plabel) # almost 0
        axes[2,2].plot(htc.Ks, np.real(nK_final[i]), label=plabel)
    pump_title = r'$\Gamma_\uparrow(L/2)/\Gamma_\downarrow$'
    axes[1,1].legend(title=pump_title)
    #axes[0,2].set_title(r'$\left\lvert \langle a(r_n) \sigma^+(r_n) \rangle \right\rvert$')
    #axes[0,2].set_title(r'$\lvert \nabla \cdot n_{nm}\rvert_{nn}$')
    axes[0,2].set_title(r'$\nabla n_{{\text{{ph}}}}(r_n)$')
    axes[2,2].set_title(r'$\langle a^\dagger_k a_k \rangle $')
    #axes[2,2].set_title(r'$J_n = J^R_n-J^L_n$')
    axes[0,2].legend(title=pump_title)
    #axes[2,0].set_xlabel(r'$r_n (\mu \text{\rm{m}})$')
    #axes[2,1].set_xlabel(r'$r_n (\mu \text{\rm{m}})$')
    #axes[0,2].set_xlabel(r'$r_n (\mu \text{\rm{m}})$')
    #axes[2,2].set_xlabel(r'$r_n (\mu \text{\rm{m}})$')
    axes[2,0].set_title(r'\rm{pop. } ' + r'$\lvert g,\nu={} \rangle$'.format(chosen_nu))
    axes[2,1].set_title(r'\rm{pop. } ' + r'$\lvert e,\nu={} \rangle$'.format(chosen_nu))
    #axes[2,2].set_title(r'\rm{pop. } ' + r'$\lvert \nu={} \rangle$'.format(chosen_nu))
    #axes[2,2].set_title(r'$J_n$')
    axes[2,0].legend(title=pump_title)
    axes[2,1].legend(title=pump_title)
    axes[2,2].legend(title=pump_title)
    if xlims is not None:
        axes[0,1].set_xlim(xlims)
        axes[0,2].set_xlim(xlims)
        axes[1,1].set_xlim(xlims)
        axes[2,0].set_xlim(xlims)
        axes[2,1].set_xlim(xlims)
    #axes[2,2].set_axis_off()
    #axes[1,2].set_axis_off()
    params = htc.params
    pump_width = round(params['pump_width'] * np.sqrt(2) / params['delta_r'])
    para_strs = \
            [r'$N_E={}$'.format(htc.NE),
             r'$\kappa={}$'.format(params['kappa']),
             r'$t={}$'.format(params['t']),
             r'$g\sqrt{{N_E}}={:.3g}$'.format(params['gSqrtNE']),
             r'$\Gamma_z={}$'.format(params['dephase']),
             r'$\Gamma_\downarrow={}$'.format(params['decay']),
             r'\rm{{pumpwidth}}$\ ={}$ \rm{{sites}}'.format(pump_width),
             r'$\omega_0={}$'.format(params['omega_0']),
             ]
    axes[1,2].get_xaxis().set_visible(False)
    axes[1,2].get_yaxis().set_visible(False)
    axes[1,2].text(0.5,0.8, '\n'.join(para_strs), ha='center', va='top', transform=axes[1,2].transAxes,
                   size='x-large')
    # k-space
    num_rows = int(np.ceil(num_cross_sections/2))
    fig2, axes2 = plt.subplots(num_rows, 2, figsize=(8,4*num_rows), constrained_layout=True)
    axes2 = axes2.flatten()
    #cm = colormaps['coolwarm'] 
    cm = colormaps['viridis'] 
    cm.set_bad('red')
    #extent = [-params['Q0'], params['Q0'], -params['Q0'], params['Q0']]
    extent = [htc.ks[0], htc.ks[-1],htc.ks[0], htc.ks[-1]]
    my_im = lambda axis, vals: axis.imshow(vals, origin='lower', aspect='auto',
                                           interpolation='none', extent=extent, cmap=cm)
    ticks = [np.pi/htc.params['delta_r'] * x for x in [-1, -0.5, 0, 0.5, 1]]
    tick_labels = [r'$-\pi/\Delta r$', r'$-\pi/2\Delta r$',r'$0$', r'$\pi/2\Delta r $',r'$\pi/\Delta r $']
    for i_a, i in enumerate(select_indices):
        plabel = r'${}$'.format(round(ratios[i_a],5))
        adaga_one = np.ma.masked_array(np.copy(fftshift(adaga_final[i,:,:])),
                                         mask=fftshift(adaga_final_mask[i,:,:]))
        #im = my_im(axes2[i_a], fftshift(adaga_final[i,:,:].real))
        im = my_im(axes2[i_a], adaga_one.real)
        cbar = fig2.colorbar(im, ax=axes2[i_a], aspect=20)
        axes2[i_a].set_xlabel(r'$k^\prime$', rotation=0)
        axes2[i_a].set_ylabel(r'$k$', rotation=0, labelpad=-20)
        axes2[i_a].set_xticks(ticks)
        axes2[i_a].set_xticklabels(tick_labels)
        axes2[i_a].set_yticks(ticks)
        axes2[i_a].set_yticklabels(tick_labels)
        axes2[i_a].set_title(pump_title + r'$=$'+plabel)
    fig2.suptitle(r'Re$\langle a^{\dagger}_{k^\prime} a_k \rangle$')
    fp2 = os.path.join(htc.DEFAULT_DIRS['figures'], 'adaggera.png')
    fig2.savefig(fp2, bbox_inches='tight', dpi=600)
    plt.close(fig2)
    fig3, axes3 = plt.subplots(num_rows, 2, figsize=(8,4*num_rows), constrained_layout=True)
    axes3 = axes3.flatten()
    cm = colormaps['viridis'] 
    extent = [-params['Q0'], params['Q0'], -params['Q0'], params['Q0']]
    for i_a, i in enumerate(select_indices):
        plabel = r'${}$'.format(round(ratios[i_a],5))
        im = my_im(axes3[i_a], np.abs(n_nm_final[i,:,:]))
        cbar = fig3.colorbar(im, ax=axes3[i_a], aspect=20)
        axes3[i_a].set_xlabel(r'$m$')
        axes3[i_a].set_ylabel(r'$n$', rotation=0)
        axes3[i_a].set_title(pump_title + r'$=$'+plabel)
    fig3.suptitle(r'$\lvert n_{nm} \rvert$')
    fp3 = os.path.join(htc.DEFAULT_DIRS['figures'], 'n_nm.png')
    fig3.savefig(fp3, bbox_inches='tight', dpi=600)
    plt.close(fig3)
    logger.info(f'<a^dagger_k a_k> plot saved to {fp2}, n_nm plot to {fp3}')
    fig4, axes4 = plt.subplots(num_rows, 2, figsize=(8,4*num_rows), constrained_layout=True)
    axes4 = axes4.flatten()
    cm = colormaps['viridis'] 
    extent = [-params['Q0'], params['Q0'], -params['Q0'], params['Q0']]
    for i_a, i in enumerate(select_indices):
        plabel = r'${}$'.format(round(ratios[i_a],5))
        abs_div = np.abs(np.sum(np.gradient(n_nm_final[i,:,:]),axis=0))
        #axes[0,2].plot(htc.ns, np.diag(abs_div), label=plabel)
        #axes[0,2].plot(htc.ns[1:],np.diag(n_nm_final[i,:,:], k=-1))
        im = my_im(axes4[i_a], abs_div)
        cbar = fig4.colorbar(im, ax=axes4[i_a], aspect=20)
        axes4[i_a].set_xlabel(r'$m$')
        axes4[i_a].set_ylabel(r'$n$', rotation=0)
        axes4[i_a].set_title(pump_title + r'$=$'+plabel)
    fig4.suptitle(r'$\lvert\nabla \cdot n_{nm}\rvert$')
    fp4 = os.path.join(htc.DEFAULT_DIRS['figures'], 'n_nm_grad.png')
    fig4.savefig(fp4, bbox_inches='tight', dpi=600)
    plt.close(fig4)
    fp = os.path.join(htc.DEFAULT_DIRS['figures'], 'input_output.png')
    fig.savefig(fp, bbox_inches='tight', dpi=450)
    plt.close(fig)
    logger.info(f'n_nm gradient plot saved to {fp4}, input_output plot to {fp}')

def plot_dynamics_and_final_state(parameters):
    # 2024-04-05 - Dynamics and steady state
    htc = HTC(parameters)
    htc.plot_dispersion_and_pump()
    results = htc.evolve(tend=100)
    htc.plot_dynamics()
    htc.plot_final_state(normalise=False)
    # If want to analyse data separately use... 
    #results['dynamics']['t'] # times
    #results['dynamics']['r'] # ensemble positions (of NE emitters)
    #results['dynamics']['nP'] # Photonic excitation n_ph(t,r_n) (time 1st index, position 2nd)
    #results['dynamics']['nM'] # Emitter excitation n_M(t,r_n) 
    #results['dynamics']['nB'] # Bright state excitation n_B(t,r_n) 
    #results['dynamics']['g1'] # Normalised first-order coherence g^(1)(t,r_n,L/2)
    #results['dynamics']['vpop'] # Populations of vibrational level at each time (1st index), for each position (2nd index), for each level 0..Nnu-1 (3rd index)

def two_mode_comparison(params, plot_ph=False):
    from copy import copy
    params1 = copy(params)
    params2 = copy(params)
    params1['model'] = 'tight-binding'
    params2['model'] = 'two-node'
    htc1 = HTC(params1)
    results1 = htc1.evolve(tend=100.0)
    t1 = results1['dynamics']['t']
    nP1 = results1['dynamics']['nP']
    nM1 = results1['dynamics']['nM']
    nK1 = results1['dynamics']['nK']
    htc2 = HTC(params2)
    results2 = htc2.evolve(tend=100.0)
    t2 = results2['dynamics']['t']
    nP2 = results2['dynamics']['nP']
    nM2 = results2['dynamics']['nM']
    nK2 = results2['dynamics']['nK']
    QL, QR = htc2.QL, htc2.QR
    #adaf = results1['final_state'][:htc1.Nk**2].reshape((htc1.Nk, htc1.Nk))
    fig, axes = plt.subplots(2, 2, figsize=(8,8), constrained_layout=True)
    p1 = axes[0,0].plot(t1, nK1[:,QL].real, label=r'$k_L$')
    p2 = axes[0,0].plot(t1, nK1[:,QR].real, label=r'$k_R$')
    axes[0,0].set_xlabel(r'$t$ \rm{(fs)}')
    axes[0,0].plot(t2, nK2[:,QL].real, ls='--', color=p1[-1].get_color(), label=r'$k_L$ \rm{{({}-mode)}}'.format(htc2.num_modes))
    axes[0,0].plot(t2, nK2[:,QR].real, ls='--', color=p2[-1].get_color(), label=r'$k_R$ \rm{{({}-mode)}}'.format(htc2.num_modes))
    axes[0,0].legend()
    axes[0,0].set_title(r'$\langle a_k^\dagger a_k^{\phantom{\dagger}} \rangle(t)$')
    axes[1,0].set_title(r'$\langle a_k^\dagger a_k^{\phantom{\dagger}} \rangle(t_f)$')
    axes[1,0].plot(htc1.Ks, nK1[-1,:].real)
    if plot_ph:
        axes[1,1].set_title(htc1.labels['Eph']+r'$(r_n, t_f)$')
        p = axes[1,1].plot(htc1.rs, nP1[-1,:], label=r'\rm{full}')
        axes[1,1].plot(htc2.rs, nP2[-1,:], ls='--', color=p[-1].get_color(), label=r'$k_R$ \rm{{({}-mode)}}'.format(htc2.num_modes))
    else:
        axes[1,1].set_title(htc1.labels['EnM']+r'$(r_n, t_f)$')
        p = axes[1,1].plot(htc1.rs, nM1[-1,:], label=r'\rm{full}')
        axes[1,1].plot(htc2.rs, nM2[-1,:], ls='--', color=p[-1].get_color(),  label=r'$k_R$ \rm{{({}-mode)}}'.format(htc2.num_modes))
    #axes[0,1].set_axis_off()
    axes[1,1].set_xlabel(htc1.labels['rn'])
    axes[1,1].legend()
    axes[1,0].set_xlabel(r'$K$')
    axes[1,0].axvline(htc1.Ks[QL], color='m', ls='--')
    axes[1,0].axvline(htc1.Ks[QR], color='m', ls='--')
    axes[1,0].annotate(r'${:.2f}$'.format(nK1[-1,QL]), xy=(htc1.Ks[QL],nK1[-1,QL]), xytext=(5,0),
                       textcoords='offset points')
    axes[1,0].annotate(r'${:.2f}$'.format(nK1[-1,QR]), xy=(htc1.Ks[QR],nK1[-1,QR]), xytext=(5,0),
                       textcoords='offset points')
    para_strs = \
            [r'$N_E={}$'.format(htc1.NE),
             r'$\kappa={}$'.format(params['kappa']),
             r'@@@ $t={}$ @@@'.format(params['t']),
             r'$g={}$'.format(params['g']),
             r'$\Gamma_z={}$'.format(params['dephase']),
             r'$\Gamma_\uparrow(L/2)={}$'.format(params['pump_strength']),
             r'$\Gamma_\downarrow={}$'.format(params['decay']),
             #r'$\omega_0={}$'.format(params['omega_0']),
             ]
    axes[0,1].get_xaxis().set_visible(False)
    axes[0,1].get_yaxis().set_visible(False)
    axes[0,1].text(0.5,0.8, '\n'.join(para_strs), ha='center', va='top', transform=axes[0,1].transAxes,
                   size='large')
    fig.savefig('figures/two-mode_comparison.png', bbox_inches='tight', dpi=400)

def mf_test(params):
    assert params['Nnu'] == 1
    htc = HTC(params)
    tf = 100
    results = htc.evolve(tf)
    ada, l, al, ll = htc.split_reshape_return(results['final_state'])
    nk = fftshift(np.diag(ada))
    ak0s = np.sqrt(nk/htc.Nm)
    l0s = l[0]
    lp0s = np.sqrt(ll[0,0][htc.diag_indices_Nk])
    y0 = np.concatenate((ak0s, lp0s, l0s),axis=None)
    htc.evolve_mf(tf, y0=y0)
    fig, axes = plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    axes[0].plot(htc.dynamics['t'], np.sum(htc.dynamics['nP'], axis=1), label=r'\rm{C2}')
    axes[0].plot(htc.dynamics['t'], np.sum(htc.dynamics_mf['nP'], axis=1), label=r'\rm{mf}')
    #axes[0].plot(htc.dynamics['t'], np.sum(htc.dynamics_mf2['nP'], axis=1), label=r'\rm{mf2}', ls='--')
    axes[0].set_title(r'$\sum_n n_{\text{\rm{ph}}}(r_n,t)$')
    axes[0].set_xlabel(r'$t$')
    axes[1].plot(htc.dynamics['t'], np.sum(htc.dynamics['nM'], axis=1), label=r'\rm{C2}')
    axes[1].plot(htc.dynamics['t'], np.sum(htc.dynamics_mf['nM'], axis=1), label=r'\rm{mf}')
    #axes[1].plot(htc.dynamics['t'], np.sum(htc.dynamics_mf2['nM'], axis=1), label=r'\rm{mf2}', ls='--')
    axes[1].set_title(r'$\sum_n n_{\text{\rm{M}}}(r_n,t)$')
    axes[1].set_xlabel(r'$t$')
    axes[0].legend()
    axes[1].legend()
    fig.savefig('figures/mean-field_real_space.png', bbox_inches='tight')

if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    plasmon_parameters = {
            'Q0': 30, # N_k = 2*Q0+1 nanoparticles (123)
            'NE': 4, # Number of emitters per gap
            'w': 1, # Gap width, nm (Emitter spacing Delta_r = 2a+w = 81nm)
            'a': 40, # Nanoparticle radius, nm (Chain length L = N_k * Delta_r = 10.0 nm)
            'omega_p': 1.88, # Plasmon resonance, eV
            'omega_0': 1.86, # Dye resonance, eV
            'g': 0.095, # Individual light-matter coupling, eV (collective gSqrt(NE))
            't': 1.0, # 'Hopping' strength - only used to multiply current plot 
            'kappa': 1e-01, # photon loss
            'dephase': 0.0, # Emitter pure dephasing
            'pump_strength': 1e-01, # Emitter pump magnitude
            'pump_width': 500, # Emitter pump spot width, nm (approx. 6 nanoparticles)
            'decay': 1e-01, # Emitter non-resonant decay
            'Nnu': 1, # Number of vibrational levels for each emitter
            'S': 0.01, # Huang-Rhys parameter
            'omega_nu': 0.19, # Vibrational mode resonance, eV
            'T':0.026, # k_B T in eV (.026=302K)
            'gam_nu': 1e-03, # vibrational damping rate
            'gam_as': 0.0, # Antistokes decay
            'as_coherent': False, # True to turn on g(a . sigma^+ . b^- + H.C.) terms
            'dt': 0.5, # interval at which solution is sampled. Does not affect accuracy of solution 
            'model': 'plasmonic',
            }
    gn = 0.2
    NE = 4
    g = gn/np.sqrt(NE)
    tb_parameters = {
            'Q0': 30, # Chain of Nk = 2*Q0+1 = 51 sites
            'NE': NE, # Number of emitters per gap
            'w': 1, # Gap width, nm (Emitter spacing Delta_r = 2a+w = 81nm) [not used in dynamics calculation]
            'a': 40, # Nanoparticle radius, nm (Chain length L = N_k * Delta_r = 10.0 nm) [not used in dynamics]
            'omega_p': 0.0, # Plasmon resonance, eV [not used in tight-binding model]
            'omega_0': 0.0, # Dye resonance, eV [use to control detuning in tight binding model]
            't': 0.1, # hopping parameter, eV [not used in plasmonic model]
            'g': g, # Individual light-matter coupling, eV, g=0.1/sqrt(NE) 
            'kappa': 0.1, # photon loss
            'dephase': 0.0, # Emitter pure dephasing
            'pump_strength': 0.1, #  Magnitude of pump strength (changed in plot_input_output below)
            #'pump_width': 324, # Pump spot width, nm (4 sites = 4 * Delta r = 4 * 81 = 324nm)
            'pump_width': 229, # Pump spot width, nm, to match JB Aexp{-((n-25)/4)^2} (see def gaussian above)
            'decay': 0.05, # Emitter non-resonant decay
            'Nnu': 1, # Number of vibrational levels for each emitter
            'S': 0.1, # Huang-Rhys parameter [Not relevant if Nnu=1]
            'omega_nu': 0.19, # Vibrational mode resonance, eV [N/A when Nnu=1]
            'T':0.026, # k_B T in eV for vibrational environment (.026=302K) [N/A when Nnu=1]
            'gam_nu': 1e-04, # vibrational damping rate [N/A when Nnu=1]
            'gam_as': 0.0, # Antistokes decay
            'as_coherent': False, # True to turn on g(a . sigma^+ . b^- + H.C.) terms
            'dt': 1.0, # interval at which solution is sampled. Does not affect accuracy of solution 
            'model': 'tight-binding', # dispersion to use 
            #'model': 'two-node', # tight-binding dispersion... but only counting two modes
            }
    #mf_parameters = plasmon_parameters.copy() # a and coherences decay..
    #mf_parameters = tb_parameters.copy() # a diverges)
    #mf_parameters['dt'] = 1.0
    #mf_parameters['Q0'] = 30
    #mf_parameters['NE'] = 100
    #mf_parameters['g'] = 0.2 / np.sqrt(100)
    #mf_test(mf_parameters)
    #sys.exit()
    #
    # parameters for two-node: Q0=30, NE=100, g=0.01, t=0.1, 1.0, 5.0
    #pump_strengths = np.logspace(-3, 0.6, num=20) # set pump strength magnitudes for input-output curve
    #pump_strengths = np.logspace(0, 1, num=6) # set pump strength magnitudes for input-output curve
    #pump_strengths = [100*0.05] # set pump strength magnitudes for input-output curve
    #plot_input_output(tb_parameters.copy(), pump_strengths, tend=100) # all other parameters fixed
    #GD = tb_parameters['decay']
    #pump_strengths = [GD, 5*GD, 10*GD, 20*GD]
    #plot_input_output(tb_parameters.copy(), pump_strengths, tend=100, xlims=[-30,30]) 
    #tb_parameters['pump_strength'] = 0.2
    #plot_dynamics_and_final_state(tb_parameters.copy()) 
    two_mode_comparison(tb_parameters, plot_ph=True)

