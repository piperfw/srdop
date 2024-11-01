#!/usr/bin/env python

import logging, os, sys, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from time import time
import numpy as np
from scipy.integrate import RK45, DOP853
from types import SimpleNamespace
SOLVER = DOP853
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy import constants
from copy import copy
from opt_einsum import contract
import matplotlib.pyplot as plt
from matplotlib import colormaps
try:
    import pretty_traceback
    pretty_traceback.install()
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

class Parameters(SimpleNamespace):
    def __init__(self,
                 omega_0=1.0,
                 omega_c=1.0,
                 Q0=30,
                 NE=4,
                 g=0.01,
                 t=0.1,
                 tau=0.0,
                 kappa=0.1,
                 Gam_z=0.0,
                 Gam_down=0.01,
                 gam_ee=0.01,
                 pump_strength=0.1,
                 pump_width=4,
                 dt=10,
                 ):
        super().__init__(**locals())

class RealHTC:
    EV_TO_FS = (constants.hbar/constants.e)*1e15 # convert time in electronvolts to time in fs
    DEFAULT_DIRS = {'data':'./data', 'figures':'./figures'} # output directories

    def __init__(self, params):
        self.params = params
        self.add_useful_params()
        self.create_slices()
        self.create_pump()
        self.create_coeffs()
        self.create_initial_state()
        return

    def add_useful_params(self):
        params = self.params
        params.gSqrtNE = params.g * np.sqrt(params.NE)
        params.Nk = 2 * params.Q0 + 1
        self.Q0, self.Nk, self.NE = params.Q0, params.Nk, params.NE
        params.gam_E = ( params.NE - 1 ) * params.gam_ee
        self.ns = np.arange(self.Nk)
        self.Ks = self.ns - self.Q0
        self.delta = np.eye(self.Nk)
        self.ee_hop = True if not np.isclose(params.tau, 0.0) else False

    def create_slices(self):
        Nk = self.Nk
        names = ['a_dag_a', 'sig_z', 'a_sig_plus',
                 'sig_plus_sig_minus', 'sig_z_sig_z']
        shapes = [(Nk,Nk), (Nk,), (Nk,Nk), (Nk,Nk)]
        if self.ee_hop:
            shapes.append((Nk, Nk))
        else:
            shapes.append((Nk, )) # only need diagonal elements of sigzsigz
        state_index = 0
        state_dic = {}
        split_list = []
        for name, shape in zip(names, shapes):
            next_index = state_index + np.prod(shape)
            state_dic[name] = {}
            state_dic[name]['shape'] = shape
            state_dic[name]['slice'] = slice(state_index,  next_index)
            split_list.append(next_index)
            state_index = next_index
        split_list.pop()
        self.state_dic = state_dic
        self.split_list = split_list
        self.state_length = next_index

    def create_initial_state(self):
        state = np.zeros(self.state_length, dtype=complex)
        state_dic = self.state_dic
        state[state_dic['sig_z']['slice']] = - 1.0
        if self.ee_hop:
            sigzsigz = np.eye(self.Nk)
            state[state_dic['sig_z_sig_z']['slice']] = sigzsigz.flatten()
        else:
            state[state_dic['sig_z_sig_z']['slice']] = 1.0
        self.initial_state = state

    def gaussian(self, n, _max, _width, _offset=0):
        n0 = self.Q0 + _offset
        return _max * np.exp(- 0.5 * ((n-n0)/_width)**2)
    
    def pump(self, n):
        return self.gaussian(n,
                             self.params.pump_strength,
                             self.params.pump_width)

    def create_pump(self):
        Nk = self.Nk
        self.pumps = self.pump(self.ns)
        self.Gam_T = self.pumps + self.params.Gam_down
        self.Gam_n = self.Gam_T + 4 * self.params.Gam_z
        self.Gam_D = self.pumps - self.params.Gam_down


    def omega(self, K):
        kdr = K * (2*np.pi/self.Nk)
        return self.params.omega_c - 2 * self.params.t * np.cos(kdr)

    def create_coeffs(self):
        params = self.params
        self.asp_coeff = 1j * (params.omega_0 - params.omega_c) \
                - 0.5 * (self.Gam_n + params.kappa + params.gam_E)# omega_c centre of band
        pump_2d = np.broadcast_to(self.Gam_n, (self.Nk, self.Nk))
        #X, Y = np.meshgrid(self.Gam_n, self.Gam_n)
        #tot_pump = X + Y
        self.spsm_coeff = - 0.5 * (pump_2d.T + pump_2d) - params.gam_E
        GamT_n = np.broadcast_to(self.Gam_T, (self.Nk, self.Nk)).T
        GamT_m = GamT_n.T
        self.Gam_T2 = GamT_n + GamT_m # used in szsz equation with ee hopping


    def split_reshape(self, state):
        Nk = self.Nk
        #a_dag_a = state[:Nk**2].reshape(Nk, Nk)                       # <a_n^† a_m>
        #sig_z = state[Nk**2:Nk**2 + Nk]                 # <σ_n^z>, an array of length Nk (.reshape((Nk,)))
        #a_sig_plus = state[Nk**2 + Nk:2*Nk**2 + Nk].reshape(Nk, Nk)   # <a_m σ_n^+>
        #sig_plus_sig_minus = state[2*Nk**2 + Nk:3*Nk**2 + Nk].reshape(Nk, Nk)  # <a_m σ_n^->
        #sig_z_sig_z = state[3*Nk**2 + Nk:]                          # <σ_n^z σ_m^z> (1D array)
        #return a_dagger_a, sig_z, a_sig_plus, sig_plus_sig_minus, sig_z_sig_z
        #a_dagger_a, sig_z, a_sig_plus, sig_plus_sig_minus, sig_z_sig_z =\
        #        np.split(state, self.split_list)
        to_return = []
        for name in self.state_dic:
            to_return.append(state[self.state_dic[name]['slice']].reshape(self.state_dic[name]['shape']))
        return to_return

    def eoms(self, t, state):
        """Real space EoMs"""
        delta_nm = self.delta
        params = self.params
        roll_plus = lambda arr, ax: np.roll(arr, -1, axis=ax)
        roll_minus = lambda arr, ax: np.roll(arr, 1, axis=ax)
        kappa, g, t_hop, NE, gam_E, gam_ee =\
                params.kappa, params.g, params.t, params.NE, params.gam_E, params.gam_ee
        Gam_T, Gam_D, Gam_n = self.Gam_T, self.Gam_D, self.Gam_n
        a_dag_a, sig_z, a_sig_plus, sig_plus_sig_minus, sig_z_sig_z = \
                self.split_reshape(state)
    
        # <a_n^† a_m> 
        da_dag_a = - kappa * a_dag_a \
                       + 1j * g * NE * (a_sig_plus.T - np.conj(a_sig_plus)) \
                       - 1j * t_hop * (roll_plus(a_dag_a, 0) + roll_minus(a_dag_a, 0)
                                       - roll_plus(a_dag_a, 1) - roll_minus(a_dag_a, 1))
    
        # <σ_n^z> 
        dsig_z = -(Gam_n + 2 * gam_E) * sig_z \
                      + (Gam_D - gam_E) \
                      + 4 * g * np.imag(np.diag(a_sig_plus)) \
                      - gam_E * sig_z_sig_z
    
        # <a_m σ_n^+>  # self.asp_coeff - self.create_coeffs
        da_sig_plus = contract('n,mn->mn', self.asp_coeff, a_sig_plus) \
                         - 0.5 * gam_E * contract('mn,n->mn', a_sig_plus, sig_z) \
                          + 1j * t_hop * (roll_minus(a_sig_plus, 0) + roll_plus(a_sig_plus, 0)) \
                          - 1j * g * (contract('n,nm->mn', sig_z, a_dag_a + 0.5 * delta_nm)
                                      + 0.5 * delta_nm + (NE - delta_nm) * np.swapaxes(sig_plus_sig_minus, 0, 1))
    
        # <σ_n^+ σ_m^-> # self.spsm_coeff - self.create_coeffs
        dsig_plus_sig_minus = self.spsm_coeff *  sig_plus_sig_minus \
                                  - 0.5 * gam_E * (contract('n,nm->nm', sig_z, sig_plus_sig_minus) 
                                                     + contract('m,nm->nm', sig_z, sig_plus_sig_minus)) \
                                 + 1j * g * contract('m,mn->nm', sig_z,  a_sig_plus) \
                                 - 1j * g * contract('n,nm->nm', sig_z,  np.conj(a_sig_plus)) \
                                 + gam_ee * delta_nm * sig_plus_sig_minus * (1 + sig_z) # CHECK

        # <σ_n^z σ_m^z>
        dsig_z_sig_z = - 2 * Gam_T * sig_z_sig_z \
                          + 2 * Gam_D * sig_z \
                          + 8 * g * sig_z * np.imag(np.diag(a_sig_plus)) \
                          - 2 * gam_ee * ((NE-2) * (sig_z + 2 * sig_z_sig_z +
                                                   sig_z * (3 * sig_z_sig_z - 2 * sig_z**2)))
                          #- gam_ee * (1 + NE * sig_z + (2 * NE - 3) * sig_z_sig_z 
                          #            + (NE - 2) * sig_z * (3 * sig_z_sig_z - 2 * sig_z**2)
                          #            )
    
        dy = np.concatenate((da_dag_a, [dsig_z], da_sig_plus, dsig_plus_sig_minus, [dsig_z_sig_z]), axis=None)        
        return dy

    def eoms_ee_hop(self, t, state):
        """Real space EoMs with EE hopping"""
        delta_nm = self.delta
        params = self.params
        roll_plus = lambda arr, ax: np.roll(arr, -1, axis=ax)
        roll_minus = lambda arr, ax: np.roll(arr, 1, axis=ax)
        delta_np_nm = roll_plus(delta_nm, 1) + roll_minus(delta_nm, 1)
        kappa, g, t_hop, tau_hop, NE, gam_E, gam_ee =\
                params.kappa, params.g, params.t, params.tau, params.NE, params.gam_E, params.gam_ee
        Gam_T, Gam_D, Gam_n, Gam_T2 = self.Gam_T, self.Gam_D, self.Gam_n, self.Gam_T2
        a_dag_a, sig_z, a_sig_plus, sig_plus_sig_minus, sig_z_sig_z = \
                self.split_reshape(state)

        sig_plus_sig_minus_roll_plus_0 = roll_plus(sig_plus_sig_minus, 0)
        sig_plus_sig_minus_roll_minus_0 = roll_minus(sig_plus_sig_minus, 0)
        sig_plus_sig_minus_roll_plus_1 = roll_plus(sig_plus_sig_minus, 1)
        sig_plus_sig_minus_roll_minus_1 = roll_minus(sig_plus_sig_minus, 1)
    
        # <a_n^† a_m> 
        da_dag_a = - kappa * a_dag_a \
                       + 1j * g * NE * (a_sig_plus.T - np.conj(a_sig_plus)) \
                       - 1j * t_hop * (roll_plus(a_dag_a, 0) + roll_minus(a_dag_a, 0)
                                       - roll_plus(a_dag_a, 1) - roll_minus(a_dag_a, 1))

    
        # <σ_n^z> 
        dsig_z = -(Gam_n + 2 * gam_E) * sig_z \
                      + (Gam_D - gam_E) \
                      + 4 * g * np.imag(np.diag(a_sig_plus)) \
                      - gam_E * np.diag(sig_z_sig_z) \
                      - 4 * tau_hop * NE * np.imag(
                              np.diag(sig_plus_sig_minus_roll_plus_1)
                              +
                              np.diag(sig_plus_sig_minus_roll_minus_1))
    
        # <a_m σ_n^+>  # self.asp_coeff - self.create_coeffs
        da_sig_plus = contract('n,mn->mn', self.asp_coeff, a_sig_plus) \
                         - 0.5 * gam_E * contract('mn,n->mn', a_sig_plus, sig_z) \
                          + 1j * t_hop * (roll_minus(a_sig_plus, 0) + roll_plus(a_sig_plus, 0)) \
                          - 1j * g * (contract('n,nm->mn', sig_z, a_dag_a + 0.5 * delta_nm)
                                      + 0.5 * delta_nm + (NE - delta_nm) * np.swapaxes(sig_plus_sig_minus, 0, 1)) \
                          + 1j * tau_hop * NE * contract('n,mn->mn', sig_z,
                                                         (roll_plus(a_sig_plus, 1) + roll_minus(a_sig_plus, 1)))
    
        # <σ_n^+ σ_m^-> # self.spsm_coeff - self.create_coeffs
        dsig_plus_sig_minus = self.spsm_coeff *  sig_plus_sig_minus \
                                  - 0.5 * gam_E * (contract('n,nm->nm', sig_z, sig_plus_sig_minus) 
                                                     + contract('m,nm->nm', sig_z, sig_plus_sig_minus)) \
                                 + 1j * g * contract('m,mn->nm', sig_z,  a_sig_plus) \
                                 - 1j * g * contract('n,nm->nm', sig_z,  np.conj(a_sig_plus)) \
                                 + gam_ee * delta_nm * sig_plus_sig_minus * (1 + sig_z) \
                                 + 1j * tau_hop * NE * contract(
                                         'n,nm->nm', sig_z, sig_plus_sig_minus_roll_plus_0 
                                                             + sig_plus_sig_minus_roll_minus_0) \
                                 - 1j * tau_hop * NE * contract(
                                         'm,nm->nm', sig_z, sig_plus_sig_minus_roll_plus_1 
                                                             + sig_plus_sig_minus_roll_minus_1) \
                                + 1j * tau_hop * contract('nm,n,mm->nm', delta_np_nm,
                                                               sig_z, 0.5 - sig_plus_sig_minus) \
                                - 1j * tau_hop * contract('nm,m,nn->nm', delta_np_nm,
                                                               sig_z, 0.5 - sig_plus_sig_minus) 

        # <σ_n^z σ_m^z>
        a_sig_plus_im_diag = np.imag(np.diag(a_sig_plus))
        sz_n2, sz_m2 = np.meshgrid(sig_z, sig_z, indexing='ij') # or broadcast_to...
        sz_sum = sz_n2 + sz_m2
        sz_diff = sz_n2 - sz_m2
        spsm_sz_diff = contract('nm,nm->nm', sig_plus_sig_minus, sz_diff)
        sig_z_sig_z_diag = np.diag(sig_z_sig_z) # useful below. Note in delta term could instead contract delta will full sig_z_sig_z matrix (possibly more efficient)
                #+ 4 * g * contract('n,mm->nm', sig_z, np.imag(a_sig_plus)) \
        dsig_z_sig_z = - contract('nm,nm->nm', Gam_T2, sig_z_sig_z) \
                + contract('n,m->nm', Gam_D, sig_z) + contract('m,n->nm', Gam_D, sig_z) \
                + 4 * g * contract('n,m->nm', sig_z, a_sig_plus_im_diag) \
                + 4 * g * contract('m,n->nm', sig_z, a_sig_plus_im_diag) \
                - gam_E * (contract('n,mm->nm', sig_z, sig_z_sig_z) + contract('m,nn->nm', sig_z, sig_z_sig_z) 
                           + 4 * sig_z_sig_z + sz_sum + 2 * contract('nm,nm->nm', sz_sum, sig_z_sig_z)
                           - 2 * contract('nm,n,m->nm', sz_sum, sig_z, sig_z)) \
                + 2 * gam_ee * delta_nm * (2 * sig_z_sig_z_diag + sig_z * (1 + 3 * sig_z_sig_z_diag - 2 * sig_z**2)) \
                - 4 * tau_hop * NE * np.imag(
                        contract('n,mm->nm', sig_z, sig_plus_sig_minus_roll_minus_1 - sig_plus_sig_minus_roll_plus_0)
                        + contract('m,nn->nm', sig_z, sig_plus_sig_minus_roll_minus_1 - sig_plus_sig_minus_roll_plus_0) ) \
                + 4 * tau_hop * contract('nm,nm->nm', delta_np_nm, np.imag(spsm_sz_diff))

        dy = np.concatenate((da_dag_a, [dsig_z], da_sig_plus, dsig_plus_sig_minus, dsig_z_sig_z), axis=None)        
        return dy

    def evolve(self, tend=250.0, atol=1e-8, rtol=1e-6):
        """Integrate second-order cumulants equations of motion from t=0 to  t=tend (femptoseconds)"""
        params = self.params
        dt_fs = params.dt
        self.t_fs = np.arange(0.0, tend+dt_fs/2, step=dt_fs)
        self.t = self.t_fs / self.EV_TO_FS
        self.num_t = len(self.t)
        dt = dt_fs / self.EV_TO_FS
        self.setup_dynamics_storage() # creates self.dynamics data dictionary
        #
        t_index = 0 # indicates current position in output grid of times
        num_checkpoints = 5 # checkpoints at 0, 25%,...
        checkpoint_spacing = int(round(self.num_t/num_checkpoints))
        checkpoints = np.linspace(0, self.num_t-1, num=num_checkpoints, dtype=int)
        next_check_i = 1
        last_solver_i = 0
        solver_t = [] # keep track of solver times too (not fixed grid)
        eoms = self.eoms_ee_hop if self.ee_hop else self.eoms
        logger.info(f'Evolving {eoms.__doc__} to tend={tend} fs at pump_strength={params.pump_strength:.2f}')
        tic = time() # time the computation
        solver = SOLVER(eoms,
                        t0=0.0,
                        y0=self.initial_state,
                        t_bound=self.t[-1],
                        atol=atol,
                        rtol=rtol,
                    )
        assert solver.t == self.t[t_index], 'Solver initial time incorrect'
        self.record_dynamics(t_index, solver.y) # record physical dynamics for initial state
        solver_t.append(solver.t) # record initial time t=0
        t_index += 1
        next_t = self.t[t_index]
        while solver.status == 'running':
            end = False # flag to break integration loop
            step_message = solver.step() # perform one step (necessary before call to dense_output())
            solver_t.append(solver.t)
            if solver.t >= next_t: # solver has gone past one (or more) of our grid points; evaluate solution
                soln = solver.dense_output() # interpolation function for the last timestep
                while solver.t >= next_t: # until soln has been evaluated at all grid points up to solver time
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
                solver_dt_fs = np.mean(solver_diffs) * self.EV_TO_FS
                if not np.isclose(solver_dt_fs, dt_fs, atol=0.0, rtol=1.0):
                    if solver_dt_fs < dt_fs:
                        logger.warning('Average solver step size {:.2g}fs is far smaller'\
                            ' than target grid spacing {}fs. Consider decreasing parameter dt.'.format(
                                solver_dt_fs, dt_fs))
                next_check_i += 1
                last_solver_i = len(solver_t) - 1
            if end:
                break # safety, stop solver if we have already calculated state at self.t[-1]
        toc = time()
        self.compute_time = toc-tic # ptoc-ptic
        if solver.status == 'failed':
            logger.warning(f'Solver failed at t={solver.t:.1f} with message "{step_message}"')
        logger.info('Done ({:.0f}s)'.format(self.compute_time))
        self.results = {'parameters': self.params,
                        'dynamics': self.dynamics,
                        'final_state': y, # save entire final state
                        }
        return self.results


    def setup_dynamics_storage(self):
        """Prepare dictionary self.dynamics to store values of relevant dynamics
        These arrays (or arrays in dictionaries) are zero initialised and then assigned
        non-zero values in place by self.record_dynamics during the computation
        """
        Nt = self.num_t
        nPs = np.zeros((Nt, self.Nk), dtype=float)
        nKs = np.zeros((Nt, self.Nk), dtype=float)
        nMs = np.zeros((Nt, self.Nk), dtype=float)
        g1s = np.zeros((Nt, self.Nk), dtype=complex)
        g1RRs = np.zeros((Nt, self.Q0+1), dtype=complex)
        Vs = np.zeros((Nt, self.Q0+1), dtype=float)
        self.dynamics = {'t': self.t_fs,
                         'nP': nPs,
                         'nK': nKs,
                         'nM': nMs,
                         'g1': g1s,
                         'g1RR': g1RRs,
                         'V': Vs, 
                         }

    def record_dynamics(self, t_index, y):
        """Calculates and saves observable values from state y at timestep t_index
        To add additional dynamics, add a key-empty array to self.dynamics e.g.
        self.dynamics['my_obs'] in self.setup_storage_dynamics and then write a
        function to take state, calculate value of observable and assign to
        self.dynamics['my_obs'][t_index]
        """
        # This is only copy of entire state we make. Only absolutely required if
        # modifying the state (e.g. rescale)
        state = y.copy()
        a_dag_a, sig_z, a_sig_plus, sig_plus_sig_minus, sig_z_sig_z = self.split_reshape(state)
        # The following directly update the instance variable self.dynamics 
        #self.calculate_photonic(t_index, a_dag_a) # Photon exciton densities
        nPh = np.diag(a_dag_a)
        self.check_real(nPh, t_index, 'photon number')
        self.dynamics['nP'][t_index] = np.real(nPh)
        nkp = fftshift(fft(ifft(a_dag_a, axis=0), axis=1))
        nkk = np.diag(nkp)
        self.check_real(nkk, t_index, 'photon number (k-space)')
        self.dynamics['nK'][t_index] = np.real(nkk)
        nM = self.NE * 0.5 * (sig_z + 1)
        self.check_real(nM, t_index, 'electronic population')
        self.dynamics['nM'][t_index] = np.real(nM)
        if np.allclose(nPh, 0.0):
            g1 = np.zeros(self.Nk, dtype=complex)
        else:
            g1 = a_dag_a[:,self.Q0]/np.sqrt(nPh * a_dag_a[self.Q0,self.Q0])
        self.dynamics['g1'][t_index] = g1


    def calculate_photonic(self, t_index, ada):
        nk = fftshift(np.diag(ada))
        self.check_real(t_index, nk, 'Photon number (k-space)')
        alpha = ifft(ada, axis=0) # including 1/N_k normalisation!
        dft2 = fft(alpha, axis=-1) # real space so no fftshift... (start with position 0...)
        nph = np.diag(dft2) # n(r_n) when n=m
        self.check_real(t_index, nph, 'Photon density')
        mid_n = self.Q0
        g1 = np.zeros(self.Nk, dtype=complex)
        for n in self.ns:
            numer = dft2[n, mid_n]
            demon = np.sqrt(np.abs(np.real(dft2[n,n]) * np.real(dft2[mid_n, mid_n])))
            if np.isclose(demon, 0.0, atol=1e-8):
                g1[n] = np.zeros_like(numer)
            else:
                g1[n] = numer/demon
        # 2024-08-16 Calculate g^(1)(R,-R) for R=0,1,...,Q0 (R=0 meaning the centre)
        g1RR = np.zeros(self.Q0+1, dtype=complex)
        for n in range(self.Q0+1):
            numer = dft2[mid_n - n, mid_n + n]
            denom = np.sqrt(np.abs(np.real(dft2[mid_n - n, mid_n - n]) * np.real(dft2[mid_n + n, mid_n + n])))
            if not np.isclose(demon, 0.0, atol=1e-8):
                g1RR[n] = numer / denom
        # Visibility
        V = np.zeros(self.Q0+1, dtype=float)
        for n in range(self.Q0+1):
            numerV = 2 * np.abs(dft2[self.Q0+n, self.Q0-n])
            denomV = np.real(dft2[self.Q0+n,self.Q0+n]) +  np.real(dft2[self.Q0-n,self.Q0-n]) 
            if np.isclose(denomV, 0.0, atol=1e-8):
                V[n] = 0.0
            else:
                V[n] = numerV/denomV
        self.dynamics['nP'][t_index] = np.real(nph)
        self.dynamics['nK'][t_index] = np.real(nk)
        self.dynamics['g1'][t_index] = g1
        self.dynamics['g1RR'][t_index] = g1RR
        self.dynamics['V'][t_index] = V

    WARN_REAL = {}
    def check_real(self, step, arr, name):
        if name not in self.WARN_REAL:
            self.WARN_REAL[name] = True
        if not self.WARN_REAL[name]:
            return
        if not np.allclose(np.imag(arr), 0.0, atol=1e-6):
            t = self.t[step]
            logger.warning(f'{name} at t={t} has non-zero imaginary part (further warnings suppressed)')
            self.WARN_REAL[name] = False

    def plot_dispersion_pump(self):
        fig, axes = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
        axes[0].set_xlabel(r'$K$')
        axes[0].set_title(r'$\hbar\omega_K$' + r' $(\rm{eV})$')
        axes[1].set_title(r'$\Gamma_\uparrow(r_n)\ (\sigma={}$'.format(
            params.pump_width)+r'$\rm{nm})$')
        all_Ks = np.linspace(-self.Q0, self.Q0, 250)
        all_y = self.omega(all_Ks)
        all_ns = np.linspace(0, self.Nk, 250)
        all_pumps = self.pump(all_ns)
        select_pumps = self.pump(self.ns)
        axes[0].plot(all_Ks, all_y)
        Q0=self.Q0
        Nk=self.Nk
        ticks = [-(Nk/2), -(Nk/4), 0, (Nk/4), (Nk/2)]
        tick_labels = [r'$-\pi/\Delta r$', r'$-\pi/(2\Delta r)$',r'$0$', r'$\pi/2(\Delta r) $',r'$\pi/\Delta r $']
        axes[0].set_xlim([-(Nk/2), (Nk/2)])
        axes[0].set_xticks(ticks)
        axes[0].set_xticklabels(tick_labels)
        axes[0].axhline(params.omega_0, c='r', label=r'$\omega_0$')
        axes[0].legend()
        axes[1].plot(all_ns, all_pumps)
        axes[1].scatter(self.ns, select_pumps, c='r', s=8, zorder=2)
        fp = os.path.join(self.DEFAULT_DIRS['figures'], 'real_space_dispersion_pump.png')
        fig.savefig(fp, bbox_inches='tight', dpi=350)
        plt.close(fig)

    def cauchy_mask(self, ada):
        delta = 1e-8
        ada = fftshift(ada)
        diags = np.diag(ada)
        Nk = len(diags)
        ada_p = np.array([x * np.ones(Nk) for x in diags])
        ada_k = ada_p.T
        diff =   ada_p * ada_k - np.abs(ada)**2
        mask = diff < - delta # delta for numerical tolerance
        return mask, diff 

def plot_dynamics(params, tend=250):
    htc = RealHTC(params)
    results = htc.evolve(tend=tend)
    nph_tots = np.sum(results['dynamics']['nP'], axis=1)
    nM_tots = np.sum(results['dynamics']['nM'], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(8,6), constrained_layout=True, sharex=False)
    axes[0].plot(htc.t_fs, nph_tots)
    axes[1].plot(htc.t_fs, nM_tots/params.NE)
    fig.savefig('figures/real_space_dynamics.png', bbox_inches='tight', dpi=350)
    plt.close(fig)

def plot_input_output(params, pump_strngths, tend=250,
                      normalise=False, 
                      max_nph_curves=5,
                      xlims=None):
    num_pumps = len(pump_strengths)
    Nk = 2 * params.Q0 + 1
    ratios = np.array(pump_strengths) / params.Gam_down
    nph_final = np.zeros((num_pumps, Nk), dtype=float)
    nK_final = np.zeros((num_pumps, Nk), dtype=float)
    nM_final = np.zeros((num_pumps, Nk), dtype=float)
    nK_final = np.zeros((num_pumps, Nk), dtype=float)
    g1_final = np.zeros((num_pumps, Nk), dtype=complex)
    adaga_final = np.zeros((num_pumps, Nk, Nk), dtype=complex) 
    adaga_final_mask = np.zeros((num_pumps, Nk, Nk), dtype=bool) 
    fig, axes = plt.subplots(3, 2, figsize=(8,10), constrained_layout=True, sharex='col')
    figk, axesk = plt.subplots(1,2, figsize=(8,3), constrained_layout=True)
    select_indices = np.round(np.linspace(0, num_pumps-1, max_nph_curves)).astype(int)
    pump_title = r'$\Gamma_\uparrow(0)/\Gamma_\downarrow$'
    for i, pump in enumerate(pump_strengths):
        logger.info(f'On pump {i+1} of {num_pumps}')
        params.pump_strength = pump
        #if i == 0:
        #    params.gam_ee = 0.0
        #else:
        #    params.gam_ee = 0.1
        htc = RealHTC(params)
        results = htc.evolve(tend=tend)
        nph_final[i, :] = results['dynamics']['nP'][-1, :]
        nK_final[i, :] = results['dynamics']['nK'][-1, :] # already fftshifted to ascending order
        nM_final[i, :] = results['dynamics']['nM'][-1, :]
        g1_final[i, :] = results['dynamics']['g1'][-1, :]
        if i not in select_indices:
            continue
        if normalise:
            y1 = nph_final[i, :]/nph_final[i,:][htc.Q0]
        else:
            y1 = nph_final[i, :]
        y2 = nM_final[i, :]/params.NE
        y3 = nK_final[i, :]
        pump_str = r'${:.2g}$'.format(round(ratios[i],5))
        axes[0,1].plot(y1[htc.Q0:], label=pump_str)
        axes[1,1].plot(y2[htc.Q0:], label=pump_str)
        #from scipy.signal import argrelmax
        #print(argrelmax(np.abs(g1_final[i,htc.Q0:]), mode='wrap'))
        #print(argrelmax(-np.abs(g1_final[i,htc.Q0:]), mode='wrap'))
        axes[2,1].plot(np.abs(g1_final[i,htc.Q0:]), label=pump_str)
        axesk[0].plot(htc.Ks, y3, label=pump_str)
        if i == num_pumps - 1:
            Nk = htc.Nk
            final_ada = results['final_state'][htc.state_dic['a_dag_a']['slice']].reshape((Nk, Nk))
            nkp = fftshift(fft(ifft(final_ada, axis=0), axis=1))
            mask, diff = htc.cauchy_mask(nkp)
            adaga_one = np.ma.masked_array(np.copy(nkp),
                                           mask=mask)
            cm = colormaps['viridis'] 
            cm.set_bad('red')
            extent = [htc.Ks[0], htc.Ks[-1],htc.Ks[0], htc.Ks[-1]]
            im = axesk[1].imshow(np.real(adaga_one), origin='lower', aspect='auto',
                            interpolation='none', extent=extent, cmap=cm,
                            label=r'${:.2g}$'.format(round(ratios[i],5)))
            cbar = figk.colorbar(im, ax=axesk[1], aspect=20)
            axesk[1].set_title(r'$\rm{Re}\,n_{kp}\quad($' + pump_title + r'$=$'+pump_str+r'$)$')
    htc.plot_dispersion_pump()
    if xlims is not None:
        axes[0,1].set_xlim(xlims)
        axes[1,1].set_xlim(xlims)
        axes[2,1].set_xlim(xlims)
    nph_tots = np.sum(nph_final, axis=1) # Sum over all lattice positions 
    nM_tots = np.sum(nM_final, axis=1) # sum over all lattice positions
    axes[2,0].set_xlabel(pump_title)
    axes[2,1].set_xlabel(r'$n$')
    axesk[0].set_xlabel(r'$K$')
    axesk[1].set_xlabel(r'$K$')
    axesk[1].set_ylabel(r'$K$')
    axesk[0].set_title(r'$n_{kk}$')
    #axes[0,1].set_title(r'$n_{\rm{ph}}(r_n)$')
    #axes[0,0].set_title(r'$\sum_n n_{\rm{ph}}(r_n)$')
    axes[0,1].set_title(r'$n_{nn}$')
    axes[0,0].set_title(r'$\sum_n n_{nn}$')
    axes[1,0].set_title(r'$ \sum_n\left(N_Ep^\uparrow_n\right)$')
    axes[1,1].set_title(r'$p^\uparrow_n$')
    axes[2,1].set_title(r'$|g^{(1)}(R)|$')
    axes[2,0].set_title(r'$|g^{(1)}(0)|$')
    axes[0,0].loglog(ratios, nph_tots)
    axes[1,0].loglog(ratios, nM_tots)
    axes[2,0].plot(ratios, np.abs(g1_final[:,htc.Q0]))
    axes[2,0].set_xscale('log')
    axes[1,1].legend(title=pump_title)
    axes[0,1].legend(title=pump_title)
    axesk[0].legend(title=pump_title)
    fig.suptitle(r'$N_k={Nk}\ N_E={NE}\ g={g}\ \kappa={kappa}\ \Gamma^\downarrow={Gam_down:.2g}\  t={t}\ \gamma^{{\rm{{ee}}}}={gam_ee}$'.format(**params.__dict__))
    fig.savefig('figures/real_space_input_output.png', bbox_inches='tight', dpi=350)
    figk.savefig('figures/real_space_cauchy.png', bbox_inches='tight', dpi=350)
    plt.close(fig)
    plt.close(figk)

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        #format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    params = Parameters(omega_0=1.0, # zero-phonon line
                        omega_c=1.0, # MIDDLE of tight-binding dispersion
                        Q0=30, # 2*Q0+1 sites (so Q0 to the right of 0)
                        NE=100, # Number of emitters per gap
                        g=0.01, # INDIVIDUAL light-matter coupling (collective gSqrtNE)
                        t=0.4, # Hopping parameter (photon)
                        #tau=0.0, # Hopping parameter (exciton)
                        tau=1e-4, # TESTING
                        kappa=0.1, # photon loss
                        Gam_z=0.0, # emitter pure dephasing
                        Gam_down=1e-4, # emitter decay
                        gam_ee=1e-4, # emitter EEA rate
                        pump_strength=0.1, # emitter pump strength (maximum of Gaussian), overwritten in plot_input_output below
                        pump_width=4, # Pump width (Gaussian s.d.) in number of SITES
                        )
    #plot_dynamics(params) # total photon number and molecular population vs time (check convergence)
    #min_dec, max_dec = 0, 2
    #ratios = np.logspace(min_dec, max_dec, num=max_dec-min_dec+1)
    #pump_strengths = ratios * params.Gam_down
    #pump_strengths = params.Gam_down * np.logspace(0.5, 1.6, num=5) # gam_ee = 0.0
    pump_strengths = params.Gam_down * np.logspace(1, 3, num=5) # gam_ee = 1e-4
    plot_input_output(params, pump_strengths,
                      normalise=True, # optional, normalise photon population by the population at R=0
                      max_nph_curves=5, # optional, only plot this many curves (if pump_strengths contains more)
                      xlims=[None, 15], # optional, x limits (number of sites) for nph and molecular probability plots
                      )

