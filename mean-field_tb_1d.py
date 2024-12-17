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
from scipy.ndimage import gaussian_filter1d
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
        self.create_initial_state()
        return

    def add_useful_params(self):
        params = self.params
        params.gSqrtNE = params.g * np.sqrt(params.NE)
        params.Nk = 2 * params.Q0 + 1
        params.Nm = params.Nk * params.NE
        self.Q0, self.Nk, self.NE = params.Q0, params.Nk, params.NE
        params.gam_E = ( params.NE - 1 ) * params.gam_ee
        self.ns = np.arange(self.Nk)
        self.Ks = self.ns - self.Q0
        self.delta = np.eye(self.Nk)

    def create_slices(self):
        Nk = self.Nk
        names = ['a', 'sig_minus',  'sig_z']
        state_index = 0
        slice_dic = {}
        split_list = []
        for name in names:
            next_index = state_index + self.Nk
            slice_dic[name] = slice(state_index,  next_index)
            split_list.append(next_index)
            state_index = next_index
        split_list.pop()
        self.slice_dic = slice_dic
        self.split_list = split_list
        self.state_length = next_index

    def create_initial_state(self):
        state = np.zeros(self.state_length, dtype=complex)
        state[self.slice_dic['a']] = 0.01 #1/self.params.gSqrtNE
        state[self.slice_dic['sig_z']] = - 1.0
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

    def eoms(self, t, state):
        """Mean-field EoMs in real space"""
        a, sig_minus, sig_z = np.split(state, self.split_list)
        params = self.params

        da = - 0.5 * params.kappa * a \
             - 1j * params.gSqrtNE * sig_minus \
             + 1j * params.t * (np.roll(a, 1) + np.roll(a, -1))

        dsig_minus = -(1j * params.omega_0 + 0.5 * (self.Gam_n + params.gam_E)) * sig_minus \
                     + 1j * params.gSqrtNE * sig_z * a \
                     - 0.5 * params.gam_E * sig_z * sig_minus

        dsig_z = -(self.Gam_T + 2 * params.gam_E) * sig_z \
                 +(self.Gam_D - params.gam_E) \
                 - 4 * params.gSqrtNE * np.imag(sig_minus * np.conj(a)) \
                 - params.gam_E * sig_z**2
        
        return np.concatenate((da, dsig_minus, dsig_z), axis=None)

    def evolve(self, tend=250.0, atol=1e-8, rtol=1e-6):
        """Integrate equations of motion from t=0 to  t=tend (femptoseconds)"""
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
        logger.info(f'Evolving {self.eoms.__doc__} to tend={tend} fs at pump_strength={params.pump_strength:.2f}')
        tic = time() # time the computation
        solver = SOLVER(self.eoms,
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
        a_n = np.zeros((Nt, self.Nk), dtype=complex)
        sig_minus_n = np.zeros((Nt, self.Nk), dtype=complex)
        sig_z_n = np.zeros((Nt, self.Nk), dtype=float)
        #g1s = np.zeros((Nt, self.Nk), dtype=complex)
        #ZQns = np.zeros((Nt, self.Nk), dtype=float)
        #ZQs = np.zeros(Nt, dtype=float)
        self.dynamics = {'t': self.t_fs,
                         'a_n': a_n,
                         'sig_minus_n': sig_minus_n,
                         'sig_z_n': sig_z_n,
                         }

    def record_dynamics(self, t_index, y):
        """Calculates and saves observable values from state y at timestep t_index
        To add additional dynamics, add a key-empty array to self.dynamics e.g.
        self.dynamics['my_obs'] in self.setup_storage_dynamics and then write a
        function to take state, calculate value of observable and assign to
        self.dynamics['my_obs'][t_index]
        """
        a, sig_minus, sig_z = np.split(y, self.split_list)
        self.dynamics['a_n'][t_index] = a
        self.dynamics['sig_minus_n'][t_index] = sig_minus
        self.dynamics['sig_z_n'][t_index] = np.real(sig_z)

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


def plot_dynamics(params, tend=250):
    htc = RealHTC(params)
    htc.plot_dispersion_pump()
    results = htc.evolve(tend=tend)
    samples = 6
    #sample_i = np.linspace(htc.Q0//2, 3*htc.Q0//2, samples, dtype=int)
    sample_i = np.linspace(0, htc.Q0, samples, dtype=int)
    a_ns = results['dynamics']['a_n']
    sig_minus_ns = results['dynamics']['sig_minus_n']
    sig_z_ns = results['dynamics']['sig_z_n']
    ts = results['dynamics']['t']
    fig, axes = plt.subplots(2, 2, figsize=(8,8), constrained_layout=True, sharex=True)
    axes[1,0].set_xlabel(r'$t$ (fs)')
    axes[0,0].set_title(r'$|\langle a_n\rangle|/\sqrt{N_E}$')
    axes[0,1].set_title(r'Arg$\langle a_n\rangle$')
    axes[1,0].set_title(r'$|\langle \sigma^-_n\rangle|$')
    axes[1,1].set_xlabel(r'$t$ (fs)')
    axes[1,1].set_title(r'$\langle \sigma^z_n\rangle$')
    for i in sample_i:
        a_n = a_ns[:, i]
        sig_minus_n = sig_minus_ns[:, i]
        sig_z_n = sig_z_ns[:, i]
        axes[0,0].plot(ts, np.abs(a_n), label=r'${}$'.format(htc.ns[i]))
        axes[0,1].plot(ts, np.angle(a_n), label=r'${}$'.format(htc.ns[i]))
        axes[1,0].plot(ts, np.abs(sig_minus_n), label=r'${}$'.format(htc.ns[i]))
        axes[1,1].plot(ts, sig_z_n, label=r'${}$'.format(htc.ns[i]))
    axes[0,0].legend(title='Site #')
    fig.savefig('figures/mean-field_photon_dynamics.png', bbox_inches='tight', dpi=350)
    plt.close(fig)

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    ################################
    params = Parameters(omega_0=0.0, # zero-phonon line
                        omega_c=0.0, # MIDDLE of tight-binding dispersion
                        Q0=20, # 2*Q0+1 sites (so Q0 to the right of 0)
                        NE=100, # Number of emitters per gap
                        g=0.01, # INDIVIDUAL light-matter coupling (collective gSqrtNE)
                        t=0.1, # Hopping parameter
                        kappa=0.01, # photon loss
                        Gam_z=0.0, # emitter pure dephasing
                        Gam_down=0.0001, # emitter decay
                        #gam_ee=0.0001, # emitter EEA rate
                        gam_ee=0.0,
                        pump_strength=0.001, # emitter pump strength (maximum of Gaussian), overwritten in plot_input_output below
                        pump_width=2, # Pump width (Gaussian s.d.) in number of SITES
                        dt=0.1,

                        )

    pump_strengths = params.Gam_down * np.logspace(0.1, 3, num=25)
    #params.pump_strength = pump_strengths[-2]
    #plot_waterfalls(params, tend=200, num=6)
    #params.pump_strength = pump_strengths[-1]
    plot_dynamics(params, tend=1000)
    #plot_waterfalls(params, tend=100, num=8)
    #plot_input_output(params, pump_strengths,
    #                  normalise=True, # optional, normalise photon population by the population at R=0
    #                  max_nph_curves=5, # optional, only plot this many curves (if pump_strengths contains more)
    #                  xlims=[-0.5, 15], # optional, x limits (number of sites) for nph and molecular probability plots
    #                  )

###
