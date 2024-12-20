#!/usr/bin/env python

import logging, os, sys, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from time import time
import numpy as np
from scipy.integrate import RK45, DOP853, solve_ivp
from types import SimpleNamespace
SOLVER = DOP853
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy import constants
from copy import copy
from opt_einsum import contract
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import pretty_traceback
pretty_traceback.install()

logger = logging.getLogger(__name__)


# SNS
sns.set_theme(context='notebook', style='ticks', palette='colorblind6', # 'colorblind' if need more than 6 lines
              rc={'legend.fancybox':False,
                  #'text.usetex':True,
                  #'text.latex.preamble':r'\usepackage{amsmath}',
                  'figure.dpi':400.0, 
                  'figure.figsize': [6,4],
                  'figure.constrained_layout.use': True,
                  'savefig.bbox': 'tight',
                  'legend.edgecolor':'0.0', # '0.0' for opaque, '1.0' for transparent
                  'legend.borderpad':'0.2',
                  'legend.fontsize':'9',
                  }
              )

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
        self.nxs, self.nys = np.meshgrid(np.arange(self.Nk), np.arange(self.Nk), indexing='ij')
        self.Kxs, self.Kys = self.nxs - self.Q0, self.nys - self.Q0
        self.delta = np.eye(self.Nk)
    def create_slices(self):
        Nk = self.Nk
        names = ['a', 'sig_minus',  'sig_z']
        state_index = 0
        slice_dic = {}
        split_list = []
        for name in names:
            next_index = state_index + self.Nk**2
            slice_dic[name] = slice(state_index,  next_index)
            split_list.append(next_index)
            state_index = next_index
        split_list.pop()
        self.slice_dic = slice_dic
        self.split_list = split_list
        self.state_length = next_index

    def create_initial_state(self):
        state = np.zeros(self.state_length, dtype=complex)
        state[self.slice_dic['a']] = 0.1/np.sqrt(self.NE)
        state[self.slice_dic['sig_z']] = - 1.0
        self.initial_state = state
    def gaussian(self, nx, ny, _max, _width, _offset=0):
        n0 = self.Q0 + _offset
        return _max * np.exp(- 0.5 * ((nx-n0)/_width)**2 - 0.5 * ((ny-n0)/_width)**2)
    
    def pump(self, n_x, n_y):
        return self.gaussian(n_x, n_y,
                             self.params.pump_strength,
                             self.params.pump_width)
    def create_pump(self):
        Nk = self.Nk
        self.pumps = self.pump(self.nxs, self.nys)
        self.Gam_T = self.pumps + self.params.Gam_down
        self.Gam_n = self.Gam_T + 4 * self.params.Gam_z
        self.Gam_D = self.pumps - self.params.Gam_down

    def omega(self, K_x, K_y):
        kxdr = K_x * (2*np.pi/self.Nk)
        kydr = K_y * (2*np.pi/self.Nk)
        return self.params.omega_c - 2 * self.params.t * (np.cos(kxdr) + np.cos(kydr))

    def split_reshape(self, state):
        #a, sig_minus, sig_z = np.split(state, self.split_list)
        return [x.reshape((self.Nk, self.Nk)) for x in np.split(state, self.split_list)]

    def eoms(self, t, state):
        """Mean-field EoMs in real space"""
        params = self.params
        a_nm, s_nm, z_nm = self.split_reshape(state)

        da_nm = - (1j * params.omega_c + 0.5 * params.kappa) * a_nm \
                - 1j * params.gSqrtNE * s_nm \
                + 1j * params.t * (np.roll(a_nm, 1, axis=0) + np.roll(a_nm, -1, axis=0)
                                   + np.roll(a_nm, 1, axis=1) + np.roll(a_nm, -1, axis=1))

        ds_nm = -(1j * params.omega_0 + 0.5 * (self.Gam_n + params.gam_E)) * s_nm \
                + 1j * params.gSqrtNE * z_nm * a_nm \
                - 0.5 * params.gam_E * z_nm * s_nm

        dz_nm = -(self.Gam_T + 2 * params.gam_E) * z_nm \
                 +(self.Gam_D - params.gam_E) \
                 - 4 * params.gSqrtNE * np.imag(s_nm * np.conj(a_nm)) \
                 - params.gam_E * s_nm**2
        
        return np.concatenate((da_nm, ds_nm, dz_nm), axis=None)

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
        logger.info(f'Evolving {self.eoms.__doc__} to tend={tend} fs at pump_strength={params.pump_strength:.2g}')
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
        a_nm = np.zeros((Nt, self.Nk, self.Nk), dtype=complex)
        s_nm = np.zeros((Nt, self.Nk, self.Nk), dtype=complex)
        z_nm = np.zeros((Nt, self.Nk, self.Nk), dtype=float)
        self.dynamics = {'t': self.t_fs,
                         'a_nm': a_nm,
                         's_nm': s_nm,
                         'z_nm': z_nm,
                         }

    def record_dynamics(self, t_index, y):
        """Calculates and saves observable values from state y at timestep t_index
        To add additional dynamics, add a key-empty array to self.dynamics e.g.
        self.dynamics['my_obs'] in self.setup_storage_dynamics and then write a
        function to take state, calculate value of observable and assign to
        self.dynamics['my_obs'][t_index]
        """
        a_nm, s_nm, z_nm = self.split_reshape(y)
        self.dynamics['a_nm'][t_index] = a_nm
        self.dynamics['s_nm'][t_index] = s_nm
        self.dynamics['z_nm'][t_index] = np.real(z_nm)

    def plot_dispersion_pump(self):
        fig, axes = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)
        axes[0].set_xlabel(r'$K$')
        axes[0].set_title(r'$\hbar\omega_K$' + r' $(\rm{eV})$')
        axes[1].set_title(r'$\Gamma_\uparrow(r_n)\ (\sigma={}$'.format(
            params.pump_width)+r'$\rm{nm})$')
        all_Kxs, all_Kys = np.meshgrid(np.linspace(-self.Q0, self.Q0, 250), np.linspace(-self.Q0, self.Q0,250), indexing='ij')
        all_y = self.omega(all_Kxs, all_Kys)
        all_pumps = self.pump(self.nxs, self.nys)
        cm = colormaps['viridis'] 
        im = axes[0].imshow(all_y, origin='lower', aspect='auto',
                            interpolation='none', extent=[-self.Q0,self.Q0,-self.Q0,self.Q0],
                            cmap=cm)
        axes[0].contour(all_Kxs, all_Kys, all_y, [self.params.omega_0])
        im = axes[1].imshow(all_pumps, origin='lower', aspect='auto',
                            interpolation='none', extent=[-self.Q0,self.Q0,-self.Q0,self.Q0],
                            cmap=cm)
        fp = os.path.join('figures/2d_real_space_dispersion_pump.png')
        fig.savefig(fp, bbox_inches='tight', dpi=350)
        plt.close(fig)

def plot_phases(params, tend=500):
    fig_t, ax_t = plt.subplots()
    fig, axes=plt.subplots(2,4,figsize=(12,6.5),sharex=True)
    axes=axes.flatten()
    htc = RealHTC(params)
    cm = colormaps['viridis']
    htc.evolve(tend=tend)
    Q0=htc.Q0
    min_n = Q0//2
    max_n = 3 * Q0//2
    num = 8
    for i in range(num):
        Delta_t = i * params.dt
        ind = -1 - i
        axes[num-1-i].set_title(rf'$t_f-{Delta_t:.1f}$')
        vals = np.angle(htc.dynamics['a_nm'][ind])[min_n:max_n+1, min_n:max_n+1]
        #vals -= np.angle(htc.dynamics['a_nm'][ind-1])[min_n:max_n+1, min_n:max_n+1]
        im = axes[num-1-i].imshow(vals,
                              interpolation='none',
                              cmap=cm,
                              extent=[-min_n,min_n,-min_n,min_n],
                              origin='lower',
                              vmin=-np.pi,
                              vmax=np.pi,
                              aspect='equal')
        if i < num-4:
            continue
        cbar = fig.colorbar(im, ax=axes[i], aspect=20, location='bottom')
        cbar.ax.set_xticks([-np.pi,0,np.pi])
        cbar.ax.set_xticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
    ax_t.plot(htc.t_fs, np.abs(htc.dynamics['a_nm'][:,Q0,Q0])**2)
    fig_t.savefig('figures/mean-field_2d_phase_dynamics.png')
    plt.close(fig_t)
    fig.suptitle(r'$t_f={}$'.format(tend))
    fig.savefig('figures/mean-field_2d_phase.png')
    plt.close(fig)

def plot_input_output(params, pump_strengths, tend=500):
    fig, axes = plt.subplots(2,2,figsize=(10,8))
    params.dt = 100
    for i, pump in enumerate(pump_strengths):
        params.pump_strength = pump
        pump_str = '{:.2g}'.format(pump/params.Gam_down)
        htc = RealHTC(params)
        htc.evolve(tend=tend)
        n_nn = np.abs(np.diag(htc.dynamics['a_nm'][-1]))**2
        Q0 = htc.Q0
        xs = np.arange(-Q0, Q0+1, dtype=int)[Q0:]
        n_norm = np.max(n_nn)
        #n_norm=1
        cm = colormaps['viridis']
        axes[0,0].plot(xs, n_nn[Q0:]/n_norm, label=pump_str)
        if i == len(pump_strengths)-1:
            im = axes[0,1].imshow(np.angle(htc.dynamics['a_nm'][-1]),
                              interpolation='none',
                              cmap=cm,
                              extent=[-Q0,Q0,-Q0,Q0],
                              origin='lower',
                              aspect='auto')
            cbar = fig.colorbar(im, ax=axes[0,1], aspect=20)
        axes[1,0].plot(xs, np.diag(htc.dynamics['z_nm'][-1])[Q0:])
        axes[1,1].plot(htc.dynamics['t'], np.abs(htc.dynamics['a_nm'][:,htc.Q0, htc.Q0])**2)
    axes[0,0].set_title(r'$|\langle a_{nn} \rangle|^2$ (normalised)')
    axes[1,1].set_title(r'$|\langle a_{n=m=0} \rangle|^2/N_E$')
    axes[0,1].set_title(r'Arg$\langle a_{nm} \rangle$')
    axes[1,0].set_title(r'$\langle \sigma^z_{nn} \rangle$')
    axes[0,0].legend(title=r'$\Gamma^\uparrow/\Gamma^\downarrow$')
    axes[1,0].set_xlabel(r'$n$')
    axes[1,1].set_xlabel(r'$t$ (fs)')
    fig.savefig('figures/mean-field_2d_input_output.png')

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    ################################
    params = Parameters(omega_0=0.0, # zero-phonon line
                        omega_c=0.0, # MIDDLE of tight-binding dispersion
                        Q0=40, # 2*Q0+1 sites (so Q0 to the right of 0)
                        NE=100, # Number of emitters per gap
                        g=0.01, # INDIVIDUAL light-matter coupling (collective gSqrtNE)
                        t=0.0001, # Hopping parameter
                        kappa=0.01, # photon loss
                        Gam_z=0.0, # emitter pure dephasing
                        Gam_down=0.001, # emitter decay
                        gam_ee=0.00001, # emitter EEA rate
                        pump_strength=0.1, # emitter pump strength (maximum of Gaussian), overwritten in plot_input_output below
                        pump_width=2, # Pump width (Gaussian s.d.) in number of SITES
                        dt=10,
                        )
    #single_mode_comparison()
    #pump_strengths = params.Gam_down * np.logspace(0, 1, num=4)
    #pump_strengths = [0.002, 0.004, 0.008]
    #plot_input_output(params, pump_strengths, tend=5000)
    params.pump_strength = 0.004
    plot_phases(params, tend=2500)
