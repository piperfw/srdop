#!/usr/bin/env python

import logging, os, sys, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from time import time
import numpy as np
from scipy.integrate import RK45, DOP853
from types import SimpleNamespace
SOLVER = DOP853
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
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
        #self.create_coeffs()
        self.create_initial_state()
        return

    def add_useful_params(self):
        params = self.params
        params.gSqrtNE = params.g * np.sqrt(params.NE)
        params.Nk = 2 * params.Q0 + 1
        self.Q0, self.Nk, self.NE = params.Q0, params.Nk, params.NE
        params.gam_E = ( params.NE - 1 ) * params.gam_ee
        self.nxs, self.nys = np.meshgrid(np.arange(self.Nk), np.arange(self.Nk), indexing='ij')
        self.Kxs, self.Kys = self.nxs - self.Q0, self.nys - self.Q0
        self.delta = np.eye(self.Nk)

    def create_slices(self):
        Nk = self.Nk
        names = ['a_dag_a', 'sig_z', 'a_sig_plus',
                 'sig_plus_sig_minus', 'sig_z_sig_z']
        shapes = [(Nk,Nk,Nk,Nk), (Nk,Nk), (Nk,Nk,Nk,Nk), (Nk,Nk,Nk,Nk), (Nk,Nk)]
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
        state[state_dic['sig_z_sig_z']['slice']] = 1.0
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

    #def create_coeffs(self):
    #    params = self.params
    #    self.asp_coeff = 1j * (params.omega_0 - params.omega_c) \
    #            - 0.5 * (self.Gam_n + params.kappa + params.gam_E)# omega_c centre of band
    #    #pump_2d = np.broadcast_to(self.Gam_n, (self.Nk, self.Nk))
    #    ##X, Y = np.meshgrid(self.Gam_n, self.Gam_n)
    #    ##tot_pump = X + Y
    #    #self.spsm_coeff = - 0.5 * (pump_2d.T + pump_2d) - params.gam_E

    def split_reshape(self, state):
        to_return = []
        for name in self.state_dic:
            to_return.append(state[self.state_dic[name]['slice']].reshape(self.state_dic[name]['shape']))
        return to_return

    def eoms(self, t, state):
        """Real space EoMs (Square lattice)"""
        delta_nxmx, delta_nymy = self.delta, self.delta
        delta_nm = contract('xa,yb->xyab', self.delta, self.delta)
        params = self.params
        roll_plus = lambda arr, ax: np.roll(arr, -1, axis=ax)
        roll_minus = lambda arr, ax: np.roll(arr, 1, axis=ax)
        roll_plus_minus_sum = lambda arr, ax: np.roll(arr, -1, axis=ax) + np.roll(arr, +1, axis=ax)
        #swap_nm = lambda arr: np.transpose(arr, axes=[2,3,0,1]) #e.g. a_n^d a_m -> a_m a_n^d. OR use contract..
        kappa, g, t_hop, NE, gam_E, gam_ee =\
                params.kappa, params.g, params.t, params.NE, params.gam_E, params.gam_ee
        Gam_T, Gam_D, Gam_n = self.Gam_T, self.Gam_D, self.Gam_n
        a_dag_a, sig_z, a_sig_plus, sig_plus_sig_minus, sig_z_sig_z = \
                self.split_reshape(state)
    
        # <a_n^† a_m> 
        da_dag_a = - kappa * a_dag_a \
                       + 1j * g * NE * (contract('xyab->abxy', a_sig_plus) - np.conj(a_sig_plus)) \
                       - 1j * t_hop * (roll_plus_minus_sum(a_dag_a, 0) + roll_plus_minus_sum(a_dag_a, 1)) \
                       + 1j * t_hop * (roll_plus_minus_sum(a_dag_a, 2) + roll_plus_minus_sum(a_dag_a, 3))
    
        # <σ_n^z> 
        dsig_z = -(Gam_n + 2 * gam_E) * sig_z \
                      + (Gam_D - gam_E) \
                      + 4 * g * np.imag(contract('xyxy->xy', a_sig_plus)) \
                      - gam_E * sig_z_sig_z
    
        # <a_m σ_n^+> 
        da_sig_plus = 1j * (params.omega_0 - params.omega_c) * a_sig_plus \
                        - 0.5 * contract('xy,abxy->abxy', Gam_n + kappa + gam_E, a_sig_plus) \
                        - 0.5 * gam_E * contract('abxy,xy->abxy', a_sig_plus, sig_z) \
                        + 1j * t_hop * (roll_plus_minus_sum(a_sig_plus, 0) + roll_plus_minus_sum(a_sig_plus, 1)) \
                        - 1j * g * contract('xy,xyab->abxy', sig_z, a_dag_a) \
                        - 0.5j * g * contract('xy,xyab->abxy', sig_z, delta_nm) \
                        - 0.5j * g * delta_nm \
                        - 1j * g * contract('xyab,xyab->abxy', NE - delta_nm, sig_plus_sig_minus)
                        #- 0.5j * g * (contract('xy,xa,yb->abxy', sig_z, delta_nx, delta_ny) \
                        #- 0.5j * g * contract('xyab->abxy', delta_nm) \
    
        # <σ_n^+ σ_m^-> # self.spsm_coeff - self.create_coeffs
        dsig_plus_sig_minus = - 0.5 * contract('xy,xyab->xyab', Gam_n + gam_E, sig_plus_sig_minus) \
                              - 0.5 * contract('ab,xyab->xyab', Gam_n + gam_E, sig_plus_sig_minus) \
                              - 0.5 * gam_E * contract('xyab,xy->xyab', sig_plus_sig_minus, sig_z) \
                              - 0.5 * gam_E * contract('xyab,ab->xyab', sig_plus_sig_minus, sig_z) \
                              + gam_ee * contract('xyab,xyab,xy->xyab', delta_nm, sig_plus_sig_minus, 1 + sig_z) \
                              + 1j * g * contract('abxy,ab->xyab', a_sig_plus, sig_z) \
                              - 1j * g * contract('xyab,xy->xyab', np.conj(a_sig_plus), sig_z)
                              #+ gam_ee * delta_nm * sig_plus_sig_minus * (1 + sig_z) # CHECK
                              # check np.conj() term here as well

        # <σ_n^z σ_m^z>
        dsig_z_sig_z = - 2 * Gam_T * sig_z_sig_z \
                          + 2 * Gam_D * sig_z \
                          + 8 * g * sig_z * np.imag(contract('xyxy->xy', a_sig_plus)) \
                          - 2 * gam_ee * ((NE-2) * (sig_z + 2 * sig_z_sig_z +
                                                   sig_z * (3 * sig_z_sig_z - 2 * sig_z**2)))
    
        dy = np.concatenate((da_dag_a, dsig_z, da_sig_plus, dsig_plus_sig_minus, dsig_z_sig_z), axis=None)        
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
        num_checkpoints = 11 # checkpoints at 0, 25%,...
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
        nPs = np.zeros((Nt, self.Nk, self.Nk), dtype=float)
        nKs = np.zeros((Nt, self.Nk, self.Nk), dtype=float)
        nMs = np.zeros((Nt, self.Nk, self.Nk), dtype=float)
        g1s = np.zeros((Nt, self.Nk), dtype=complex)
        g1RRs = np.zeros((Nt, self.Q0+1), dtype=complex)
        self.dynamics = {'t': self.t_fs,
                         'nP': nPs,
                         'nK': nKs,
                         'nM': nMs,
                         'g1': g1s,
                         'g1RR': g1RRs,
                         'V': None, 
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
        nPh = contract('xyxy->xy', a_dag_a)
        self.check_real(nPh, t_index, 'photon number')
        self.dynamics['nP'][t_index] = np.real(nPh)
        nkp = fftshift(fft2(ifft2(a_dag_a, axes=[0,1]), axes=[2,3]))
        nkk = contract('kpkp->kp', nkp)
        self.check_real(nkk, t_index, 'photon number (k-space)')
        self.dynamics['nK'][t_index] = np.real(nkk)
        nM = self.NE * 0.5 * (sig_z + 1)
        self.check_real(nM, t_index, 'electronic population')
        self.dynamics['nM'][t_index] = np.real(nM)
        gRR = np.zeros(self.Q0+1, dtype=complex)
        if np.allclose(nPh, 0.0):
            g1 = np.zeros(self.Nk, dtype=complex)
        else:
            g1 = a_dag_a[self.Q0,:,self.Q0,self.Q0]/np.sqrt(nPh[self.Q0,:] * nPh[self.Q0,self.Q0])
            #for n in range(self.Q0+1):
            #    numer = a_dag_a[self.Q0+n,self.Q0-n]
            #    denom = np.sqrt(nPh[self.Q0+n] * nPh[self.Q0-n])
            #    gRR[n] = numer / denom
        self.dynamics['g1'][t_index] = g1
        #self.dynamics['g1RR'][t_index] = gRR

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
        all_Kxs, all_Kys = np.meshgrid(np.linspace(-self.Q0, self.Q0, 250), np.linspace(-self.Q0, self.Q0,250), indexing='ij')
        all_y = self.omega(all_Kxs, all_Kys)
        #all_nxs, all_nys = all_Kxs + self.Q0, all_Kys + self.Q0
        all_pumps = self.pump(self.nxs, self.nys)
        cm = colormaps['viridis'] 
        im = axes[0].imshow(all_y, origin='lower', aspect='auto',
                            interpolation='none', extent=[-self.Q0,self.Q0,-self.Q0,self.Q0],
                            cmap=cm)
        axes[0].contour(all_Kxs, all_Kys, all_y, [self.params.omega_0])
        im = axes[1].imshow(all_pumps, origin='lower', aspect='auto',
                            interpolation='none', extent=[-self.Q0,self.Q0,-self.Q0,self.Q0],
                            cmap=cm)
        fp = os.path.join(self.DEFAULT_DIRS['figures'], '2d_real_space_dispersion_pump.png')
        fig.savefig(fp, bbox_inches='tight', dpi=350)
        plt.close(fig)

    #def cauchy_mask(self, ada):
    #    delta = 1e-8
    #    ada = fftshift(ada)
    #    diags = np.diag(ada)
    #    Nk = len(diags)
    #    ada_p = np.array([x * np.ones(Nk) for x in diags])
    #    ada_k = ada_p.T
    #    diff =   ada_p * ada_k - np.abs(ada)**2
    #    mask = diff < - delta # delta for numerical tolerance
    #    return mask, diff 

def plot_dynamics(results):
    nph_tots = np.sum(results['dynamics']['nP'], axis=(1,2))
    nM_tots = np.sum(results['dynamics']['nM'], axis=(1,2))
    fig, axes = plt.subplots(1, 2, figsize=(8,6), constrained_layout=True, sharex=False)
    axes[0].plot(results['dynamics']['t'], nph_tots)
    axes[1].plot(results['dynamics']['t'], nM_tots/results['parameters'].NE)
    fig.savefig('figures/2d_real_space_dynamics.png', bbox_inches='tight', dpi=350)
    plt.close(fig)

def plot_input_output(results_list,
                      pump_strengths,
                      normalise=False, 
                      max_nph_curves=5,
                      xlims=None):
    num_pumps = len(pump_strengths)
    assert num_pumps == len(results_list)
    params = results_list[0]['parameters']
    Nk = 2 * params.Q0 + 1
    ratios = np.array(pump_strengths) / params.Gam_down
    ph_final = np.zeros((num_pumps, Nk, Nk), dtype=float)
    nK_final = np.zeros((num_pumps, Nk, Nk), dtype=float)
    nM_final = np.zeros((num_pumps, Nk, Nk), dtype=float)
    nK_final = np.zeros((num_pumps, Nk, Nk), dtype=float)
    g1_final = np.zeros((num_pumps, Nk), dtype=complex)
    #g1RR_final = np.zeros((num_pumps, params.Q0+1), dtype=complex)
    #adaga_final = np.zeros((num_pumps, Nk, Nk), dtype=complex) 
    #adaga_final_mask = np.zeros((num_pumps, Nk, Nk), dtype=bool) 
    fig, axes = plt.subplots(3, 2, figsize=(8,10), constrained_layout=True, sharex=False)# sharex='col')
    figk, axesk = plt.subplots(1,2, figsize=(8,3), constrained_layout=True)
    select_indices = np.round(np.linspace(0, num_pumps-1, max_nph_curves)).astype(int)
    pump_title = r'$\Gamma_\uparrow(0)/\Gamma_\downarrow$'
    Ks = np.arange(-params.Q0, params.Q0+1)
    cm = colormaps['viridis'] 
    Kextent = [-params.Q0,params.Q0,-params.Q0,params.Q0]
    myim = lambda ax, y, extent: ax.imshow(y, origin='lower', aspect='auto',
                            interpolation='none', extent=extent,
                            cmap=cm)
    for i, pump in enumerate(pump_strengths):
        logger.info(f'On pump {i+1} of {num_pumps}')
        params.pump_strength = pump
        results = results_list[i]
        ph_final[i] = results['dynamics']['nP'][-1]
        nK_final[i] = results['dynamics']['nK'][-1] # already fftshifted to ascending order
        nM_final[i] = results['dynamics']['nM'][-1]
        g1_final[i] = results['dynamics']['g1'][-1]
        #g1RR_final[i, :] = results['dynamics']['g1RR'][-1, :]
        if i not in select_indices:
            continue
        if normalise:
            y1 = ph_final[i]/ph_final[i][params.Q0, params.Q0]
        else:
            y1 = ph_final[i]
        y2 = nM_final[i]/params.NE
        y3 = nK_final[i]
        pump_str = r'${:.2g}$'.format(round(ratios[i],5))
        axes[0,1].plot(y1[params.Q0, params.Q0:], label=pump_str)
        axes[1,1].plot(y2[params.Q0, params.Q0:], label=pump_str)
        if i==0:
            #im = myim(axesk[1], y1, Kextent)
            im = myim(axesk[1], y3, Kextent)
            cbar = figk.colorbar(im, ax=axesk[1], aspect=20)
        #axes[2,1].plot(np.abs(g1_final[i,htc.Q0:]), label=pump_str)
        #axes[2,0].plot(np.abs(g1RR_final[i,:]), label=pump_str)
        axesk[0].plot(Ks, y3[params.Q0,:], label=pump_str)
        #if i == num_pumps - 1:
        #    Nk = htc.Nk
        #    final_ada = results['final_state'][htc.state_dic['a_dag_a']['slice']].reshape((Nk, Nk))
        #    nkp = fftshift(fft(ifft(final_ada, axis=0), axis=1))
        #    mask, diff = htc.cauchy_mask(nkp)
        #    adaga_one = np.ma.masked_array(np.copy(nkp),
        #                                   mask=mask)
        #    cm = colormaps['viridis'] 
        #    cm.set_bad('red')
        #    extent = [htc.Ks[0], htc.Ks[-1],htc.Ks[0], htc.Ks[-1]]
        #    im = axesk[1].imshow(np.real(adaga_one), origin='lower', aspect='auto',
        #                    interpolation='none', extent=extent, cmap=cm,
        #                    label=r'${:.2g}$'.format(round(ratios[i],5)))
        #    cbar = figk.colorbar(im, ax=axesk[1], aspect=20)
        #    axesk[1].set_title(r'$\rm{Re}\,n_{kp}\quad($' + pump_title + r'$=$'+pump_str+r'$)$')
    #htc.plot_dispersion_pump()
    if xlims is not None:
        axes[0,1].set_xlim(xlims)
        axes[1,1].set_xlim(xlims)
        axes[2,1].set_xlim(xlims)
        axes[2,0].set_xlim(xlims)
    ph_tots = np.sum(ph_final, axis=(1,2)) # Sum over all lattice positions 
    nM_tots = np.sum(nM_final, axis=(1,2)) # sum over all lattice positions
    axes[1,0].set_xlabel(pump_title)
    axes[2,0].set_xlabel(r'$n$')
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
    axes[2,1].set_title(r'$|g^{(1)}(R)|$')
    axes[2,0].set_title(r'$|g^{(1)}(R,-R)|$')
    axes[0,0].loglog(ratios, ph_tots)
    axes[1,0].loglog(ratios, nM_tots)
    #axes[2,0].plot(ratios, np.abs(g1_final[:,htc.Q0]))
    #axes[2,0].set_xscale('log')
    axes[1,1].legend(title=pump_title)
    axes[0,1].legend(title=pump_title)
    axesk[0].legend(title=pump_title)
    fig.suptitle(r'$N_k={Nk}\ N_E={NE}\ g={g}\ \kappa={kappa}\ \Gamma^\downarrow={Gam_down:.2g}\  t={t}\ \gamma^{{\rm{{ee}}}}={gam_ee}$'.format(**params.__dict__))
    fig.savefig('figures/2d_real_space_input_output.png', bbox_inches='tight', dpi=350)
    figk.savefig('figures/2d_real_space_cauchy.png', bbox_inches='tight', dpi=350)
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
                        Q0=20, # 2*Q0+1 sites (so Q0 to the right of 0)
                        NE=100, # Number of emitters per gap
                        g=0.01, # INDIVIDUAL light-matter coupling (collective gSqrtNE)
                        t=0.4, # Hopping parameter (photon)
                        kappa=0.1, # photon loss
                        Gam_z=0.0, # emitter pure dephasing
                        Gam_down=1e-4, # emitter decay
                        gam_ee=1e-4, # emitter EEA rate
                        pump_strength=None, # emitter pump strength (maximum of Gaussian)
                        pump_width=4, # Pump width (Gaussian s.d.) in number of SITES
                        dt=10, # Timestep to record variables
                        )
    results_list = []
    pump_strengths = [0.005, 0.01, 0.05, 0.1, 0.25]
    for pump in pump_strengths:
        params.pump_strength = pump
        htc = RealHTC(params)
        fp = 'data/2d/w0{}Q0{}_pump{}.pkl'.format(params.omega_0, params.Q0, pump)
        print(pump, fp)
        if os.path.exists(fp):
            with open(fp, 'rb') as fb:
                results = pickle.load(fb)
        else:
            results = htc.evolve(tend=250)
            with open(fp, 'wb') as fb:
                pickle.dump(results, fb)
        results_list.append(results)
    htc.plot_dispersion_pump()
    #plot_dynamics(results) # total photon number and molecular population vs time (check convergence)
    plot_input_output(results_list, pump_strengths, normalise=True, xlims=None)

