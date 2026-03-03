from difai.aif import AIF_Env
from jax import numpy as jnp
from jax.nn import sigmoid
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
import numpy as np

class Mouse_Cursor(AIF_Env):

    def __init__(self, x0=jnp.array([-0.1, 0.0, 0.0, 0.03]), k=40.0, d=0.7, dt=0.02):
        self.x0 = x0
        self.dt = dt
        self.sys_params = {'k': k, 'd': d}
        self.non_negative_sys_params = [0, 1]
        self.dim_action = 1
        self.dim_observation = 4
        self.jitable = True

    # Overriding abstract method
    @staticmethod
    def _forward_complete(x, u, dt, random_realisation, key, k, d):
        """
        Computes one forward step with the spring-damper system (unit mass)
        :param x: initial state (position, velocity, target position, target radius)
        :param u: control value (acceleration)
        :param dt: time step duration in seconds
        :param k: stiffness parameter
        :param d: damping parameter
        :return:
            - x: resulting state
        """
        # 1. Compute required system matrices
        A = jnp.array([[0,          1,          0],    
                       [-k,       - d,          1],
                       [0,          0,          0]])

        step_fn = lambda y, t: A @ y
        
        # 2. (Online) Control Algorithm 
        y = jnp.hstack([x[0], x[1], u]) 
        solution = odeint(step_fn, y, jnp.array([0,dt]), rtol=1.4e-8, atol=1.4e-8)
        y = solution[1]

        x = x.at[:2].set(y[:2]) # update cursor position and velocity
       
        return x 

    # Overriding abstract method
    @staticmethod
    def _get_observation_complete(x, k, d):
        sigmoid_steepness = 1e6
        inside_button = sigmoid(-sigmoid_steepness*(jnp.abs(x[0] - x[2]) - x[3]))
        return jnp.hstack([x[0], inside_button, x[2], x[3]]) # observe position, whether inside button, target position, target radius


def plot_results(dt, xx, oo, bb, aa, aa_applied, lll, NEFE_PLAN=[], PRAGMATIC_PLAN=[], INFO_GAIN_PLAN=[], NEFES=[], PRAGMATICS=[], INFO_GAINS=[], bb_sys=[], bb_noise=[], bb_after_rt=[], reaction_time_steps=0, belief_button=False, hide_belief=False,
                 plot_axes=['pos', 'vel', 'button','target_position', 'target_radius',  'acc', 'loss', 'nefe'],
                 distance_unit="m", ic_timesteps=None, ic_pred_error=None, fig=None, ax=None, figsize_x=None):
    rows = []
    
    if 'pos' in plot_axes:
        rows.append(['pos', 'pos'])
    if 'vel' in plot_axes:
        rows.append(['vel', 'vel'])
    if 'button' in plot_axes:
        rows.append(['button', 'button'])
    if len(xx[0]) > 2:
        if 'target_position' in plot_axes:
            rows.append(['target_position', 'target_position'])
        if 'target_radius' in plot_axes:
            rows.append(['target_radius', 'target_radius'])
    if 'loss' in plot_axes and 'lr' in plot_axes:
        rows.append(['loss', 'lr'])
    elif 'loss' in plot_axes and len(lll) > 0:
        rows.append(['loss', 'loss'])
    elif 'lr' in plot_axes and len(llr) > 0:
        rows.append(['lr', 'lr'])
    elif 'nefe' in plot_axes and len(NEFE_PLAN) > 0:
        rows.append(['nefe', 'nefe'])

    print(f"Rows: {rows}")
    if fig is None:
        if figsize_x is None:
            figsize_x = 7
        fig, ax = plt.subplot_mosaic(rows, figsize=(figsize_x, 3*len(rows)))
    
    numsteps = len(aa)

    std_mult = 3
    t = np.arange(0, numsteps * dt + 1e-6, dt)

    ## Position
    if 'pos' in plot_axes:  
        ax['pos'].fill_between(t, [x[2] - x[3] for x in xx], [x[2] + x[3] for x in xx], alpha=0.2, label='Target', color='red',  hatch='//')
        
        # Real position
        ax['pos'].plot(t, [x[0] for x in xx], label=['Cursor'], color="blue")
        ax['pos'].plot(t[1:], [o[0] for o in oo], label=['$o_1(t)$'], color = "orange", alpha=0.7, linestyle='--')

        # Belief
        if bb is not None and not hide_belief:
            mean = [b[0][0] for b in bb]
            var = [b[1][0, 0] for b in bb]
            ax['pos'].plot(t, mean, label=['$Q^s_1(t-\\tau)$'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['pos'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor="purple", alpha=1 / 4)


        # Buttons position and width belief
        if belief_button:
            for i,b in enumerate(bb_sys):
                ax['pos'].fill_between(t[i], [b[0][0] - b[0][1] for _ in t], [b[0][0] + b[0][1] for _ in t], alpha=0.5, color='orange')

        ax['pos'].set_xlabel('Time [s]')
        ax['pos'].set_ylabel(f'Cursor Position [{distance_unit}]')
        # ax['pos'].legend()

    ## Button State
    if 'button' in plot_axes:
        # Observations
        ax['button'].scatter(t[1:], [o[1] for o in oo], label=['button observation'], color='orange')

        ax['button'].set_xlabel('Time [s]')
        ax['button'].set_ylabel('Button State')
        ax['button'].legend()

    ## Velocity
    if 'vel' in plot_axes:
        ax['vel'].plot(t, [x[1] for x in xx], label=['velocity'])

        # Belief 
        if bb is not None and not hide_belief:
            mean = [b[0][1] for b in bb]
            var = [b[1][1, 1] for b in bb]
            ax['vel'].plot(t, mean, label=['belief velocity'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['vel'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor="purple", alpha=1 / 4)

        ax['vel'].set_xlabel('Time [s]')
        ax['vel'].set_ylabel(f'Velocity [{distance_unit}/s]')
        ax['vel'].legend()

    if len(xx[0]) > 2:
        if 'target_position' in plot_axes:
            ax['target_position'].plot(t, [x[2] for x in xx], label=['target_position'])
            mean = [b[0][2] for b in bb]
            var = [b[1][2, 2] for b in bb]
            ax['target_position'].plot(t, mean, label=['target_position belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['target_position'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            ax['target_position'].plot(t[1:], [o[2] for o in oo], label=['target_position observation'], color='orange', linestyle='--')        
            ax['target_position'].set_xlabel('Time [s]')
            ax['target_position'].set_ylabel('Target Position')
            ax['target_position'].legend()
        if 'target_radius' in plot_axes:
            mean = [b[0][3] for b in bb]
            var = [b[1][3, 3] for b in bb]
            ax['target_radius'].plot(t, [x[3] for x in xx], label=['target_radius'])
            ax['target_radius'].plot(t, mean, label=['target_radius belief'], color='purple')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['target_radius'].fill_between(t, ribbons[:, 0], ribbons[:, 1], facecolor='purple', alpha=1 / 4)
            ax['target_radius'].plot(t[1:], [o[3] for o in oo], label=['target_radius observation'], color='orange', linestyle='--')
            ax['target_radius'].set_xlabel('Time [s]')
            ax['target_radius'].set_ylabel('Target Radius')
            ax['target_radius'].legend()


    ## Control
    # Mouse acceleration
    if 'acc' in plot_axes:
        ax['acc'].plot(t[:-1], [a[0] for a in aa], label='chosen mouse acceleration')
        ax['acc'].plot(t[:-1], [a[0] for a in aa_applied], label='applied mouse acceleration')
        ax['acc'].set_xlabel('Time [s]')
        ax['acc'].set_ylabel(f'Control [{distance_unit}/s^2]')
        ax['acc'].legend()

    ## Loss in belief update
    if 'loss' in plot_axes:
        for i, l in enumerate(lll):
            ax['loss'].plot(l, color='purple', alpha=0.1)

    if 'nefe' in plot_axes:
        if len(NEFE_PLAN) > 0:
            ax['nefe'].plot(t[1:], NEFE_PLAN, label='NEFE')
            if len(PRAGMATIC_PLAN) > 0:
                ax['nefe'].plot(t[1:], PRAGMATIC_PLAN, label='Pragmatic')
            if len(INFO_GAIN_PLAN) > 0:
                ax['nefe'].plot(t[1:], INFO_GAIN_PLAN, label='Info Gain')
            ax['nefe'].set_xlabel('Time [s]')
            ax['nefe'].set_ylabel('NEFE')
            ax['nefe'].legend()
        if len(NEFES) > 0:
            ax['nefe'].violinplot(NEFES, positions=t[1:], showmeans=False, showmedians=False, widths=0.1, showextrema=False)

    if reaction_time_steps >0  and len(bb_after_rt) > 0:
        if 'pos' in plot_axes:
            mean = [b[0][0] for b in bb_after_rt]
            var = [b[1][0, 0] for b in bb_after_rt]
            ax['pos'].plot(t[:-1], mean, label=['$\\tilde{Q}^s_1(t)$'], color='green')
            ribbons = np.array([[mean[i] - std_mult * np.sqrt(var[i]), mean[i] + std_mult * np.sqrt(var[i])] for i in range(len(mean))])
            ax['pos'].fill_between(t[:-1], ribbons[:, 0], ribbons[:, 1], facecolor="green", alpha=1 / 6)
            ax['pos'].set_xlabel(None)

    if ic_timesteps is not None:
        line_max = np.array([x[0] for x in xx]).max()
        line_min = np.array([x[0] for x in xx]).min()
        # IC timesteps
        ax['pos'].vlines(np.array(ic_timesteps)*dt, ymin=line_min, ymax=line_max, color='grey', linestyle='dotted', label='IC Timesteps', alpha=0.5)
        if ic_pred_error is not None:
            for ic_timestep, pred_belief_state, prediction_error in ic_pred_error:
                # Plot box displaying the predicted belief at that time
                ax['pos'].errorbar(ic_timestep*dt, pred_belief_state[0][0], yerr=3*np.sqrt(pred_belief_state[1][0, 0]), color='grey', fmt="x")
                ax['pos'].text(ic_timestep*dt, line_min-0.1, f"{prediction_error:.2f}", fontsize=8, color='grey', ha='center', va='bottom')

    fig.tight_layout()

    return fig, ax, t
    