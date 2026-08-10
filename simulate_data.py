''' Aim of this file is to simulate population activity data according  to dynamical system model.

Model 1: Wilson-Cowan model (WC)
Model 2 : Lin model (see PDF paper)

Each model specifies the dynamics of the excitatory and inhibitory populations.

Signature of each model should be : initial_state, parameters -> time_series of population activity
Both initial state and parameters are dictionaries of float values.
'''
import numpy as np

def _alpha_drive(boxcar, tau_alpha, dt):
    """Convert a 0/1 light-on boxcar into an alpha-function drive time course.

    Paper S1.9: because C1V1_T/T has slower kinetics than ChR2, the inhibitory
    external input X_I is modeled as an alpha function t*exp(-t/tau_alpha) rather
    than a boxcar. The alpha kernel is triggered at each rising edge of the
    boxcar (so paired pulses sum), and peak-normalized to 1 at t=tau_alpha so the
    XI amplitude parameter cleanly means "peak inhibitory drive".

    Parameters
    ----------
    boxcar : (n_times,) array of 0/1 marking when the inhibitory light is on.
    tau_alpha : float, alpha-function time constant (same units as dt*index).
    dt : float, timestep.
    """
    boxcar = np.asarray(boxcar)
    onsets = np.flatnonzero((boxcar > 0) & (np.r_[0, boxcar[:-1]] == 0))
    idx = np.arange(len(boxcar))
    drive = np.zeros(len(boxcar))
    for o in onsets:
        s = (idx - o) * dt
        on = s >= 0
        drive[on] += (s[on] / tau_alpha) * np.exp(1.0 - s[on] / tau_alpha)
    return drive

def wilson_cowan_model(initial_state, parameters, stimuli, time_steps, h=0.01):
    """
    Simulate the Wilson-Cowan model of excitatory and inhibitory populations.

    Parameters
    ----------
    initial_state : dict
        Initial state of the populations, e.g., {'E': 0.1, 'I': 0.1}
    parameters : dict
        Model parameters, e.g., {'E_max': 1.0, 'I_max': 1.0, 'W_EE': 1.0, 'W_EI': 1.0, 'W_IE': 1.0, 'W_II': 1.0, 'C_E': 0.1, 'C_I': 0.1}
    stimuli : (n_time_steps, 2) np.ndarray 
        [:, 0] is the transient input to the excitatory population and [:, 1] is the transient input to the inhibitory population. 
        Each value is either 0 or 1, indicating whether the stimulus is off or on at that time step.
    time_steps : int
        Number of time steps to simulate.

    Returns
    -------
    np.ndarray
        Time series of population activity with shape (time_steps, 2) for E and I.
    """
    E = np.zeros(time_steps)
    I = np.zeros(time_steps)
    
    E[0] = initial_state['E']
    I[0] = initial_state['I']

    E_max = parameters['E_max']
    I_max = parameters['I_max']
    W_EE = parameters['W_EE']
    W_EI = parameters['W_EI']
    W_IE = parameters['W_IE']
    W_II = parameters['W_II']
    tau_E = parameters['tau_E']
    tau_I = parameters['tau_I']

    C_E = parameters['C_E'] # constant input to the excitatory population
    C_I = parameters['C_I'] # constant input to the inhibitory population    

    XE = parameters['XE'] # transient input to the excitatary population 
    XI = parameters['XI'] # transient input to the inhibitory population
    
    ## p27 : inhibitory pulse is considered slower and is applied with an alpha function in the paper. For now use a boxcar function for both
    # tau_alpha = parameters['tau_alpha'] # time constant for alpha function for inhibitory input
    # XI_drive = XI * _alpha_drive(stimuli[:, 1], tau_alpha, h)

    for t in range(1, time_steps):
        stim_E_on = stimuli[t][0]
        stim_I_on = stimuli[t][1]
        E_dot = -E[t-1] + (E_max - E[t-1])* np.maximum((W_EE * E[t-1] - W_EI * I[t-1] + C_E + XE * stim_E_on), 0)
        E_dot /= tau_E
        I_dot = -I[t-1] + (I_max - I[t-1])* np.maximum((W_IE * E[t-1] - W_II * I[t-1] + C_I + XI * stim_I_on), 0)
        I_dot /= tau_I
        E[t] = E[t-1] + h * E_dot
        I[t] = I[t-1] + h * I_dot
            
    return np.column_stack((E, I))

def wilson_cowan_slow_inhibition_model(initial_state, parameters, stimuli, time_steps, h=0.01):
    """
    Simulate the Wilson-Cowan model with slow inhibition - as proposed in Lin's paper

    Parameters
    ----------
    initial_state : dict
        Initial state of the populations, e.g., {'E': 0.1, 'I': 0.1}
    parameters : dict
        Model parameters, e.g., {'E_max': 1.0, 'I_max': 1.0, 'W_EE': 1.0, 'W_EI': 1.0, 'W_IE': 1.0, 'W_II': 1.0, 'C_E': 0.1, 'C_I': 0.1}
    stimuli : (n_time_steps, 2) np.ndarray 
        [:, 0] is the transient input to the excitatory population and [:, 1] is the transient input to the inhibitory population. 
        Each value is either 0 or 1, indicating whether the stimulus is off or on at that time step.
    time_steps : int
        Number of time steps to simulate.

    Returns
    -------
    np.ndarray
        Time series of population activity with shape (time_steps, 2) for E and I.
    """
    E = np.zeros(time_steps)
    I = np.zeros(time_steps)
    S = np.zeros(time_steps) # slow inhibition from TRN activity
    
    E[0] = initial_state['E']
    I[0] = initial_state['I']
    S[0] = initial_state['S']

    E_max = parameters['E_max']
    I_max = parameters['I_max']
    W_EE = parameters['W_EE']
    W_EI = parameters['W_EI']
    W_IE = parameters['W_IE']
    W_II = parameters['W_II']
    W_ES = parameters['W_ES'] # slow inhibition from TRN activity
    W_IS = parameters['W_IS'] # slow inhibition from TRN activity

    tau_E = parameters['tau_E']
    tau_I = parameters['tau_I']
    tau_S = parameters['tau_S']

    C_E = parameters['C_E'] # constant input to the excitatory population
    C_I = parameters['C_I'] # constant input to the inhibitory population    

    XE = parameters['XE'] # transient input to the excitatary population 
    XI = parameters['XI'] # transient input to the inhibitory population

    for t in range(1, time_steps):
        stim_E_on = stimuli[t][0]
        stim_I_on = stimuli[t][1]

        S_dot = -S[t-1] + I[t-1]
        S_dot /= tau_S
        S[t] = S[t-1] + h * S_dot
        
        E_dot = -E[t-1] + (E_max - E[t-1])* np.maximum((W_EE * E[t-1] - W_EI * I[t-1] - W_ES * S[t-1] + C_E + XE * stim_E_on), 0)
        E_dot /= tau_E
        I_dot = -I[t-1] + (I_max - I[t-1])* np.maximum((W_IE * E[t-1] - W_II * I[t-1] - W_IS * S[t-1] + C_I + XI * stim_I_on), 0)
        I_dot /= tau_I
        E[t] = E[t-1] + h * E_dot
        I[t] = I[t-1] + h * I_dot
            
    return np.column_stack((E, I, S))

def lin_model(initial_state, parameters, stimuli, time_steps, h=0.01):
    """
    Simulate Lin's model of excitatory and inhibitory populations as described in paper.

    Parameters
    ----------
    initial_state : dict
        Initial state of the populations + latent variables e.g., {'E': 0.1, 'I': 0.1, 'S': 0.1, 'L': 0.1, 'B': 0.1, 'T': 0.1}
    parameters : dict
        Model parameters, e.g., {'W_EE': 1.0, 'W_EI': 1.0, 'W_IE': 1.0, 'I_EE': 1.0, 'C_E': 0.1, 'C_I': 0.1}
    stimuli : (n_time_steps, 2) np.ndarray 
        [:, 0] is the transient input to the excitatory population and [:, 1] is the transient input to the inhibitory population. 
        Each value is either 0 or 1, indicating whether the stimulus is off or on at that time step.
    time_steps : int
        Number of time steps to simulate.

    Returns
    -------
    np.ndarray
        Time series of population activity with shape (time_steps, 2) for E and I.
    """
    E = np.zeros(time_steps)
    I = np.zeros(time_steps)
    # These 5 values are the latent variables in the model, which will form the "state"
    S = np.zeros(time_steps)
    L = np.zeros(time_steps)
    R = np.zeros(time_steps)
    B = np.zeros(time_steps)
    T = np.zeros(time_steps)
    
    E[0] = initial_state['E']
    I[0] = initial_state['I']
    S[0] = initial_state['S']
    L[0] = initial_state['L'] # LGN neurons
    R[0] = initial_state['R'] # TRN neurons
    B[0] = initial_state['B']
    T[0] = initial_state['T'] # slow inhibition from TRN activity 

    E_max = parameters['E_max']
    I_max = parameters['I_max']
    L_max = parameters['L_max']

    W_EE = parameters['W_EE']
    W_EI = parameters['W_EI']
    W_IE = parameters['W_IE']
    W_II = parameters['W_II']
    W_ES = parameters['W_ES']
    W_IS = parameters['W_IS']
    W_EL = parameters['W_EL']
    W_IL = parameters['W_IL']
    W_LB = parameters['W_LB']
    W_LT = parameters['W_LT']
    W_LE = parameters['W_LE']
    W_RE = parameters['W_RE']
    W_RL = parameters['W_RL']
    
    tau_E = parameters['tau_E']
    tau_I = parameters['tau_I']
    tau_S = parameters['tau_S']
    tau_R = parameters['tau_R']
    tau_B = parameters['tau_B']

    C_E = parameters['C_E'] # constant input to the excitatory population
    C_I = parameters['C_I'] # constant input to the inhibitory population    
    C_L = parameters['C_L']
    C_R = parameters['C_R']

    XE = parameters['XE'] # transient input to the excitatory population 
    XI = parameters['XI'] # transient input to the inhibitory population

    for t in range(1, time_steps):
        stim_E_on = stimuli[t][0]
        stim_I_on = stimuli[t][1]

        S_dot = -S[t-1] + I[t-1]
        S_dot /= tau_S
        S[t] = S[t-1] + h * S_dot

        L_dot = -L[t-1] + (L_max - L[t-1])* np.maximum((W_LE * E[t-1] + C_L + W_LB * L[t-1] * B[t-1] - W_LT * T[t-1]), 0)
        L_dot /= tau_E # use tau_E - same time constant 
        L[t] = L[t-1] + h * L_dot

        T_dot = -T[t-1] + R[t-1]
        T_dot = T_dot / tau_S
        T[t] = T[t-1] + h * T_dot

        B_dot = -B[t-1] + (1 - B[t-1]) * np.maximum(1 - L[t-1], 0)
        B_dot /= tau_B
        B[t] = B[t-1] + h * B_dot

        R_dot = -R[t-1] + (1 - R[t-1]) * np.maximum(W_RE * E[t-1] + W_RL*L[t-1] - C_R, 0)
        R_dot /= tau_R
        R[t] = R[t-1] + h * R_dot        

        E_dot = -E[t-1] + (E_max - E[t-1])* np.maximum(W_EE * E[t-1] - W_EI * I[t-1] + C_E + XE * stim_E_on - W_ES * S[t-1] + W_EL * L[t-1], 0)
        E_dot /= tau_E
        I_dot = -I[t-1] + (I_max - I[t-1])* np.maximum(W_IE * E[t-1] - W_II * I[t-1] + C_I + XI * stim_I_on - W_IS * S[t-1] + W_IL * L[t-1], 0)
        I_dot /= tau_I

        E[t] = E[t-1] + h * E_dot
        I[t] = I[t-1] + h * I_dot

        # state = {
        #     'S' : S[t],
        #     'L' : L[t],
        #     'B' : B[t],
        #     'R' : R[t],
        #     'T' : T[t]
        # }

    return np.column_stack((E, I, S, L, B, R, T))