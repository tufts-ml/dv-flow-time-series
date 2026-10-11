import torch as th
import numpy as np

from .integrators import ode

    

def check_interval(reverse=False):
    t0 = 0
    t1 = 1
    
    if reverse:
        t0, t1 = 1 - t0, 1 - t1

    return t0, t1




def get_drift():
    """member function for obtaining the drift of the probability flow ODE"""
    
    def velocity_ode(x, t, model, **model_kwargs):
        if 'x_deter' in model_kwargs:
            x_deter = model_kwargs['x_deter']
            model_kwargs.pop("x_deter", None)
            if x_deter is not None:
                model_output = model(x, t, **model_kwargs) + x_deter
            else:
                model_output = model(x, t, **model_kwargs)
        else:
            model_output = model(x, t, **model_kwargs)
        return model_output

    drift_fn = velocity_ode
    
    def body_fn(x, t, model, **model_kwargs):
        model_output = drift_fn(x, t, model, **model_kwargs)
        assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
        return model_output

    return body_fn
    


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.drift = get_drift()
    
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        drift = self.drift

        t0, t1 = check_interval()

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

