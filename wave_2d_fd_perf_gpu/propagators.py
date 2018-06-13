"""Propagate a 2D wavefield using different implementations of a
finite difference method so that runtimes can be compared.
"""
import numpy as np

class Propagator(object):
    """A finite difference propagator for the 2D wave equation.
    """
    def __init__(self, model, batch_size, pad, dx, dt=None):

        assert model.ndim == 2
        self.nz = model.shape[0]
        self.nx = model.shape[1]
        self.batch_size = batch_size
        self.pad = pad
        self.dx = dx

        max_vel = np.max(model)
        if dt:
            self.dt = dt
        else:
            self.dt = 0.6 * self.dx / max_vel

        self.nx_padded = self.nx + 2 * pad
        self.nz_padded = self.nz + 2 * pad

        self.model = model**2 * self.dt**2
        self.model_pad = np.pad(self.model, ((pad, pad), (pad, pad)), 'edge')

    def step(self, source_amplitude, sources_z, sources_x):
        """Propagate wavefield."""
        raise NotImplementedError

    def finalise(self):
        """Clean-up when finished with propagator."""
        pass
