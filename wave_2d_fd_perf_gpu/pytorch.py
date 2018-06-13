import torch
import numpy as np
from wave_2d_fd_perf_gpu.propagators import Propagator


class VPytorch(Propagator):
    """PyTorch implementations."""
    def __init__(self, model, batch_size, pad, dx, dt=None):
        super(VPytorch, self).__init__(model, batch_size, pad, dx, dt)

        self.device = torch.device('cuda:0')

        self.model = torch.tensor(self.model_pad[np.newaxis, np.newaxis])\
                .to(self.device)
        self.wfc = torch.zeros([batch_size, 1, self.nz_padded,
                                self.nx_padded],
                               device=self.device)
        self.wfp = torch.zeros_like(self.wfc)
        self.d2_kernel = (torch.tensor([[0.0,   0.0, -1/12, 0.0, 0.0],
                                        [0.0,   0.0, 4/3,   0.0, 0.0],
                                        [-1/12, 4/3, -10/2,  4/3, -1/12],
                                        [0.0,   0.0, 4/3,   0.0, 0.0],
                                        [0.0,   0.0, -1/12, 0.0, 0.0]]) / dx**2)
        self.d2_kernel = self.d2_kernel.reshape(1, 1, 5, 5).to(self.device)
        torch.backends.cudnn.benchmark = True

    def finalise(self):
        del self.model, self.wfc, self.wfp, self.d2_kernel


class VPytorch1(VPytorch):

    def __init__(self, model, batch_size, dx, dt=None):
        pad = 0
        super(VPytorch1, self).__init__(model, batch_size, pad, dx, dt)

    def step(self, source_amplitude, sources_z, sources_x):

        nb = source_amplitude.shape[0]
        ns = source_amplitude.shape[1]
        nt = source_amplitude.shape[2]
        source_amplitude = torch.tensor(source_amplitude).to(self.device)
        sources_z = torch.tensor(sources_z).to(self.device).long()
        sources_x = torch.tensor(sources_x).to(self.device).long()

        shotidx = torch.arange(nb, device=self.device).reshape(-1, 1)\
                                                      .repeat(1, ns).long()

        for i in range(nt):
            lap = torch.nn.functional.conv2d(self.wfc, self.d2_kernel,
                                             padding=(2, 2))
            self.wfp[:] = self.model * lap + 2 * self.wfc - self.wfp

            self.wfp[shotidx, 0, sources_z, sources_x] += \
                    (source_amplitude[:, :, i]
                     * self.model[0, 0, sources_z, sources_x])

            self.wfc, self.wfp = self.wfp, self.wfc

        return self.wfc[:, 0].cpu().numpy()


class VPytorch2(VPytorch):

    def __init__(self, model, batch_size, dx, dt=None):
        pad = 3
        super(VPytorch2, self).__init__(model, batch_size, pad, dx, dt)
        self.lap = torch.zeros_like(self.wfc)
        self.fd = (torch.tensor([-10/2, 4/3, -1/12]) / dx**2).to(self.device)

    def step(self, source_amplitude, sources_z, sources_x):

        nb = source_amplitude.shape[0]
        ns = source_amplitude.shape[1]
        nt = source_amplitude.shape[2]
        source_amplitude = torch.tensor(source_amplitude).to(self.device)
        sources_z = torch.tensor(sources_z + self.pad).to(self.device).long()
        sources_x = torch.tensor(sources_x + self.pad).to(self.device).long()

        shotidx = torch.arange(nb, device=self.device).reshape(-1, 1)\
                                                      .repeat(1, ns).long()

        for i in range(nt):
            self.lap[:, 0, 2:-2, 2:-2] = (self.fd[0] *
                                          self.wfc[:, 0, 2:-2, 2:-2] +
                                          self.fd[1] *
                                          (self.wfc[:, 0, 2:-2, 3:-1] +
                                           self.wfc[:, 0, 2:-2, 1:-3] +
                                           self.wfc[:, 0, 3:-1, 2:-2] +
                                           self.wfc[:, 0, 1:-3, 2:-2]) +
                                          self.fd[2] *
                                          (self.wfc[:, 0, 2:-2, 4:] +
                                           self.wfc[:, 0, 2:-2, :-4] +
                                           self.wfc[:, 0, 4:, 2:-2] +
                                           self.wfc[:, 0, :-4, 2:-2]))

            self.wfp[:, 0, 2:-2, 2:-2] = (self.model[0, 0, 2:-2, 2:-2]
                                          * self.lap[:, 0, 2:-2, 2:-2]
                                          + 2 * self.wfc[:, 0, 2:-2, 2:-2]
                                          - self.wfp[:, 0, 2:-2, 2:-2])

            self.wfp[shotidx, 0, sources_z, sources_x] += \
                    (source_amplitude[:, :, i]
                     * self.model[0, 0, sources_z, sources_x])

            self.wfc, self.wfp = self.wfp, self.wfc

        return self.wfc[:, 0, self.pad:self.nz_padded-self.pad,
                     self.pad:self.nx_padded-self.pad].cpu().numpy()
