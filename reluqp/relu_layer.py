
import torch
import numpy as np
from reluqp.classes import Settings, Results, Info, QP


class ReLU_Layer(torch.nn.Module):
    def __init__(self, QP=None, settings=Settings()):
        super(ReLU_Layer, self).__init__()

        torch.set_default_dtype(settings.precision)
        self.QP = QP
        self.settings = settings
        self.rhos = self.setup_rhos()
        
        self.W_ks = self.setup_matrices()
        self.clamp_inds = (self.QP.nx, self.QP.nx + self.QP.nc)
        self.setup_quantization()

    def setup_rhos(self):
        """
        Setup rho values for ADMM
        """
        stng = self.settings
        rhos = [stng.rho]
        if stng.adaptive_rho:
            rho = stng.rho/stng.adaptive_rho_tolerance
            while rho >= stng.rho_min:
                rhos.append(rho)
                rho = rho/stng.adaptive_rho_tolerance
            rho = stng.rho*stng.adaptive_rho_tolerance
            while rho <= stng.rho_max:
                rhos.append(rho)
                rho = rho*stng.adaptive_rho_tolerance
            rhos.sort()
        # conver to torch tensor
        rhos = torch.tensor(rhos, device=stng.device, dtype=stng.precision).contiguous()
        return rhos
    
    def setup_quantization(self, n_int=17):
        n_frac = 35 - n_int
        self.n_int = n_int
        self.n_frac = n_frac
        self.scale = 2 ** n_frac
        self.max_val = 2 ** (n_int + n_frac - 1) - 1
        self.min_val = -self.max_val - 1

        # Ensure everything is on the same device and dtype
        stng = self.settings
        device = self.QP.H.device if self.QP and hasattr(self.QP.H, 'device') else torch.device(stng.device)
        dtype = self.QP.H.dtype if self.QP and hasattr(self.QP.H, 'dtype') else stng.precision

        self.scale = torch.as_tensor(self.scale, device=device, dtype=dtype)
        self.min_val = torch.as_tensor(self.min_val, device=device, dtype=dtype)
        self.max_val = torch.as_tensor(self.max_val, device=device, dtype=dtype)

    def q(self, v: torch.Tensor) -> torch.Tensor:
        # Scale input
        scaled = v * self.scale

        # Check for out-of-range values (keep on GPU if possible)
        if torch.any((scaled < self.min_val) | (scaled > self.max_val)):
            raise RuntimeError("Quantization overflow: some values outside representable range")

        qv = torch.round(scaled)

        # Return dequantized tensor
        return qv / self.scale


    def setup_matrices(self):
        """
        Setup ADMM matrices for ReLU-QP solver for each rho
        """
        # unpack values
        H, g, A, l, u = self.QP.H, self.QP.g, self.QP.A, self.QP.l, self.QP.u
        nx, nc = self.QP.nx, self.QP.nc
        sigma = self.settings.sigma
        stng = self.settings

        # Calculate kkt_rhs_invs
        kkt_rhs_invs = []
        for rho_scalar in self.rhos:
            rho = rho_scalar * torch.ones(nc).to(g)
            rho[(u - l) <= stng.eq_tol] = rho_scalar * 1e3
            rho = torch.diag(rho)
            kkt_rhs_invs.append(torch.inverse(H + sigma * torch.eye(nx).to(g) + A.T @ (rho @ A)))

        W_ks = {}

        
        # Other layer updates for each rho
        for rho_ind, rho_scalar in enumerate(self.rhos):
            rho = rho_scalar * torch.ones(nc, device=stng.device, dtype=stng.precision).contiguous()
            rho[(u - l) <= stng.eq_tol] = rho_scalar * 1e3
            rho_inv = torch.diag(1.0 / rho)
            rho = torch.diag(rho).to(device=stng.device, dtype=stng.precision).contiguous()
            K = kkt_rhs_invs[rho_ind]
            Ix = torch.eye(nx, device=stng.device, dtype=stng.precision).contiguous()
            Ic = torch.eye(nc, device=stng.device, dtype=stng.precision).contiguous()
            W_ks[rho_ind] = torch.cat([
                torch.cat([ K @ (sigma * Ix - A.T @ (rho @ A)),           2 * K @ A.T @ rho,            -K @ A.T], dim=1),
                torch.cat([ A @ K @ (sigma * Ix - A.T @ (rho @ A)) + A,   2 * A @ K @ A.T @ rho - Ic,  -A @ K @ A.T + rho_inv], dim=1),
                torch.cat([ rho @ A,                                      -rho,                         Ic], dim=1)
            ], dim=0).contiguous()
        return W_ks

    def forward(self, input, idx):
        input = self.q(input)
        input = self.jit_forward(input, self.W_ks[idx], self.QP.l, self.QP.u, self.clamp_inds[0], self.clamp_inds[1])
        return input
    
    @torch.jit.script
    def jit_forward(input, W, l, u, idx1: int, idx2: int):
        torch.matmul(W, input, out=input)
        input[idx1:idx2].clamp_(l, u)
        return input
    
