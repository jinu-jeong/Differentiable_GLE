import argparse
import os
import pickle
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint_adjoint
from torchmd.systems import System

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
DEFAULT_SYSTEM = "Confined_H2O"


class GLE_TS(torch.nn.Module):
    def __init__(self, gamma_uniform=0.001, filter_length=None, BOLTZMAN=0.001987191):
        super(GLE_TS, self).__init__()
        self.BOLTZMAN = BOLTZMAN
        self.filter_length = filter_length
        dummy = torch.full((self.filter_length,), gamma_uniform)
        self.h = torch.nn.Parameter(dummy[None, None, :])
        self.v_list = None
        self.w_list = None
        self.w_history = []

    def construct_memory(self, T, mass, dt):
        h_padded = torch.nn.functional.pad(self.h, (0, self.h.size(2) - 1)).detach()
        self.theoretical_RACF = torch.nn.functional.conv1d(h_padded, self.h)
        self.memory_kernel = self.theoretical_RACF * mass[0, 0].item() * dt / (self.BOLTZMAN * T)
        self.memory_kernel_trapezoidal = self.memory_kernel.clone()
        self.memory_kernel_trapezoidal[..., 0] *= 0.5
        self.memory_kernel_trapezoidal[..., -1] *= 0.5

    def get_v_and_sample_w(self, v, dt):
        if self.v_list is None:
            n = len(v.flatten()) // 3
            self.v_list = torch.zeros(n * 3, 1, self.filter_length * 2 + 1, device=v.device)
            self.w_list = torch.zeros(n * 3, 1, self.filter_length * 2 + 1, device=v.device)

        self.v_list = torch.roll(self.v_list, -1, dims=2)
        self.v_list[:, :, -1] = v.view(-1, 1).clone().detach()
        self.w_list = torch.roll(self.w_list, -1, dims=2)
        self.w_list[:, :, -1] = torch.randn_like(v, device=v.device).view(-1, 1)

    def forward(self, v, T, dt, mass, t):
        self.construct_memory(T, mass, dt)
        self.get_v_and_sample_w(v, dt)
        friction = -torch.nn.functional.conv1d(self.v_list[:, :, -self.filter_length:], self.memory_kernel_trapezoidal.flip(2))
        random = torch.nn.functional.conv1d(self.w_list[:, :, -self.filter_length:], self.h.flip(2).detach())
        delta_v_langevin = friction.view(v.shape) + random.view(v.shape)
        if len(self.w_history) == self.filter_length * 2:
            self.w_history.pop(0)
        self.w_history.append(random.clone().detach().cpu().numpy().flatten())
        return delta_v_langevin

    @contextmanager
    def adjoint_backward(self):
        try:
            yield
        finally:
            N = self.filter_length
            self.v_list[:, :, -N:] = self.v_list[:, :, -N:].flip(2)
            self.w_list[:, :, -N:] = self.w_list[:, :, -N:].flip(2)

    def verify_filter_convolution(self, show_plot):
        data = np.array(self.w_history)
        RACF = []
        td = self.filter_length + 1
        for t0 in range(0, len(data) - td, 10):
            RACF.append((data[t0:t0 + td] * data[t0]).mean(1))
        RACF = np.mean(RACF, axis=0)

        for data_line, title, path in [
            (self.h.clone().detach().cpu().flatten(), "Filter", "filter.png"),
            (self.memory_kernel.clone().detach().cpu().flatten(), "Memory Kernel", "memory_kernel.png"),
            (None, "Theoretical RACF", "racf.png"),
        ]:
            plt.figure()
            if data_line is not None:
                plt.plot(data_line, 'bo--')
            else:
                plt.plot(self.theoretical_RACF.clone().detach().cpu().flatten(), 'bo--')
                plt.plot(RACF, 'ro--')
            plt.title(title)
            show_plot(path)
            plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="DiffGLE confined water demo with transfer learning")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="Data directory name")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--iterations", "-n", type=int, default=300, help="Optimization iterations")
    parser.add_argument(
        "--pretrained",
        default=os.path.join(PROJECT_ROOT, "Result", "H2O", "model_state_dict.pth"),
        help="Pretrained bulk H2O GLE kernel",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Skip transfer learning init")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: ./Result/<system>)")
    parser.set_defaults(no_show=True)
    parser.add_argument("--show", dest="no_show", action="store_false", help="Display plots interactively")
    return parser.parse_args()


def make_show_plot(args, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def show_plot(filename=None):
        if filename and filename.startswith("optimization_iter_"):
            plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        if not args.no_show:
            plt.show()

    return show_plot


def run(args):
    os.chdir(PROJECT_ROOT)

    system_name = args.system
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "Result", system_name)
    show_plot = make_show_plot(args, output_dir)

    mass_value = 18.01528
    filter_length = 1000
    pre_equil_T = 2000

    device = args.device
    data_dir = os.path.join(DATA_DIR, system_name)
    z = torch.tensor(np.load(os.path.join(data_dir, "z.npy")), device=device)
    pos0 = np.load(os.path.join(data_dir, "pos0.npy"))
    vel0 = np.load(os.path.join(data_dir, "vel0.npy"))
    box = np.load(os.path.join(data_dir, "box.npy"))
    z_aa = np.load(os.path.join(data_dir, "z_aa.npy"))
    density_aa = np.load(os.path.join(data_dir, "density_aa.npy"))
    msd_aa = np.load(os.path.join(data_dir, "msd_aa.npy"))
    vacf_aa = np.load(os.path.join(data_dir, "vacf_aa.npy"))
    vacf_aa_gt = np.load(os.path.join(data_dir, "vacf_aa_gt.npy"))

    Natom = pos0.shape[0]
    precision = torch.float
    mass = torch.full((Natom, 1), mass_value).to(device)
    TIMEFACTOR = 48.88821
    dt = 1

    md_system = System(Natom, nreplicas=1, precision=precision, device=device)
    md_system.set_positions(torch.tensor(pos0, device=device, dtype=precision)[:, :, None])
    md_system.set_box(box)
    md_system.set_velocities(torch.tensor(vel0, device=device, dtype=precision)[None])

    from force import Langevin_TS, Tabulated_specific
    from utility import MSD_computer, SDE, VACF_computer, atom_types_map, density_computer, write_xyz_dump

    Tabulated_data_FF = np.loadtxt(os.path.join(data_dir, "CG", "CG1_CG1.pot"), skiprows=3)
    Tabulated_data_FF = Tabulated_data_FF[~np.isnan(Tabulated_data_FF[:, -1])]
    Tabulated_data_WF = np.loadtxt(os.path.join(data_dir, "CG", "CG1_GR1.pot"), skiprows=3)
    Tabulated_data_WF = Tabulated_data_WF[~np.isnan(Tabulated_data_WF[:, -1])]

    potential_GT = Tabulated_specific(
        md_system.box[0], z, [Tabulated_data_FF, Tabulated_data_WF], [[7, 7], [7, 5]]
    ).to(device)

    Thermostat_LE = Langevin_TS(gamma=0.5).to(device)
    func = SDE(
        potential_GT,
        Thermostat=Thermostat_LE,
        Temp_target=300,
        timestep=dt,
        TIMEFACTOR=TIMEFACTOR,
        mass=mass,
        non_integrand_mask=(z == 5),
        saver=False,
    )
    func.force_mode = True

    q, cell, v = md_system.pos[0], md_system.box[0], md_system.vel[0] * 0
    y0 = torch.concatenate((v, q))
    func(0., y0)

    T = 1000
    y0 = torch.concatenate((v, q))
    t = torch.tensor(np.arange(0, dt / TIMEFACTOR * T, dt / TIMEFACTOR)).to(device)

    Density = density_computer(cell, z_mask=(z == 7), axis=2, device=device, dr=0.5)
    MSD = MSD_computer(500)
    VACF = VACF_computer(500)

    with torch.no_grad():
        y_vanilla = odeint_adjoint(func, y0, t, method="euler")
        y0 = y_vanilla[-1].detach()

    z_GT = torch.tensor(z_aa, device=device)
    density_GT = torch.tensor(density_aa, device=device)
    z_vanilla, density_vanilla = Density(y_vanilla[500::10, Natom:])
    plt.figure()
    plt.plot(z_vanilla.detach().cpu(), density_vanilla.detach().cpu(), 'r')
    plt.plot(z_GT.detach().cpu(), density_GT.detach().cpu(), 'k--')
    show_plot("density_vanilla.png")
    plt.close()

    MSD_GT = torch.tensor(msd_aa, device=device)
    MSD_vanilla = MSD(y_vanilla[5000:, Natom:][:, z == 7])
    plt.figure()
    plt.plot(MSD_vanilla.detach().cpu(), 'r')
    plt.plot(MSD_GT.detach().cpu(), 'k--')
    show_plot("msd_vanilla.png")
    plt.close()

    VACF_GT = torch.tensor(vacf_aa_gt, device=device)
    VACF_vanilla = VACF(y_vanilla[5000:, :Natom][:, z == 7])
    plt.figure()
    plt.plot(VACF_vanilla.detach().cpu(), 'r')
    plt.plot(VACF_GT.detach().cpu(), 'k--')
    show_plot("vacf_vanilla.png")
    plt.close()

    Thermostat_GLE = GLE_TS(gamma_uniform=0.001, filter_length=filter_length).to(device)
    func = SDE(
        potential_GT,
        Thermostat=Thermostat_GLE,
        Temp_target=300,
        timestep=dt,
        TIMEFACTOR=TIMEFACTOR,
        mass=mass,
        non_integrand_mask=(z == 5),
        saver=False,
    )
    func.force_mode = True

    if not args.no_pretrained and os.path.exists(args.pretrained):
        func.load_state_dict(torch.load(args.pretrained, map_location=device), strict=False)
        print("Transfer learning from", args.pretrained)
    elif not args.no_pretrained:
        print("Pretrained weights not found at", args.pretrained)

    q, cell, v = md_system.pos[0], md_system.box[0], md_system.vel[0]
    y0 = torch.concatenate((v * 0, q))
    func(0., y0)
    Thermostat_GLE.verify_filter_convolution(show_plot)

    T = 3000
    t = torch.tensor(np.arange(0, dt / TIMEFACTOR * T, dt / TIMEFACTOR)).to(device)
    y0 = torch.concatenate((v, q))
    with torch.no_grad():
        y_GLE = odeint_adjoint(func, y0, t, method="euler")
    y0 = y_GLE[-1]

    z_GLE, density_GLE = Density(y_GLE[::10, Natom:])
    MSD_GLE = MSD(y_GLE[:, Natom:])
    VACF_GLE = VACF(y_GLE[:, :Natom])

    plt.figure()
    plt.plot(z_GT.cpu(), density_GT.cpu(), 'k')
    plt.plot(z_GLE.cpu(), density_GLE.cpu(), 'r')
    show_plot("density_compare.png")
    plt.close()
    plt.figure()
    plt.plot(MSD_vanilla.cpu(), 'k')
    plt.plot(MSD_GLE.cpu(), 'r')
    show_plot("msd_compare.png")
    plt.close()
    plt.figure()
    plt.plot(VACF_vanilla.cpu(), 'k')
    plt.plot(VACF_GLE.cpu(), 'r')
    show_plot("vacf_compare.png")
    plt.close()

    VACF = VACF_computer(1000)
    VACF.ensemble_average = True
    VACF.normalize = True
    VACF_AA = torch.tensor(vacf_aa, device=device)

    optimizer = torch.optim.Adam(func.parameters(), lr=1e-5, weight_decay=1e-6)

    for iter in range(args.iterations):

        T = pre_equil_T
        t = torch.tensor(np.arange(0, dt / TIMEFACTOR * T, dt / TIMEFACTOR)).to(device)
        with torch.no_grad():
            y_curr = odeint_adjoint(func, y0, t, method="euler")
        y0 = y_curr[-1].detach()

        T = max(Thermostat_GLE.filter_length, VACF.td_max) + 10
        t = torch.tensor(np.arange(0, dt / TIMEFACTOR * T, dt / TIMEFACTOR)).to(device)
        y_curr = odeint_adjoint(func, y0, t, method="euler")
        y0 = y_curr[Thermostat_GLE.filter_length + 1].detach()

        VACF.ensemble_average = True
        VACF.normalize = True
        VACF_curr = VACF(y_curr[:, :Natom][:, z == 7])
        loss = (VACF_curr - VACF_AA).pow(2).sum()

        optimizer.zero_grad()
        with Thermostat_GLE.adjoint_backward():
            loss.backward(retain_graph=True)
        optimizer.step()

        print(iter, loss.item())

        plt.figure(figsize=(10, 8))
        plt.subplot(4, 1, 1)
        plt.plot(VACF_curr.detach().cpu().numpy(), 'r')
        plt.plot(VACF_AA.detach().cpu().numpy(), 'k--')
        plt.subplot(4, 1, 2)
        filter_vals = Thermostat_GLE.h.clone().detach().cpu().flatten()
        plt.plot(filter_vals, 'r')
        plt.subplot(4, 1, 3)
        plt.plot(Thermostat_GLE.h.grad.clone().detach().cpu().flatten(), 'g--')
        plt.subplot(4, 1, 4)
        plt.plot(Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten(), 'r')
        show_plot(f"optimization_iter_{iter:03d}.png")
        plt.close()

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model_state_dict.pth")
    torch.save(func.state_dict(), model_path)

    plot_dict = {
        "r": z_GT.cpu().numpy(),
        "RDF_AA": density_GT.cpu().numpy(),
        "RDF_CG": density_GLE.cpu().numpy(),
        "VACF_GLE": VACF_curr.detach().cpu().numpy(),
        "VACF_AA": VACF_AA.detach().cpu().numpy(),
        "filter": filter_vals.numpy(),
        "filter_gradient": Thermostat_GLE.h.grad.clone().detach().cpu().flatten().numpy(),
        "memory_kernel": Thermostat_GLE.memory_kernel.clone().detach().cpu().flatten().numpy(),
    }
    with open(os.path.join(output_dir, "plot_dict.pkl"), 'wb') as f:
        pickle.dump(plot_dict, f)

    T = 10000
    t = torch.tensor(np.arange(0, dt / TIMEFACTOR * T, dt / TIMEFACTOR)).to(device)
    with torch.no_grad():
        y_curr = odeint_adjoint(func, y0, t, method="euler")

    VACF_curr = VACF(y_curr[:, :Natom][:, z == 7])
    plt.figure()
    plt.plot(VACF_curr.detach().cpu().numpy(), 'r')
    plt.plot(VACF_AA.detach().cpu().numpy(), 'k--')
    show_plot("vacf_production.png")
    plt.close()

    z_GLE, density_GLE = Density(y_curr[:, Natom:])
    plt.figure()
    plt.plot(z_GT.cpu(), density_GT.cpu(), 'k')
    plt.plot(z_GLE.cpu(), density_GLE.cpu(), 'r')
    plt.xlim([0, 80])
    show_plot("density_production.png")
    plt.close()

    xyz_path = os.path.join(output_dir, f"{system_name}.xyz")
    write_xyz_dump(xyz_path, z, atom_types_map, y_curr[:, Natom:, :], cell)
    print(f"Saved model to {model_path}")
    print(f"Saved trajectory to {xyz_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.no_show and os.environ.get("MPLBACKEND") is None:
        import matplotlib
        matplotlib.use("Agg")
    run(args)
