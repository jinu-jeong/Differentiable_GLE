"""Memory field (diffgle.memory_field): exact O step for any d-dependence, knot curves, superposition, layer VACF."""

import numpy as np
import torch

from diffgle import memory_field as mf
from diffgle.constants import DT


def random_field(groups=2, M=4, seed=0, **kw):
    """A memory field with random knot values (every mode positive-real by construction)."""
    g = torch.Generator().manual_seed(seed)
    n = len(mf.KNOTS)
    a = torch.rand(groups, n, M, generator=g, dtype=torch.float64) + 0.05
    g1 = torch.rand(groups, n, M, generator=g, dtype=torch.float64) * 5 + 0.05
    g2 = torch.rand(groups, n, M, generator=g, dtype=torch.float64) * 5 + 0.05
    w = torch.randn(groups, n, M, generator=g, dtype=torch.float64) * 3
    return mf.MemoryField(a, g1, g2, w, groups=groups, **kw)


def test_ostep_maps_preserve_the_distribution():
    """Every tabulated map satisfies T T^T + S S^T = I, i.e. keeps N(0, kT/m I) exactly at every d."""
    th = random_field()
    th.prepare(DT)
    T, S = th._oc[1]                                                 # (P, G, K, K)
    eye = torch.eye(T.shape[-1])
    err = (T @ T.transpose(-1, -2) + S @ S.transpose(-1, -2) - eye).abs().max().item()
    assert err < 1e-5, err


def test_ostep_keeps_temperature_for_moving_beads():
    """Beads at random distances, x ~ N(0, kT/m I): after O steps at new random distances, still N(0, kT/m I)."""
    torch.manual_seed(1)
    mf.set_wall_gap(40.0)
    th = random_field()
    N, m, T = 40000, 18.0, 300.0
    kTm = th.BOLTZMAN * T / m
    mass = torch.full((N, 1), m)
    M = th.n_modes
    v = torch.randn(N, 3) * np.sqrt(kTm)
    th.s = torch.randn(N, 3, 2 * M) * np.sqrt(kTm)
    with torch.no_grad():
        th.prepare(DT)
        for _ in range(5):
            z = torch.rand(N) * 40.0
            v = th.O_step(v, T, DT, mass, z)
    x = torch.cat((v[..., None], th.s), -1)
    var = x.pow(2).mean((0, 1)) / kTm                                 # per component of (v, s)
    assert (var - 1).abs().max().item() < 0.03, var


def test_knot_curves():
    """Values at the knots are reproduced; constant below the first and beyond the last knot."""
    th = random_field()
    k = th.knots
    raw = th.raw_at(k)
    assert torch.allclose(raw, th.values, atol=1e-6)
    assert torch.allclose(th.raw_at(torch.tensor([0.0, 1.0])), th.values[:, :, :1].expand(-1, -1, 2, -1), atol=1e-6)
    assert torch.allclose(th.raw_at(torch.tensor([30.0])), th.values[:, :, -1:], atol=1e-6)
    mid = 0.5 * (k[3] + k[4])
    assert torch.allclose(th.raw_at(mid[None])[:, :, 0], 0.5 * (th.values[:, :, 3] + th.values[:, :, 4]), atol=1e-6)


def test_superposition_equals_one_wall_far_from_the_other():
    """With both walls added, a bead 20 A or more from the far wall gets the one-wall field."""
    one = random_field()
    two = mf.MemoryFieldSuperposed(*[torch.zeros(2, len(mf.KNOTS), 4, dtype=torch.float64) + 0.1] * 4,
                                   groups=2, wall_gap=80.0)
    sd = one.state_dict()
    sd.pop("d_grid")
    two.load_state_dict(sd, strict=False)
    z = torch.linspace(2.5, 60.0, 50)
    for p1, p2 in zip(one.params_at(z), two.params_at(z)):
        assert torch.allclose(p1, p2, atol=1e-5)


def test_layer_vacf_matches_direct_products():
    """layer_vacf_cg (batched form) against the direct sum over origins and beads."""
    torch.manual_seed(2)
    mf.set_wall_gap(20.0)
    Tn, N, td, stride = 300, 50, 40, 7
    vs, zs = torch.randn(Tn, N, 3), torch.rand(Tn, N) * 20.0
    edges = [0.0, 3.0, 6.0, 10.0]
    par, perp, cnt = mf.layer_vacf_cg(vs, zs, edges, td, stride)
    B = len(edges) - 1
    p_ref, z_ref, c_ref = torch.zeros(B, td), torch.zeros(B, td), torch.zeros(B)
    for t0 in range(0, Tn - td + 1, stride):
        b = torch.bucketize(mf.wall_distance(zs[t0]), torch.tensor(edges[1:-1]), right=True)
        prod = vs[t0:t0 + td] * vs[t0]
        for i in range(N):
            p_ref[b[i]] += 0.5 * (prod[:, i, 0] + prod[:, i, 1])
            z_ref[b[i]] += prod[:, i, 2]
            c_ref[b[i]] += 1
    assert torch.allclose(par, p_ref / c_ref[:, None], atol=1e-5)
    assert torch.allclose(perp, z_ref / c_ref[:, None], atol=1e-5)
    mf.set_wall_gap(80.0)
