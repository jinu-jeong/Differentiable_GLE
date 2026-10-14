"""ConvGLE.backward_adjoint (hand-written discrete adjoint) == plain autograd BPTT gradient."""

import torch

from diffgle.constants import BOLTZMAN
from diffgle.thermostats import ConvGLE


def check(paper_detach):
    """Compare the hand-written adjoint to plain autograd on a small random problem."""
    torch.manual_seed(0)
    L, Wn, D, dt = 6, 25, 4, 0.1
    mass = torch.ones(1, 1)
    model = ConvGLE(torch.rand(L) * 0.3, mass, dt, 1 / BOLTZMAN, paper_detach=paper_detach)
    model.v_hist = torch.randn(L - 1, D)
    model.w_hist = torch.randn(L - 1, D)
    Vall_pre, W_all = model.v_hist.clone(), torch.cat((model.w_hist, torch.randn(Wn, D)))
    A = torch.randn(Wn, D)
    target = torch.randn(Wn + 1, D)

    # autograd reference
    vs = list(Vall_pre) + [torch.randn(D)]
    v0 = vs[-1].clone()
    K, h = model.kernel(), model.h
    for n in range(Wn):
        hist = torch.stack(vs[n:n + L])
        fric = -(K.flip(0) @ hist)
        rand = (h.detach() if paper_detach else h).flip(0) @ W_all[n:n + L]
        vs.append(vs[L - 1 + n] + (A[n] + fric + rand) * dt)
    V = torch.stack(vs[L - 1:])
    loss = ((V - target) ** 2).sum() + (V[5] * V[-1]).sum()
    g_ref = torch.autograd.grad(loss, model.h)[0]

    # adjoint
    Vd = V.detach().requires_grad_(True)
    loss_d = ((Vd - target) ** 2).sum() + (Vd[5] * Vd[-1]).sum()
    gV = torch.autograd.grad(loss_d, Vd)[0]
    Vall = torch.cat((Vall_pre, V.detach()[:-1]))
    model.h.grad = None
    model.backward_adjoint(gV, Vall, W_all)
    err = (model.h.grad - g_ref).abs().max() / g_ref.abs().max()
    print(f"conv adjoint vs autograd (paper_detach={paper_detach}): max rel err {err:.2e}")
    assert err < 1e-4


def test_conv_adjoint_matches_autograd():
    check(paper_detach=False)


def test_conv_adjoint_paper_detach_matches_autograd():
    check(paper_detach=True)


if __name__ == "__main__":
    test_conv_adjoint_matches_autograd()
    test_conv_adjoint_paper_detach_matches_autograd()
