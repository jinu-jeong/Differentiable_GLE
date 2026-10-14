"""Exact stationary statistics of the discrete integrators on a harmonic well (linear maps -> discrete Lyapunov).

Run from the repo root:  python -m experiments.check_integrator_bias

Reports <v^2>/(kT/m) (kinetic temperature ratio) and <q^2> w^2/(kT/m) (configurational ratio); exact = 1.
"""
import os
import pickle

import numpy as np
import torch
from scipy.linalg import cholesky, expm, solve_discrete_lyapunov

from diffgle.constants import DT as dt, RESULT_DIR, TIMEFACTOR

fs = lambda P: 2 * np.pi / (P / TIMEFACTOR)  # angular freq (sim units) for period P fs


def stationary(M, B):
    S = solve_discrete_lyapunov(M, B @ B.T)
    return S[1, 1], S[0, 0]  # (v^2, q^2) with state (q, v, ...)


# ---------- Markov Langevin, m = kT = 1 ----------
def markov_current(w, g):
    # q' = q + dt v + dt^2/2 a(q);  v' = v + dt/2 (a(q) + a(q')) - dt g v + sqrt(2 g dt) xi
    Mq = np.array([1 - 0.5 * dt**2 * w**2, dt])
    q1 = Mq  # q' as a row in (q, v)
    Mv = np.array([-0.5 * dt * w**2, 1 - dt * g]) - 0.5 * dt * w**2 * q1
    return np.vstack((q1, Mv)), np.array([[0.0], [np.sqrt(2 * g * dt)]])


def markov_baoab(w, g):
    B = np.array([[1, 0], [-0.5 * dt * w**2, 1]]); A = np.array([[1, 0.5 * dt], [0, 1]])
    c = np.exp(-g * dt); O = np.diag([1, c])
    M = B @ A @ O @ A @ B
    noise = B @ A @ np.array([[0.0], [np.sqrt(1 - c**2)]])
    return M, noise


# ---------- damped-osc GLE (trained params) ----------
def load_osc_kernel(run="H2O_osc"):
    """Block-diagonal drift matrix A and coupling c of a trained damped-osc thermostat."""
    d = pickle.load(open(os.path.join(RESULT_DIR, run, "plot_dict.pkl"), "rb"))
    a, g1, g2, om = (np.asarray(d[k], float) for k in ["a", "gamma1", "gamma2", "omega"])
    Ablk = np.zeros((2 * len(a), 2 * len(a))); cvec = np.zeros(2 * len(a))
    for k in range(len(a)):
        Ablk[2*k:2*k+2, 2*k:2*k+2] = [[g1[k], om[k]], [-om[k], g2[k]]]
        cvec[2 * k] = np.sqrt(a[k])
    return Ablk, cvec


def osc_current(w, Ablk, cvec):
    n_s, Mmodes = len(cvec), len(cvec) // 2
    # VV_GLE: s' = P s - g v + L xi ; thermo = c.s' ; q' = q + dt v + dt^2/2 a ; v' = v + dt/2(a+a') + dt thermo
    aug = np.zeros((n_s + 1, n_s + 1)); aug[:n_s, :n_s] = -Ablk * dt; aug[:n_s, n_s] = cvec * dt
    E = expm(aug); P, gin = E[:n_s, :n_s], E[:n_s, n_s]
    Lc = np.zeros((n_s, n_s))
    for k in range(Mmodes):
        blk = P[2*k:2*k+2, 2*k:2*k+2]
        Lc[2*k:2*k+2, 2*k:2*k+2] = cholesky(np.eye(2) - blk @ blk.T + 1e-15 * np.eye(2), lower=True)
    n = 2 + n_s  # (q, v, s)
    M = np.zeros((n, n)); Bn = np.zeros((n, n_s))
    M[2:, 2:] = P; M[2:, 1] = -gin; Bn[2:, :] = Lc
    M[0] = [1 - 0.5 * dt**2 * w**2, dt] + [0] * n_s
    M[1] = np.r_[-0.5 * dt * w**2, 1, np.zeros(n_s)] - 0.5 * dt * w**2 * M[0] + dt * cvec @ M[2:]
    Bn[1] = dt * cvec @ Bn[2:]
    return M, Bn


def osc_baoab(w, Ablk, cvec):
    n_s = len(cvec)
    # B A O A B with exact O on the linear (v, s) block: d(v,s) = -Ap (v,s) dt + noise, stationary cov = I
    n = 2 + n_s
    Ap = np.zeros((1 + n_s, 1 + n_s)); Ap[0, 1:] = -cvec; Ap[1:, 0] = cvec; Ap[1:, 1:] = Ablk
    T = expm(-Ap * dt)
    S = cholesky(np.eye(1 + n_s) - T @ T.T + 1e-14 * np.eye(1 + n_s), lower=True)
    B = np.eye(n); B[1, 0] = -0.5 * dt * w**2
    A = np.eye(n); A[0, 1] = 0.5 * dt
    O = np.eye(n); O[1:, 1:] = T
    Nz = np.zeros((n, 1 + n_s)); Nz[1:, :] = S
    return B @ A @ O @ A @ B, B @ A @ Nz


# ---------- conv GLE_TS (VVEulerRHS + Euler, as in main.py) ----------
def conv_current(w, h, mass=18.0, T=300.0, kB=0.001987191):
    h = np.asarray(h, float).ravel(); L = len(h)
    racf = np.array([h[j:] @ h[:L - j] for j in range(L)])
    K = racf * mass * dt / (kB * T)
    n = 2 + 2 * (L - 1)  # q, v_n, v_{n-1..n-L+1}, w_{n-1..n-L+1}
    iv, iw = 2, 2 + (L - 1)
    M = np.zeros((n, n)); Bn = np.zeros((n, 1))
    M[0, 0], M[0, 1] = 1 - 0.5 * dt**2 * w**2, dt
    M[1, 0] = -0.5 * dt * w**2 * (2 - 0.5 * dt**2 * w**2)
    M[1, 1] = 1 - 0.5 * dt**2 * w**2 - dt * K[0]
    M[1, iv:iw] = -dt * K[1:]
    M[1, iw:] = dt * h[1:]; Bn[1, 0] = dt * h[0]
    M[iv, 1] = 1; M[iv + 1:iw, iv:iw - 1] = np.eye(L - 2)
    Bn[iw, 0] = 1; M[iw + 1:, iw:n - 1] = np.eye(L - 2)
    v2, q2 = stationary(M, Bn)
    kT_m = kB * T / mass
    return v2 / kT_m, q2 * w**2 / kT_m


if __name__ == "__main__":
    print("== Markov Langevin: <v2>/(kT/m), <q2>w2/(kT/m)")
    for P in [60, 120, 300]:
        for gdt in [0.02, 0.1, 0.5]:
            w, g = fs(P), gdt / dt
            c = stationary(*markov_current(w, g)); b = stationary(*markov_baoab(w, g))
            print(f"  period {P:3d} fs  g*dt {gdt:4.2f} | current v {c[0]:.4f} q {c[1] * w**2:.4f} | BAOAB v {b[0]:.4f} q {b[1] * w**2:.4f}")
    kern = load_osc_kernel()
    print("== damped-osc GLE (trained H2O params)")
    for P in [60, 120, 300]:
        w = fs(P)
        c = stationary(*osc_current(w, *kern)); b = stationary(*osc_baoab(w, *kern))
        print(f"  period {P:3d} fs | VV_GLE v {c[0]:.4f} q {c[1] * w**2:.4f} | GLE-BAOAB v {b[0]:.4f} q {b[1] * w**2:.4f}")
    print("== conv GLE_TS + VVEulerRHS (Euler-baked VV, main.py)")
    for name, path in [("adjoint-trained", "H2O/model_state_dict.pth"), ("BPTT lr3e-4", "H2O_conv_bptt_lr0.0003/model_state_dict.pth")]:
        h = torch.load(os.path.join(RESULT_DIR, path), map_location="cpu")["Thermostat.h"].numpy()
        for P in [60, 120, 300]:
            v, q = conv_current(fs(P), h)
            print(f"  {name:16s} period {P:3d} fs | v {v:.4f} q {q:.4f}")
