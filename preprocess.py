"""data/raw/<system>/aa (all-atom trajectories) -> data/processed/<system> for the bulk systems H2O and CO2.

The water-graphene systems are processed by experiments/confined/allatom/make_dataset.py.
"""

import argparse
import gc
import os
import shutil

import numpy as np
import torch

from diffgle.constants import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

BULK_SYSTEMS = {"CO2", "H2O"}
# CG pair potentials copied into data/processed/<system>/CG/
CG_SOURCE = {name: os.path.join(PROJECT_ROOT, "forcefields", "cg", name) for name in BULK_SYSTEMS}

BULK_BOX = np.array([40.0, 40.0, 40.0])

# Use CPU when a single trajectory array would exceed this share of free GPU memory.
GPU_MEMORY_FRACTION = 0.35


def parse_args():
    parser = argparse.ArgumentParser(description="data/raw/<system>/aa -> data/processed/<system>")
    parser.add_argument("--system", default=None, help="Process one system only (default: all)")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu; auto-falls back to cpu for large trajectories on cuda)",
    )
    parser.add_argument(
        "--rdf-stride",
        type=int,
        default=10,
        help="Frame stride for RDF / density (default: 10)",
    )
    return parser.parse_args()


def make_cell(box, device):
    return torch.diag(torch.tensor(box, dtype=torch.float, device=device))


def release_memory(device):
    gc.collect()
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_device(requested_device, nbytes):
    if requested_device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if not requested_device.startswith("cuda"):
        return requested_device

    free_bytes, _ = torch.cuda.mem_get_info()
    if nbytes > free_bytes * GPU_MEMORY_FRACTION:
        print(f"  large trajectory ({nbytes / 1e9:.1f} GB) -> using cpu")
        return "cpu"
    return requested_device


def to_tensor(array, device):
    return torch.tensor(np.asarray(array), device=device)


def preprocess_bulk(system_name, aa_dir, out_dir, device, rdf_stride):
    from diffgle.observables import MSD_computer, RDF_computer, VACF_computer

    pos_path = os.path.join(aa_dir, "pos_COM.npy")
    vel_path = os.path.join(aa_dir, "vel_COM.npy")
    X_np = np.load(pos_path, mmap_mode="r")
    V_np = np.load(vel_path, mmap_mode="r")

    nbytes = X_np.nbytes + V_np.nbytes
    device = pick_device(device, nbytes)
    cell = make_cell(BULK_BOX, device)

    np.save(os.path.join(out_dir, "pos0.npy"), np.asarray(X_np[-1]) % BULK_BOX[0])
    np.save(os.path.join(out_dir, "vel0.npy"), np.asarray(V_np[-1]))
    np.save(os.path.join(out_dir, "box.npy"), BULK_BOX)

    RDF = RDF_computer(cell, device)
    MSD = MSD_computer(1000)
    VACF = VACF_computer(1000)
    VACF.ensemble_average = True
    VACF.normalize = True

    with torch.no_grad():
        X_rdf = to_tensor(X_np[::rdf_stride], device)
        r_aa, rdf_aa = RDF(X_rdf)
        np.save(os.path.join(out_dir, "r_aa.npy"), r_aa.cpu().numpy())
        np.save(os.path.join(out_dir, "rdf_aa.npy"), rdf_aa.cpu().numpy())
        del X_rdf, r_aa, rdf_aa

        X = to_tensor(X_np, device)
        msd_aa = MSD(X)
        np.save(os.path.join(out_dir, "msd_aa.npy"), msd_aa.cpu().numpy())
        del X, msd_aa

        V = to_tensor(V_np, device)
        vacf_aa = VACF(V)
        np.save(os.path.join(out_dir, "vacf_aa.npy"), vacf_aa.cpu().numpy())
        del V, vacf_aa

    release_memory(device)
    print("  saved bulk preprocess: rdf, msd, vacf, pos0, vel0")


def preprocess_system(system_name, device, rdf_stride):
    aa_dir = os.path.join(RAW_DIR, system_name, "aa")
    data_dir = os.path.join(PROCESSED_DIR, system_name)
    if not os.path.isdir(aa_dir):
        print(f"skip {system_name}: missing {aa_dir}")
        return
    os.makedirs(data_dir, exist_ok=True)
    if system_name not in BULK_SYSTEMS:
        print(f"skip {system_name}: not a bulk system (water-graphene: experiments/confined/allatom/make_dataset.py)")
        return
    shutil.copytree(CG_SOURCE[system_name], os.path.join(data_dir, "CG"), dirs_exist_ok=True)

    print(f"preprocessing {system_name}...")
    preprocess_bulk(system_name, aa_dir, data_dir, device, rdf_stride)


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.system:
        preprocess_system(args.system, args.device, args.rdf_stride)
        return

    for name in sorted(BULK_SYSTEMS):
        preprocess_system(name, args.device, args.rdf_stride)


if __name__ == "__main__":
    main()
