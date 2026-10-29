"""
battery_pinn.py  –  PINN parameter tracking for the NASA battery dataset.

Pipeline (see README diagram):
    metadata.csv + data/*.csv
        → soc_ocv.load_test / add_soc / build_ocv_curve      (load + preprocess)
        → prepare_cycle()        tensors for one cycle (optionally one SOC window)
        → BatteryPINN            1st-order Thevenin ECM, learnable log R0 / R1 / C1
        → fit_cycle()            train, return {R0, R1, C1, tau, losses}
        → run_tracking()         loop cells × cycles × SOC windows → params.csv
        → overfit_contrast()     lambda_physics = 0 vs > 0 on held-out data → contrast.csv

Run from the Practice/ folder:
    python battery_pinn.py --batteries B0005 B0006 --cycle-step 20 --windows 4 --epochs 1500
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from soc_ocv import load_meta, load_test, add_soc, build_ocv_curve, ocv_lookup

OUT = Path(__file__).resolve().parent / "results"
torch.manual_seed(0)


# ============================================================================ 1. data
def prepare_cycle(meta, battery, cycle, ocv_table, soc_window=(0.0, 1.0)):
    """
    Turn one discharge test into the tensors the PINN needs.

    Returns a dict with:
        inputs  (N,3)  [t_norm, I, temp_norm]   t_norm = t / t_scale  in 0..1
        voc     (N,1)  open-circuit voltage from the lookup table
        v       (N,1)  measured terminal voltage
        t_scale        seconds, so d/dt_real = d/dt_norm / t_scale
        df             the pandas table (for plotting)
    """
    soc_grid, ocv_grid = ocv_table
    df = add_soc(load_test(meta, battery, "discharge", cycle), "discharge")
    df = df[df["Current_load"].abs() > 0.1]                       # only while current flows
    lo, hi = soc_window
    df = df[(df["SOC"] >= lo) & (df["SOC"] <= hi)].reset_index(drop=True)
    df["OCV"] = ocv_lookup(df["SOC"], soc_grid, ocv_grid)

    t = df["Time"].to_numpy(np.float32)
    t = t - t[0]
    t_scale = float(max(t[-1], 1.0))
    col = lambda a: torch.tensor(np.asarray(a, np.float32)).unsqueeze(1)
    inputs = torch.cat([col(t / t_scale),
                        col(df["Current_load"].abs()),
                        col((df["Temperature_measured"] - 25.0) / 10.0)], dim=1)
    return dict(inputs=inputs, voc=col(df["OCV"]), v=col(df["Voltage_measured"]),
                t_scale=t_scale, df=df, battery=battery, cycle=cycle, window=soc_window)


# ============================================================================ 2. model
class BatteryPINN(nn.Module):
    """
    Thevenin ECM:  V_term = V_oc - I*R0 - V_rc1
    RC branch ODE: dV_rc1/dt = I/C1 - V_rc1/(R1*C1)
    The network predicts V_rc1(t); R0, R1, C1 are learnable scalars stored as logs
    so each optimizer step is a percentage change and they can never go negative.
    """
    def __init__(self, R0=0.05, R1=0.03, C1=1500.0, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))
        self.log_R0 = nn.Parameter(torch.log(torch.tensor([R0])))
        self.log_R1 = nn.Parameter(torch.log(torch.tensor([R1])))
        self.log_C1 = nn.Parameter(torch.log(torch.tensor([C1])))

    @property
    def R0(self): return torch.exp(self.log_R0)
    @property
    def R1(self): return torch.exp(self.log_R1)
    @property
    def C1(self): return torch.exp(self.log_C1)

    def forward(self, inputs, voc):
        current = inputs[:, 1:2]
        v_rc1 = self.net(inputs)
        v_term = voc - current * self.R0 - v_rc1
        return v_term, v_rc1

    def compute_loss(self, inputs, voc, v_meas, t_scale, lambda_physics=0.1):
        inputs = inputs.clone().requires_grad_(True)
        current = inputs[:, 1:2]
        v_term, v_rc1 = self.forward(inputs, voc)

        data_loss = torch.mean((v_term - v_meas) ** 2)

        dv_dtn = torch.autograd.grad(v_rc1, inputs, torch.ones_like(v_rc1), create_graph=True)[0][:, 0:1]
        dv_dt = dv_dtn / t_scale                                     # back to volts per second
        residual = dv_dt + v_rc1 / (self.R1 * self.C1) - current / self.C1
        physics_loss = torch.mean(residual ** 2) * t_scale ** 2     # scale-free (volts^2)

        return data_loss + lambda_physics * physics_loss, data_loss, physics_loss


# ============================================================================ 3. fit one cycle
def fit_cycle(data, epochs=1500, lr=5e-3, lambda_physics=0.1, verbose=False):
    model = BatteryPINN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad()
        loss, dl, pl = model.compute_loss(data["inputs"], data["voc"], data["v"], data["t_scale"], lambda_physics)
        loss.backward(); opt.step(); sched.step()
        if verbose and ep % 300 == 0:
            print(f"  ep {ep:5d}  loss {loss.item():.2e}  data {dl.item():.2e}  phys {pl.item():.2e}")
    R0, R1, C1 = model.R0.item(), model.R1.item(), model.C1.item()
    return dict(R0=R0, R1=R1, C1=C1, tau=R1 * C1, data_loss=dl.item(), physics_loss=pl.item()), model


# ============================================================================ 4. tracking loop
def run_tracking(meta, batteries, cycles, n_windows=4, epochs=1500, lambda_physics=0.1):
    rows = []
    edges = np.linspace(1.0, 0.0, n_windows + 1)
    windows = [(1.0, 0.0)] + [(edges[i + 1], edges[i]) for i in range(n_windows)]   # full cycle + windows
    for b in batteries:
        ocv = build_ocv_curve(meta, b, cycle=0)                                   # fresh table, reused
        cap0 = meta[(meta.battery_id == b) & (meta.type == "discharge")].Capacity.iloc[0]
        for c in cycles:
            cap = meta[(meta.battery_id == b) & (meta.type == "discharge") & (meta.cycle == c)].Capacity.iloc[0]
            for lo, hi in windows:
                lo, hi = sorted((lo, hi))
                t0 = time.time()
                data = prepare_cycle(meta, b, c, ocv, (lo, hi))
                if len(data["df"]) < 10:
                    continue
                p, _ = fit_cycle(data, epochs=epochs, lambda_physics=lambda_physics)
                rows.append(dict(battery=b, cycle=c, soc_lo=lo, soc_hi=hi, soc_mid=(lo + hi) / 2,
                                 window="full" if (lo, hi) == (0.0, 1.0) else f"{lo:.2f}-{hi:.2f}",
                                 capacity=cap, soh=cap / cap0, n_points=len(data["df"]), **p))
                print(f"{b} cycle {c:3d} SOC {lo:.2f}-{hi:.2f}  R0={p['R0']*1e3:6.1f} mΩ  R1={p['R1']*1e3:6.1f} mΩ "
                      f"C1={p['C1']:7.0f} F  tau={p['tau']:6.1f} s  ({time.time()-t0:.1f}s)")
    out = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True); out.to_csv(OUT / "params.csv", index=False)
    print("wrote", OUT / "params.csv")
    return out


# ============================================================================ 5. overfitting contrast
def overfit_contrast(meta, battery="B0005", cycle=0, train_frac=0.6, epochs=1500, lambdas=(0.0, 0.1)):
    """
    Train on the first `train_frac` of the cycle, evaluate on the rest.
    Without the physics term the net can memorise the training part and invent any V_rc1,
    so the held-out prediction and the fitted R1/C1 drift; with it they stay physical.
    """
    ocv = build_ocv_curve(meta, battery, cycle=0)
    data = prepare_cycle(meta, battery, cycle, ocv)
    n = len(data["df"]); k = int(n * train_frac)
    train = {**data, "inputs": data["inputs"][:k], "voc": data["voc"][:k], "v": data["v"][:k]}
    preds, summary = [], []
    for lam in lambdas:
        p, model = fit_cycle(train, epochs=epochs, lambda_physics=lam)
        with torch.no_grad():
            v_hat, v_rc1 = model(data["inputs"], data["voc"])
        err = (v_hat - data["v"]).squeeze().numpy()
        summary.append(dict(lambda_physics=lam, train_rmse=float(np.sqrt(np.mean(err[:k] ** 2))),
                            test_rmse=float(np.sqrt(np.mean(err[k:] ** 2))), **p))
        preds.append(pd.DataFrame(dict(lambda_physics=lam, time=data["df"]["Time"], soc=data["df"]["SOC"],
                                       v_meas=data["v"].squeeze().numpy(), v_pred=v_hat.squeeze().numpy(),
                                       v_rc1=v_rc1.squeeze().numpy(), split=["train"] * k + ["test"] * (n - k))))
    OUT.mkdir(exist_ok=True)
    pd.concat(preds).to_csv(OUT / "contrast.csv", index=False)
    s = pd.DataFrame(summary); s.to_csv(OUT / "contrast_summary.csv", index=False)
    print(s.to_string(index=False))
    return s


# ============================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batteries", nargs="+", default=["B0005", "B0006"])
    ap.add_argument("--cycle-step", type=int, default=20, help="fit every k-th discharge cycle")
    ap.add_argument("--windows", type=int, default=4, help="number of SOC windows per cycle (plus full cycle)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lambda-physics", type=float, default=0.1)
    ap.add_argument("--skip-contrast", action="store_true")
    a = ap.parse_args()

    meta = load_meta()
    n_cyc = min(((meta.battery_id == b) & (meta.type == "discharge")).sum() for b in a.batteries)
    cycles = list(range(0, n_cyc, a.cycle_step))
    if cycles[-1] != n_cyc - 1:
        cycles.append(n_cyc - 1)                                   # always include the last (most aged) cycle
    print(f"cycles: {cycles}")
    run_tracking(meta, a.batteries, cycles, a.windows, a.epochs, a.lambda_physics)
    if not a.skip_contrast:
        overfit_contrast(meta, a.batteries[0], cycle=0, epochs=a.epochs, lambdas=(0.0, a.lambda_physics))
