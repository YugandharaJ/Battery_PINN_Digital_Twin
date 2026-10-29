# Battery PINN – parameter tracking on the NASA aging dataset

Goal: track how the equivalent-circuit parameters R0, R1, C1 of two NASA cells
(B0005, B0006) drift from fresh to aged, using a physics-informed neural network,
and show why the physics term matters.

## Files

| file | role |
|---|---|
| `soc_ocv.py` | load + clean one test, coulomb-count SOC, build the OCV-vs-SOC lookup table |
| `battery_pinn.py` | the PINN, one-cycle fit, tracking loop, overfitting contrast. Writes `results/` |
| `dashboard.py` | Streamlit app that reads `results/` and draws the 3D drift surface |
| `results/params.csv` | one row per (cell, cycle, SOC window): R0, R1, C1, tau, capacity, SOH |
| `results/contrast.csv` | per-time-step predictions with and without the physics loss |
| `results/contrast_summary.csv` | train / held-out RMSE and fitted parameters for both runs |

## Run

```bash
cd Practice
python battery_pinn.py --batteries B0005 B0006 --cycle-step 20 --windows 4 --epochs 1500
streamlit run dashboard.py
```

## Data flow

```
 cleaned_dataset/metadata.csv            cleaned_dataset/data/*.csv
          │                                        │
          ▼                                        ▼
 ┌─────────────────────────────── soc_ocv.py ───────────────────────────────┐
 │ load_meta()        clean Capacity/Re/Rct, add per-cell cycle counter      │
 │ load_test()        one charge or discharge file, numeric, sorted by time │
 │ add_soc()          Ah = ∫|I| dt   →  SOC = 1 − Ah / capacity              │
 │ build_ocv_curve()  discharge k + charge k+1, trim rests, average → table  │
 │ ocv_lookup()       np.interp(SOC, table)                                  │
 └───────────────────────────────────────────────────────────────────────────┘
          │  SOC, OCV per row
          ▼
 ┌────────────────────────────── battery_pinn.py ───────────────────────────┐
 │                                                                           │
 │ prepare_cycle()   one discharge cycle (optionally one SOC window)         │
 │                   inputs = [t/t_scale, |I|, (T−25)/10]   voc   v_measured │
 │        │                                                                  │
 │        ▼                                                                  │
 │ BatteryPINN       net(inputs) ──► V_rc1(t)                                │
 │   log R0,R1,C1    V_term = V_oc − I·R0 − V_rc1               ─┐ data loss │
 │                   dV_rc1/dt + V_rc1/(R1·C1) − I/C1 = 0        ─┘ physics  │
 │        │                                                                  │
 │        ▼                                                                  │
 │ fit_cycle()       Adam on net weights + log R0, log R1, log C1            │
 │                   returns {R0, R1, C1, tau}                               │
 │        │                                                                  │
 │        ├──► run_tracking()     for cell × cycle × SOC window → params.csv │
 │        │                                                                  │
 │        └──► overfit_contrast() train on first 60 % of one cycle,          │
 │                                λ_physics = 0 vs 0.1 → contrast.csv        │
 └───────────────────────────────────────────────────────────────────────────┘
          │  results/*.csv
          ▼
 ┌──────────────────────────────── dashboard.py ────────────────────────────┐
 │ tab "3D drift surface"  go.Surface  x = cycle, y = SOC window, z = param  │
 │ tab "Drift vs cycle"    whole-cycle R0/R1/C1/τ vs cycle, capacity fade    │
 │ tab "Why physics?"      measured vs predicted voltage, V_rc1, RMSE table  │
 │ tab "Table"             raw params.csv + download                         │
 └───────────────────────────────────────────────────────────────────────────┘
```

## What the model learns

* **R0** – instant voltage drop per amp. Pinned by the data loss.
* **R1, C1** – appear only in the ODE, so only the physics loss gives them a
  gradient. With `lambda_physics = 0` they never move from their initial values.
* **V_rc1(t)** – the network's output; with the physics term it must obey the RC
  equation, without it it is a free function that absorbs any mismatch.

## Known limits

* OCV table is built from one fresh cycle and reused for aged cycles.
* R0 is a single constant per fit; the SOC windows give coarse SOC dependence.
* `lambda_physics` = 0.1 is untuned. Try 0.01–1 and watch train vs held-out RMSE.
