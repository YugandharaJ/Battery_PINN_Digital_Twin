"""
dashboard.py – Streamlit dashboard for the PINN parameter-tracking results.

Reads results/params.csv and results/contrast.csv written by battery_pinn.py.
It never runs the PINN itself, so it stays responsive.

Run from the Practice/ folder:
    streamlit run dashboard.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RES = Path(__file__).resolve().parent / "results"
CELL_COLORS = {"B0005": "#2a78d6", "B0006": "#eb6834", "B0007": "#1baf7a"}
PARAM_LABEL = {"R0": "R0  ohmic resistance (mΩ)", "R1": "R1  polarisation resistance (mΩ)",
               "C1": "C1  polarisation capacitance (F)", "tau": "τ = R1·C1  time constant (s)"}
SCALE = {"R0": 1e3, "R1": 1e3, "C1": 1.0, "tau": 1.0}
LAYOUT = dict(template="plotly_white", font=dict(size=13), margin=dict(l=10, r=10, t=90, b=10))


@st.cache_data
def load_results():
    params = pd.read_csv(RES / "params.csv")
    contrast = pd.read_csv(RES / "contrast.csv") if (RES / "contrast.csv").exists() else None
    summary = pd.read_csv(RES / "contrast_summary.csv") if (RES / "contrast_summary.csv").exists() else None
    return params, contrast, summary


# ----------------------------------------------------------------------------- figures
def surface_3d(params, cells, param):
    """One surface per cell: x = cycle, y = SOC window mid-point, z = parameter."""
    fig = go.Figure()
    for b in cells:
        d = params[(params.battery == b) & (params.window != "full")]
        if d.empty:
            continue
        grid = d.pivot_table(index="soc_mid", columns="cycle", values=param) * SCALE[param]
        fig.add_trace(go.Surface(x=grid.columns, y=grid.index, z=grid.values, name=b, showscale=False,
                                 colorscale=[[0, CELL_COLORS.get(b, "#888")], [1, CELL_COLORS.get(b, "#888")]],
                                 opacity=0.75, hovertemplate=f"{b}<br>cycle %{{x}}<br>SOC %{{y:.2f}}<br>%{{z:.1f}}<extra></extra>"))
        fig.add_trace(go.Scatter3d(x=d.cycle, y=d.soc_mid, z=d[param] * SCALE[param], mode="markers",
                                   marker=dict(size=3, color=CELL_COLORS.get(b, "#888")), name=f"{b} fits",
                                   hoverinfo="skip"))
    fig.update_layout(height=620, title=f"{PARAM_LABEL[param]} across cycle number and state of charge",
                      scene=dict(xaxis_title="Discharge cycle #", yaxis_title="SOC (window mid-point)",
                                 zaxis_title=PARAM_LABEL[param].split("  ")[0],
                                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))), **LAYOUT)
    return fig


def drift_2d(params, cells, param):
    """Full-cycle fit of one parameter against cycle number, one line per cell."""
    fig = go.Figure()
    for b in cells:
        d = params[(params.battery == b) & (params.window == "full")].sort_values("cycle")
        fig.add_trace(go.Scatter(x=d.cycle, y=d[param] * SCALE[param], mode="lines+markers", name=b,
                                 line=dict(width=2, color=CELL_COLORS.get(b, "#888")), marker=dict(size=8),
                                 customdata=np.stack([d.capacity, d.soh * 100], axis=1),
                                 hovertemplate="cycle %{x}<br>%{y:.1f}<br>capacity %{customdata[0]:.3f} Ah "
                                               "(SOH %{customdata[1]:.0f}%)<extra></extra>"))
    fig.update_layout(height=380, title=f"{PARAM_LABEL[param]}, whole-cycle fit", xaxis_title="Discharge cycle #",
                      yaxis_title=PARAM_LABEL[param].split("  ")[0], hovermode="x unified",
                      legend=dict(orientation="h", y=1.12), **LAYOUT)
    return fig


def capacity_fig(params, cells):
    fig = go.Figure()
    for b in cells:
        d = params[(params.battery == b) & (params.window == "full")].sort_values("cycle")
        fig.add_trace(go.Scatter(x=d.cycle, y=d.capacity, mode="lines+markers", name=b,
                                 line=dict(width=2, color=CELL_COLORS.get(b, "#888")), marker=dict(size=8)))
    fig.add_hline(y=1.4, line=dict(color="#52514e", dash="dot", width=1), annotation_text="end of life 1.4 Ah",
                  annotation_position="bottom left")
    fig.update_layout(height=380, title="Measured capacity (from metadata.csv)", xaxis_title="Discharge cycle #",
                      yaxis_title="Ah", hovermode="x unified", legend=dict(orientation="h", y=1.12), **LAYOUT)
    return fig


def contrast_fig(contrast, column, title):
    fig = go.Figure()
    base = contrast[contrast.lambda_physics == contrast.lambda_physics.min()]
    if column == "v_pred":
        fig.add_trace(go.Scatter(x=base.time, y=base.v_meas, mode="lines", name="measured",
                                 line=dict(width=2, color="#0b0b0b")))
    for lam, color in zip(sorted(contrast.lambda_physics.unique()), ["#eb6834", "#2a78d6"]):
        d = contrast[contrast.lambda_physics == lam]
        fig.add_trace(go.Scatter(x=d.time, y=d[column], mode="lines",
                                 name=f"λ_physics = {lam:g}" + ("  (data only)" if lam == 0 else ""),
                                 line=dict(width=2, color=color)))
    t_split = base[base.split == "test"].time.min()
    fig.add_vrect(x0=t_split, x1=base.time.max(), fillcolor="#52514e", opacity=0.07, line_width=0,
                  annotation_text="held out", annotation_position="top left")
    fig.update_layout(height=380, title=title, xaxis_title="Time (s)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.12), **LAYOUT)
    return fig


# ----------------------------------------------------------------------------- page
st.set_page_config(page_title="Battery PINN drift", layout="wide")
st.title("Resistance drift of aging Li-ion cells, estimated by a physics-informed neural network")

if not (RES / "params.csv").exists():
    st.error("No results yet. Run  `python battery_pinn.py`  first to create results/params.csv.")
    st.stop()
params, contrast, summary = load_results()

with st.sidebar:
    st.header("View")
    cells = st.multiselect("Cells", sorted(params.battery.unique()), default=sorted(params.battery.unique()))
    param = st.radio("Parameter", list(PARAM_LABEL), format_func=lambda p: PARAM_LABEL[p])
    st.caption(f"{len(params)} fits · cycles {params.cycle.min()}–{params.cycle.max()} · "
               f"{params.window.nunique() - 1} SOC windows per cycle")

tab3d, tab2d, tabc, tabt = st.tabs(["3D drift surface", "Drift vs cycle", "Why physics?", "Table"])

with tab3d:
    st.plotly_chart(surface_3d(params, cells, param), use_container_width=True)
    st.caption("Each surface is one cell. Height is the parameter fitted on one SOC window of one cycle. "
               "Drag to rotate; hover for values.")

with tab2d:
    c1, c2 = st.columns(2)
    c1.plotly_chart(drift_2d(params, cells, param), use_container_width=True)
    c2.plotly_chart(capacity_fig(params, cells), use_container_width=True)
    full = params[params.window == "full"]
    first, last = full[full.cycle == full.cycle.min()], full[full.cycle == full.cycle.max()]
    cols = st.columns(len(cells) or 1)
    for col, b in zip(cols, cells):
        f, l = first[first.battery == b], last[last.battery == b]
        if len(f) and len(l):
            v0, v1 = f[param].iloc[0] * SCALE[param], l[param].iloc[0] * SCALE[param]
            col.metric(f"{b}: {param} fresh → aged", f"{v1:.1f}", f"{(v1 / v0 - 1) * 100:+.0f}% vs cycle {f.cycle.iloc[0]}")

with tabc:
    if contrast is None:
        st.info("Run battery_pinn.py without --skip-contrast to generate this comparison.")
    else:
        st.markdown("Same network, same cycle, trained on the first 60 % of the discharge only. "
                    "Without the physics term the model fits the training part and invents an unphysical "
                    "V_RC1; with it the held-out voltage and the circuit parameters stay sensible.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(contrast_fig(contrast, "v_pred", "Terminal voltage: measured vs predicted"), use_container_width=True)
        c2.plotly_chart(contrast_fig(contrast, "v_rc1", "Hidden RC voltage V_RC1 predicted by the network"), use_container_width=True)
        if summary is not None:
            s = summary.copy()
            s[["R0", "R1"]] = s[["R0", "R1"]] * 1e3
            s[["train_rmse", "test_rmse"]] = s[["train_rmse", "test_rmse"]] * 1e3
            st.dataframe(s.rename(columns={"R0": "R0 (mΩ)", "R1": "R1 (mΩ)", "C1": "C1 (F)", "tau": "τ (s)",
                                           "train_rmse": "train RMSE (mV)", "test_rmse": "held-out RMSE (mV)"})
                         .drop(columns=["data_loss", "physics_loss"], errors="ignore").round(2),
                         use_container_width=True, hide_index=True)

with tabt:
    show = params[params.battery.isin(cells)].copy()
    show[["R0", "R1"]] = show[["R0", "R1"]] * 1e3
    st.dataframe(show.rename(columns={"R0": "R0 (mΩ)", "R1": "R1 (mΩ)", "C1": "C1 (F)", "tau": "τ (s)"}).round(3),
                 use_container_width=True, hide_index=True)
    st.download_button("Download params.csv", (RES / "params.csv").read_bytes(), "params.csv", "text/csv")
