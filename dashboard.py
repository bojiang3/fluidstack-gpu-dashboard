"""
dashboard.py - Fluidstack GPU Dashboard (v3 - Real Cluster)
============================================================
Shows REAL Slurm cluster data: job queue, node status, GPU metrics.
"""
from __future__ import annotations
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = "http://localhost:8000"
REFRESH = 10

st.set_page_config(page_title="Fluidstack GPU Dashboard", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;500;700;900&display=swap');
:root{--bg:#0a0e17;--card:#111827;--border:#1e293b;--cyan:#06d6a0;--orange:#ff6b35;--red:#ef4444;--blue:#3b82f6;--purple:#8b5cf6;--text:#e2e8f0;--muted:#94a3b8;}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Outfit',sans-serif!important;}
h1,h2,h3,h4,h5,h6,p,span,label,div{color:var(--text)!important;}
.stTabs [data-baseweb="tab-list"]{gap:0;background:var(--card);border-radius:12px;padding:4px;}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:10px 20px;font-weight:600;color:var(--muted)!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--cyan),var(--blue))!important;color:#000!important;}
.mc{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px;text-align:center;}
.mv{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;background:linear-gradient(135deg,var(--cyan),var(--blue));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.ml{font-size:0.8rem;color:var(--muted)!important;text-transform:uppercase;letter-spacing:1px;margin-top:2px;}
.sl{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.7rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:2px;}
.sl-live{background:#065f46;color:#06d6a0;border:1px solid #06d6a0;}
.sl-mock{background:#78350f;color:#fbbf24;border:1px solid #fbbf24;}
.ac{background:linear-gradient(135deg,#7f1d1d,#991b1b);border:1px solid var(--red);border-radius:12px;padding:14px 18px;margin:6px 0;animation:pulse 2s infinite;}
.aw{background:linear-gradient(135deg,#78350f,#92400e);border:1px solid var(--orange);border-radius:12px;padding:14px 18px;margin:6px 0;}
.aa{background:linear-gradient(135deg,#312e81,#3730a3);border:1px solid var(--purple);border-radius:12px;padding:14px 18px;margin:6px 0;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.85}}
.dh{background:linear-gradient(135deg,rgba(6,214,160,0.08),rgba(59,130,246,0.08));border:1px solid var(--border);border-radius:20px;padding:28px 36px;margin-bottom:20px;}
.dt{font-weight:900;font-size:1.8rem;background:linear-gradient(135deg,var(--cyan),var(--blue),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.ds{color:var(--muted)!important;font-size:0.95rem;margin-top:4px;}
.nd{display:inline-block;width:42px;height:42px;border-radius:8px;margin:3px;text-align:center;line-height:42px;font-size:0.6rem;font-family:'JetBrains Mono',monospace;font-weight:700;color:#fff;cursor:default;}
.nd-mixed{background:linear-gradient(135deg,#06d6a0,#059669);}
.nd-idle{background:#1e293b;color:var(--muted);}
.nd-alloc{background:linear-gradient(135deg,#f97316,#ea580c);}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def api_get(path):
    try:
        r = requests.get(API + path, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def pdark(**kw):
    b = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)",
             font=dict(family="Outfit", color="#e2e8f0", size=13),
             margin=dict(l=50,r=30,t=50,b=40),
             xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
             legend=dict(bgcolor="rgba(0,0,0,0)"))
    b.update(kw)
    return b

def sbadge(src):
    if src in ("srun-nvidia-smi","ssh-nvidia-smi","nvidia-smi","psutil","slurm"):
        return '<span class="sl sl-live">%s LIVE</span>' % src
    return '<span class="sl sl-mock">%s MOCK</span>' % src


# ── Header ──
st.markdown('<div class="dh"><div class="dt">⚡ Fluidstack GPU Dashboard</div><div class="ds">Real Slurm Cluster · 32 Nodes × 8 GPUs · Power to Prefill</div></div>', unsafe_allow_html=True)

if "lr" not in st.session_state:
    st.session_state.lr = time.time()
el = int(time.time() - st.session_state.lr)
st.caption("🔄 %ds · %s" % (max(0, REFRESH - el), datetime.now().strftime("%H:%M:%S")))

# ── Fetch ──
data = api_get("/metrics")
hist = api_get("/metrics/history")
slurm_jobs = api_get("/slurm/jobs")
slurm_nodes = api_get("/slurm/nodes")
alerts_data = api_get("/alerts")
bench_data = api_get("/benchmark")

# ── Sources ──
if data:
    srcs = data.get("data_sources", [])
    st.markdown("**Data:** %s" % " ".join(sbadge(s) for s in srcs), unsafe_allow_html=True)

# ── Tabs ──
t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Overview", "🖥️ Cluster Map", "📋 Slurm Jobs", "🔔 Alerts", "⚔️ vs Ray", "🖥️ System"])

# ══════════════════════════════════════════════
# TAB: OVERVIEW
# ══════════════════════════════════════════════
with t1:
    if data:
        gpus = data["gpus"]
        cl = data.get("cluster", {})
        au = sum(g["utilization_pct"] for g in gpus) / max(len(gpus), 1)
        ap = sum(g["power_draw_w"] for g in gpus) / max(len(gpus), 1)
        at = sum(g["temperature_c"] for g in gpus) / max(len(gpus), 1)
        tp = sum(g["power_draw_w"] for g in gpus)

        cols = st.columns(6)
        cards = [
            (str(cl.get("total_nodes", len(gpus))), "Nodes"),
            (str(cl.get("total_gpus", len(gpus))), "Total GPUs"),
            (str(cl.get("active_jobs", 0)), "Slurm Jobs"),
            ("%.1f%%" % au, "Avg Util"),
            ("%.0fW" % ap, "Avg Power"),
            ("%.1f°C" % at, "Avg Temp"),
        ]
        for c, (v, l) in zip(cols, cards):
            c.markdown('<div class="mc"><div class="mv">%s</div><div class="ml">%s</div></div>' % (v, l), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauges
        st.subheader("GPU Power Draw")
        gcols = st.columns(min(len(gpus), 4))
        for i, g in enumerate(gpus[:4]):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=g["power_draw_w"],
                number={"suffix":"W","font":{"family":"JetBrains Mono","size":24}},
                title={"text":"%s<br><sub>%s</sub>" % (g["gpu_id"], g.get("node","")), "font":{"size":12}},
                gauge=dict(axis=dict(range=[0,1000]), bar=dict(color="#06d6a0"), bgcolor="#1e293b",
                    steps=[{"range":[0,300],"color":"rgba(59,130,246,0.1)"},
                           {"range":[300,600],"color":"rgba(6,214,160,0.1)"},
                           {"range":[600,800],"color":"rgba(255,107,53,0.1)"},
                           {"range":[800,1000],"color":"rgba(239,68,68,0.1)"}],
                    threshold=dict(line=dict(color="#ef4444",width=3),thickness=0.8,value=700)),
            ))
            fig.update_layout(**pdark(height=220, margin=dict(l=25,r=25,t=55,b=10)))
            gcols[i % len(gcols)].plotly_chart(fig, use_container_width=True, key="g%d" % i)

        # Timeline
        if hist and hist.get("history"):
            st.subheader("Utilization Timeline")
            rows = []
            for s in hist["history"]:
                for g in s["gpus"]:
                    rows.append({"time":s["timestamp"],"GPU":g["gpu_id"],"Util":g["utilization_pct"]})
            if rows:
                df = pd.DataFrame(rows)
                df["time"] = pd.to_datetime(df["time"])
                fig = px.line(df, x="time", y="Util", color="GPU",
                              color_discrete_sequence=["#06d6a0","#3b82f6","#8b5cf6","#f97316"], markers=True)
                fig.update_layout(**pdark(height=320, yaxis_range=[0,105]))
                st.plotly_chart(fig, use_container_width=True, key="tl")

        # Heatmap
        if gpus:
            st.subheader("GPU Heatmap")
            z = np.array([
                [g["temperature_c"] for g in gpus],
                [g["power_draw_w"] for g in gpus],
                [g["utilization_pct"] for g in gpus],
            ])
            fig = go.Figure(go.Heatmap(
                z=z, x=[g["gpu_id"] for g in gpus], y=["Temp","Power","Util"],
                colorscale=[[0,"#1e3a5f"],[0.3,"#06d6a0"],[0.6,"#fbbf24"],[0.8,"#f97316"],[1,"#ef4444"]],
                text=[["%.1f" % v for v in r] for r in z], texttemplate="%{text}",
                textfont={"size":12,"family":"JetBrains Mono"},
            ))
            fig.update_layout(**pdark(height=240))
            st.plotly_chart(fig, use_container_width=True, key="hm")
    else:
        st.error("Cannot reach backend. Run: python3 app.py")


# ══════════════════════════════════════════════
# TAB: CLUSTER MAP (real nodes from sinfo)
# ══════════════════════════════════════════════
with t2:
    st.subheader("Cluster Node Map (from sinfo)")
    if slurm_nodes and slurm_nodes.get("nodes"):
        nodes = slurm_nodes["nodes"]
        st.markdown("**%d nodes** · 8 GPUs each · %s" % (len(nodes), sbadge("slurm")), unsafe_allow_html=True)

        mixed = [n for n in nodes if n["state"] == "mixed"]
        idle = [n for n in nodes if n["state"] == "idle"]

        st.markdown("### 🟢 Active (%d nodes)" % len(mixed))
        html = ""
        for n in mixed:
            pct = n["cpus_alloc"] / max(n["cpus_total"], 1) * 100
            html += '<div class="nd nd-mixed" title="%s: %d/%d CPUs">%s</div>' % (
                n["name"], n["cpus_alloc"], n["cpus_total"], n["name"].split("-")[-1])
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("### ⚪ Idle (%d nodes)" % len(idle))
        html = ""
        for n in idle:
            html += '<div class="nd nd-idle" title="%s: idle">%s</div>' % (
                n["name"], n["name"].split("-")[-1])
        st.markdown(html, unsafe_allow_html=True)

        # Node details table
        st.markdown("### Node Details")
        ndf = pd.DataFrame(nodes)
        ndf["cpu_usage"] = ndf.apply(lambda r: "%d/%d" % (r["cpus_alloc"], r["cpus_total"]), axis=1)
        ndf["memory_gb"] = (ndf["memory_mb"] / 1024).round(0).astype(int)
        st.dataframe(
            ndf[["name","state","gres","cpu_usage","memory_gb"]].rename(
                columns={"name":"Node","state":"State","gres":"GPUs","cpu_usage":"CPUs Used","memory_gb":"RAM (GB)"}),
            use_container_width=True, hide_index=True)
    else:
        st.info("No cluster data. Is backend connected to Slurm?")


# ══════════════════════════════════════════════
# TAB: SLURM JOBS (real squeue data)
# ══════════════════════════════════════════════
with t3:
    st.subheader("Slurm Job Queue (Live)")
    if slurm_jobs and slurm_jobs.get("jobs"):
        jobs = slurm_jobs["jobs"]
        running = [j for j in jobs if j["state"] == "RUNNING"]
        pending = [j for j in jobs if j["state"] == "PENDING"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Jobs", len(jobs))
        c2.metric("Running", len(running))
        c3.metric("Pending", len(pending))

        st.markdown("Source: %s" % sbadge("slurm"), unsafe_allow_html=True)

        jdf = pd.DataFrame(jobs)
        st.dataframe(
            jdf[["job_id","name","user","state","time","nodelist"]].rename(
                columns={"job_id":"Job ID","name":"Name","user":"User","state":"State","time":"Runtime","nodelist":"Node"}),
            use_container_width=True, hide_index=True)

        # Jobs per node chart
        if running:
            node_counts = {}
            for j in running:
                nl = j.get("nodelist", "?")
                node_counts[nl] = node_counts.get(nl, 0) + 1
            nc_df = pd.DataFrame([{"node": k, "jobs": v} for k, v in sorted(node_counts.items(), key=lambda x: -x[1])])
            fig = px.bar(nc_df, x="node", y="jobs", color="jobs",
                         color_continuous_scale=["#06d6a0","#f97316","#ef4444"])
            fig.update_layout(**pdark(height=300, title="Jobs per Node"))
            st.plotly_chart(fig, use_container_width=True, key="jpn")
    else:
        st.info("No Slurm job data.")


# ══════════════════════════════════════════════
# TAB: ALERTS
# ══════════════════════════════════════════════
with t4:
    st.subheader("Alerts & Anomaly Detection")
    if alerts_data:
        al = alerts_data.get("alerts", [])
        if not al:
            st.success("All systems nominal.")
        else:
            anoms = [a for a in al if a.get("type") == "anomaly"]
            thresh = [a for a in al if a.get("type") == "threshold"]
            if anoms:
                st.markdown("### 🧠 Anomalies (z-score)")
                for a in anoms:
                    st.markdown('<div class="aa"><b>🟣 %s</b> %s</div>' % (a["gpu_id"], a["message"]), unsafe_allow_html=True)
            if thresh:
                st.markdown("### ⚠️ Thresholds")
                for a in thresh:
                    cls = "ac" if a["severity"] == "critical" else "aw"
                    st.markdown('<div class="%s"><b>%s %s</b> %s</div>' % (
                        cls, "🔴" if a["severity"]=="critical" else "🟡", a["gpu_id"], a["message"]), unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB: BENCHMARK
# ══════════════════════════════════════════════
with t5:
    st.subheader("Fluidstack Dashboard vs Ray Dashboard")
    st.markdown("> Purpose-built for **Fluidstack Slurm clusters** with real squeue/sinfo, SSH-based nvidia-smi, anomaly detection, and Prometheus export.")
    if bench_data and bench_data.get("comparison"):
        rows = bench_data["comparison"]
        st.dataframe(pd.DataFrame(rows).rename(columns={"metric":"Metric","fluidstack_dashboard":"⚡ Ours","ray_dashboard":"🔵 Ray"}),
                     use_container_width=True, hide_index=True)
        numeric = [r for r in rows if isinstance(r.get("fluidstack_dashboard"), (int,float))]
        if numeric:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="⚡ Ours", x=[r["metric"] for r in numeric],
                                 y=[r["fluidstack_dashboard"] for r in numeric], marker_color="#06d6a0"))
            fig.add_trace(go.Bar(name="🔵 Ray", x=[r["metric"] for r in numeric],
                                 y=[r["ray_dashboard"] for r in numeric], marker_color="#3b82f6"))
            fig.update_layout(**pdark(height=350, barmode="group", title="Lower = Better"))
            st.plotly_chart(fig, use_container_width=True, key="bm")


# ══════════════════════════════════════════════
# TAB: SYSTEM
# ══════════════════════════════════════════════
with t6:
    if data and data.get("system"):
        s = data["system"]
        st.subheader("Host System Metrics")
        st.markdown("Source: %s" % sbadge(s.get("source","?")), unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("CPU", "%.1f%%" % s["cpu_percent"])
        c2.metric("RAM", "%.1f%% (%.1f/%.1f GB)" % (s["ram_percent"], s.get("ram_used_gb",0), s.get("ram_total_gb",0)))
        c3.metric("Disk", "%.1f%%" % s.get("disk_percent",0))

# ── Footer + refresh ──
st.markdown("---")
st.markdown("<div style='text-align:center;color:#64748b;font-size:0.8rem;'>⚡ Fluidstack GPU Dashboard v3 · Real Cluster · Power to Prefill</div>", unsafe_allow_html=True)
if el >= REFRESH:
    st.session_state.lr = time.time()
    st.rerun()
