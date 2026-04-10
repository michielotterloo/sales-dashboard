"""
Sales Dashboard — Happy Users First
Focus: active users → adoption → revenue follows.
Everything is measured in quarters: set goals, achieve them, compare.
Run: streamlit run dashboard.py --server.port 8503
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Sales Dashboard", layout="wide", page_icon="📊")

# ─── Colors ─────────────────────────────────────────────────────
GREEN = "#27AE60"
BLUE = "#2E86C1"
DARK = "#1B4F72"
LIGHT = "#85C1E9"
RED = "#E74C3C"
ORANGE = "#E67E22"
GREY = "#95A5A6"


def _dedup_by_debtorid(df):
    """Aggregate multiple HubSpot entries with the same debtorID into one customer row.

    One customer (debtorID) can have multiple HubSpot company entries for different
    branches or financial streams. Revenue is summed; EdC/NPS data (already per
    debtorID) takes the first non-null; dates take the most recent.
    """
    if "debtorid" not in df.columns or df["debtorid"].isna().all():
        return df

    # Separate rows without debtorID — keep as-is
    no_id = df[df["debtorid"].isna() | (df["debtorid"] == "")].copy()
    has_id = df[df["debtorid"].notna() & (df["debtorid"] != "")].copy()

    if has_id.empty:
        return df

    # For each debtorID group, pick the "primary" row (highest revenue) for name/owner
    has_id["_rev"] = pd.to_numeric(has_id["revenue_12m"], errors="coerce").fillna(0)
    has_id = has_id.sort_values("_rev", ascending=False)

    # Aggregation rules
    agg = {
        "id": "first",                    # keep primary HubSpot id
        "name": "first",                  # highest-revenue entry name
        "domain": "first",
        "owner_id": "first",
        "owner_name": "first",
        "lifecycle": "first",
        "industry": "first",
        "billing_interval_days": "first",
        "billing_freq": "first",
        "product": "first",
        # Financials — already per debtorID from MoneyBird, take first
        "revenue_12m": "first",
        "invoice_count": "first",
        # EdC data — already per debtorID, take first non-null
        "active_users": "first",
        "paying_users": "first",
        "trend_pct": "first",
        "q1_2026_avg": "first",
        "q4_2025_avg": "first",
        "q1_2025_avg": "first",
        "yoy_pct": "first",
        "qoq_pct": "first",
        # NPS — take first non-null
        "nps_q1_26": "first",
        "nps_q4_25": "first",
        "nps_q1_25": "first",
        "nps_yoy": "first",
        "nps_qoq": "first",
        "nps_responses_q1_26": "first",
        # Dates — most recent
        "last_contacted": "max",
        "last_activity": "max",
        "last_invoice": "max",
    }
    # Only aggregate columns that exist
    agg = {k: v for k, v in agg.items() if k in has_id.columns}

    deduped = has_id.groupby("debtorid", as_index=False).agg(agg)

    # Track how many entries were merged (useful for display)
    entry_counts = has_id.groupby("debtorid").size().reset_index(name="_entry_count")
    deduped = deduped.merge(entry_counts, on="debtorid", how="left")

    no_id["_entry_count"] = 1
    result = pd.concat([deduped, no_id], ignore_index=True)
    for tmp_col in ["_rev"]:
        if tmp_col in result.columns:
            result = result.drop(columns=[tmp_col])
    return result


@st.cache_data(ttl=300)
def load_data():
    with open(DATA_DIR / "snapshot.json") as f:
        snap = json.load(f)

    companies = pd.DataFrame(snap["companies"])
    owners = snap["owners"]
    engagements = snap["engagements"]

    # Parse dates
    for col in ["last_contacted", "last_activity", "last_invoice"]:
        companies[col] = pd.to_datetime(companies[col], errors="coerce")

    now = pd.Timestamp.now(tz="UTC")
    for col in ["last_contacted", "last_invoice"]:
        if companies[col].dt.tz is None:
            companies[col] = companies[col].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

    companies["days_no_invoice"] = (now - companies["last_invoice"]).dt.days
    companies["days_no_contact"] = (now - companies["last_contacted"]).dt.days

    # Ensure numeric columns
    num_cols = ["active_users", "paying_users", "trend_pct", "billing_interval_days",
                "q1_2026_avg", "q1_2025_avg", "q4_2025_avg", "yoy_pct", "qoq_pct",
                "nps_q1_26", "nps_q4_25", "nps_q1_25", "nps_yoy", "nps_qoq",
                "nps_responses_q1_26"]
    for col in num_cols:
        if col in companies.columns:
            companies[col] = pd.to_numeric(companies[col], errors="coerce")

    # Deduplicate: one customer = one debtorID
    companies = _dedup_by_debtorid(companies)

    # Recalculate days after dedup (dates were aggregated with max)
    companies["days_no_invoice"] = (now - companies["last_invoice"]).dt.days
    companies["days_no_contact"] = (now - companies["last_contacted"]).dt.days

    # Health status per account
    companies["health"] = "unknown"
    has_au = companies["active_users"].notna()
    companies.loc[has_au & (companies["trend_pct"] > 5), "health"] = "growing"
    companies.loc[has_au & (companies["trend_pct"] >= -5) & (companies["trend_pct"] <= 5), "health"] = "stable"
    companies.loc[has_au & (companies["trend_pct"] < -5), "health"] = "declining"
    companies.loc[has_au & (companies["active_users"] == 0), "health"] = "inactive"

    # Adoption ratio: paying / active
    companies["adoption_pct"] = (
        companies["paying_users"] / companies["active_users"] * 100
    ).where(companies["active_users"] > 0)

    # Engagement data → DataFrame
    eng_rows = []
    for owner_id, months in engagements.items():
        oname = owners.get(owner_id, {}).get("name", "Unknown") if isinstance(owners.get(owner_id), dict) else owners.get(owner_id, "Unknown")
        for month, counts in months.items():
            eng_rows.append({"owner_id": owner_id, "owner_name": oname, "month": month, **counts})
    eng_df = pd.DataFrame(eng_rows)
    if not eng_df.empty:
        eng_df["month"] = pd.to_datetime(eng_df["month"] + "-01")
        eng_df["total"] = eng_df["calls"] + eng_df["meetings"] + eng_df["tasks"]

    return companies, owners, eng_df, snap["collected_at"]


all_companies, owners, eng_df, collected_at = load_data()

# ─── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("Sales Dashboard")
st.sidebar.caption(f"Data: {collected_at[:16]}")
st.sidebar.markdown("---")

# Product filter
st.sidebar.markdown("**Product**")
product_filter = st.sidebar.radio(
    "Product",
    ["Ed Controls", "FlexWhere", "All"],
    label_visibility="collapsed",
)
if product_filter == "All":
    companies = all_companies
else:
    companies = all_companies[all_companies["product"] == product_filter].copy()

st.sidebar.markdown("---")

# Global quarter comparison
st.sidebar.markdown("**Quarter**")
compare_mode = st.sidebar.radio(
    "Comparison",
    ["Q1 2026 vs Q1 2025 (YoY)", "Q1 2026 vs Q4 2025 (QoQ)"],
    label_visibility="collapsed",
)
is_yoy = "YoY" in compare_mode

# Derived column names based on comparison mode
user_prev_col = "q1_2025_avg" if is_yoy else "q4_2025_avg"
user_trend_col = "yoy_pct" if is_yoy else "qoq_pct"
nps_cur_col = "nps_q1_26"
prev_label = "Q1 2025" if is_yoy else "Q4 2025"

# Filters
sales_owners = sorted(companies["owner_name"].dropna().unique())
sales_owners = [o for o in sales_owners if o not in ("Unknown", "Unassigned", "")]
ec_clients = companies[companies["active_users"].notna()].copy()
nps_clients = companies[companies["nps_q1_26"].notna()].copy()

st.sidebar.markdown("---")
st.sidebar.markdown("**Focus:** Happy Users First")
st.sidebar.markdown(
    "**NPS** = Make People Happy\n\n"
    "**Active Users** = Stay Relevant\n\n"
    "**Revenue** = Stay Strong (outcome)"
)
st.sidebar.markdown("---")
if product_filter == "Ed Controls":
    st.sidebar.caption("NPS target: 50 · Users: A+S roles (RASCI)")
elif product_filter == "FlexWhere":
    st.sidebar.caption("NPS target: 32 · All users = paying users")
else:
    st.sidebar.caption("EC target NPS 50 · FW target NPS 32")

tab1, tab2, tab5, tab3, tab4 = st.tabs([
    "User Health",
    "Adoption Alert",
    "NPS",
    "Account Details",
    "Sales Activity",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: USER HEALTH
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("User Health — How are our users doing?")
    st.caption(f"Ed Controls accounts · Comparing Q1 2026 vs {prev_label}")

    if ec_clients.empty:
        st.warning("No Ed Controls active users data found.")
    else:
        has_trend = ec_clients[user_trend_col].notna()
        ec_with_trend = ec_clients[has_trend].copy()

        # ─ Top-level metrics ─
        total_active = int(ec_clients["active_users"].sum())
        total_prev = int(ec_with_trend[user_prev_col].sum()) if not ec_with_trend.empty else 0
        total_cur = int(ec_with_trend["q1_2026_avg"].sum()) if not ec_with_trend.empty else 0
        pct_change = round((total_cur - total_prev) / total_prev * 100, 1) if total_prev > 0 else 0
        growing = len(ec_with_trend[ec_with_trend[user_trend_col] > 5])
        declining = len(ec_with_trend[ec_with_trend[user_trend_col] < -5])
        avg_adoption = ec_clients["adoption_pct"].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Active Users (Q1 '26)", f"{total_active:,}")
        col2.metric(f"vs {prev_label}", f"{pct_change:+.1f}%",
                    delta=f"{total_cur - total_prev:+,} users", delta_color="normal")
        col3.metric("Growing", growing, delta=f"{growing} accounts", delta_color="normal")
        col4.metric("Declining", declining, delta=f"-{declining} accounts", delta_color="inverse")
        if product_filter != "Ed Controls":
            col5.metric("Adoption", f"{avg_adoption:.0f}%", help="Paying / Active users ratio")
        else:
            total_paying = int(ec_clients["paying_users"].sum())
            col5.metric("Paying Users", f"{total_paying:,}",
                        help="A+S roles (RASCI). Other roles (R,C,I) are active but non-paying.")

        st.markdown("---")

        # ─ Health overview: pie + bar ─
        col1, col2 = st.columns(2)

        with col1:
            health_counts = ec_clients["health"].value_counts().reset_index()
            health_counts.columns = ["Status", "Count"]
            status_map = {"growing": "Growing", "stable": "Stable", "declining": "Declining", "inactive": "Inactive", "unknown": "Unknown"}
            health_counts["Status"] = health_counts["Status"].map(status_map)
            color_map = {"Growing": GREEN, "Stable": BLUE, "Declining": RED, "Inactive": GREY, "Unknown": GREY}

            fig = px.pie(
                health_counts, values="Count", names="Status",
                title="Account Health Distribution",
                color="Status", color_discrete_map=color_map,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            au_am = ec_clients.groupby("owner_name").agg(
                active=("active_users", "sum"),
                paying=("paying_users", "sum"),
                accounts=("id", "count"),
            ).reset_index()
            au_am = au_am[au_am["owner_name"].isin(sales_owners)].sort_values("active", ascending=False)

            pay_label = "paying (A+S)" if product_filter == "Ed Controls" else "paying"
            au_am = au_am.rename(columns={"paying": pay_label})
            fig2 = px.bar(
                au_am, x="owner_name", y=["active", pay_label],
                title="Active & Paying Users per AM",
                labels={"owner_name": "", "value": "Users", "variable": ""},
                barmode="overlay",
                color_discrete_map={"active": LIGHT, pay_label: DARK},
            )
            fig2.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

        # ─ Top growers & top decliners ─
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Growers")
            growers = ec_clients[ec_clients["health"] == "growing"].sort_values("trend_pct", ascending=False).head(10)
            if not growers.empty:
                disp = growers[["name", "owner_name", "active_users", "trend_pct"]].copy()
                disp.columns = ["Client", "AM", "Active Users", "Growth %"]
                disp["Active Users"] = disp["Active Users"].astype(int)
                disp["Growth %"] = disp["Growth %"].apply(lambda x: f"+{x:.0f}%")
                st.dataframe(disp, hide_index=True, use_container_width=True)
            else:
                st.info("No growing accounts found.")

        with col2:
            st.subheader("Needs Attention — Declining Users")
            decliners = ec_clients[ec_clients["health"] == "declining"].sort_values("trend_pct").head(10)
            if not decliners.empty:
                disp = decliners[["name", "owner_name", "active_users", "trend_pct", "paying_users"]].copy()
                disp.columns = ["Client", "AM", "Active Users", "Decline %", "Paying"]
                disp["Active Users"] = disp["Active Users"].astype(int)
                disp["Paying"] = disp["Paying"].astype(int)
                disp["Decline %"] = disp["Decline %"].apply(lambda x: f"{x:.0f}%")
                st.dataframe(disp, hide_index=True, use_container_width=True)
            else:
                st.success("No declining accounts!")

        # ─ AM Health Scorecard ─
        st.markdown("---")
        st.subheader("AM Scorecard — User Health per Portfolio")

        scorecard = ec_with_trend.groupby("owner_name").agg(
            accounts=("id", "count"),
            active_q1_26=("q1_2026_avg", "sum"),
            active_prev=(user_prev_col, "sum"),
            total_paying=("paying_users", "sum"),
            growing=(user_trend_col, lambda x: (x > 5).sum()),
            stable=(user_trend_col, lambda x: ((x >= -5) & (x <= 5)).sum()),
            declining=(user_trend_col, lambda x: (x < -5).sum()),
            avg_trend=(user_trend_col, "mean"),
            avg_adoption=("adoption_pct", "mean"),
        ).reset_index()
        scorecard = scorecard[scorecard["owner_name"].isin(sales_owners)]
        scorecard["portfolio_pct"] = ((scorecard["active_q1_26"] - scorecard["active_prev"]) / scorecard["active_prev"] * 100).round(1)
        scorecard["health_score"] = (
            scorecard["growing"] * 2 + scorecard["stable"] * 1 - scorecard["declining"] * 2
        ) / scorecard["accounts"] * 100
        scorecard = scorecard.sort_values("health_score", ascending=False)

        disp = scorecard.rename(columns={
            "owner_name": "Account Manager", "accounts": "EdC Accounts",
            "active_q1_26": "Users Q1 '26", "active_prev": f"Users {prev_label}",
            "total_paying": "Paying",
            "growing": "Growing", "stable": "Stable", "declining": "Declining",
            "avg_trend": "Avg Trend", "portfolio_pct": f"Portfolio %",
            "avg_adoption": "Adoption %", "health_score": "Health Score",
        })
        disp["Avg Trend"] = disp["Avg Trend"].apply(lambda x: f"{x:+.1f}%")
        disp["Portfolio %"] = disp["Portfolio %"].apply(lambda x: f"{x:+.1f}%")
        disp["Adoption %"] = disp["Adoption %"].apply(lambda x: f"{x:.0f}%")
        disp["Health Score"] = disp["Health Score"].apply(lambda x: f"{x:.0f}")
        disp["Users Q1 '26"] = disp["Users Q1 '26"].astype(int)
        disp[f"Users {prev_label}"] = disp[f"Users {prev_label}"].astype(int)
        st.dataframe(disp, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: ADOPTION ALERT
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("Adoption Alert — Where are users dropping off?")
    st.caption("Accounts that need action to retain or activate users")

    if ec_clients.empty:
        st.warning("No Ed Controls active users data found.")
    else:
        alert_tab1, alert_tab2, alert_tab3 = st.tabs([
            "Declining Users",
            "Low Adoption",
            "No Contact",
        ])

        with alert_tab1:
            st.subheader("Accounts with declining active users")
            threshold_decline = st.slider("Minimum decline %", -50, 0, -10, step=5, key="decline")

            declining = ec_clients[ec_clients["trend_pct"] < threshold_decline].sort_values("trend_pct")

            if not declining.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Accounts", len(declining))
                col2.metric("Active Users at risk", int(declining["active_users"].sum()))
                col3.metric("Avg decline", f"{declining['trend_pct'].mean():.0f}%")

                am_dec = declining.groupby("owner_name").agg(
                    accounts=("id", "count"),
                    users_at_risk=("active_users", "sum"),
                ).reset_index().sort_values("users_at_risk", ascending=False)

                fig = px.bar(
                    am_dec, x="owner_name", y="users_at_risk", text="accounts",
                    title="Users at risk per AM (declining accounts)",
                    labels={"owner_name": "", "users_at_risk": "Active Users", "accounts": ""},
                    color_discrete_sequence=[RED],
                )
                fig.update_traces(texttemplate="%{text} accounts", textposition="outside")
                fig.update_layout(height=350, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                detail = declining[["name", "owner_name", "active_users", "trend_pct",
                                     "paying_users", "last_contacted", "revenue_12m"]].copy()
                detail.columns = ["Client", "AM", "Active Users", "Trend %", "Paying",
                                   "Last Contact", "Revenue 12m"]
                detail["Active Users"] = detail["Active Users"].astype(int)
                detail["Paying"] = detail["Paying"].astype(int)
                detail["Trend %"] = detail["Trend %"].apply(lambda x: f"{x:.0f}%")
                detail["Last Contact"] = detail["Last Contact"].dt.strftime("%Y-%m-%d").fillna("-")
                detail["Revenue 12m"] = detail["Revenue 12m"].apply(lambda x: f"€{x:,.0f}")
                st.dataframe(detail, hide_index=True, use_container_width=True, height=400)
            else:
                st.success("No accounts with declining users!")

        with alert_tab2:
            if product_filter == "Ed Controls":
                st.subheader("Paying vs Active Users (RASCI)")
                st.caption(
                    "In Ed Controls, only A (Accountable) and S (Support) roles are paying. "
                    "R, C, and I roles are active but non-paying by design. "
                    "A low ratio is normal — it reflects the RASCI model, not underutilization."
                )
                has_both = ec_clients[
                    (ec_clients["active_users"] > 5) &
                    (ec_clients["paying_users"].notna())
                ].sort_values("adoption_pct")
                if not has_both.empty:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accounts", len(has_both))
                    col2.metric("Total Active", int(has_both["active_users"].sum()))
                    col3.metric("Total Paying (A+S)", int(has_both["paying_users"].sum()))

                    detail = has_both[["name", "owner_name", "active_users", "paying_users",
                                       "adoption_pct", "trend_pct"]].copy()
                    detail.columns = ["Client", "AM", "Active (all roles)", "Paying (A+S)", "A+S %", "Trend %"]
                    detail["Active (all roles)"] = detail["Active (all roles)"].astype(int)
                    detail["Paying (A+S)"] = detail["Paying (A+S)"].astype(int)
                    detail["A+S %"] = detail["A+S %"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
                    detail["Trend %"] = detail["Trend %"].apply(lambda x: f"{x:+.0f}%" if pd.notna(x) else "-")
                    st.dataframe(detail, hide_index=True, use_container_width=True, height=400)
            else:
                st.subheader("Low adoption — licenses underutilized")
                st.caption("Accounts where paying/active user ratio is low.")

                adoption_threshold = st.slider("Maximum adoption %", 10, 80, 40, step=5, key="adoption")

                low_adoption = ec_clients[
                    (ec_clients["adoption_pct"].notna()) &
                    (ec_clients["adoption_pct"] < adoption_threshold) &
                    (ec_clients["active_users"] > 5)
                ].sort_values("adoption_pct")

                if not low_adoption.empty:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accounts", len(low_adoption))
                    col2.metric("Unused licenses", int(low_adoption["active_users"].sum() - low_adoption["paying_users"].sum()))
                    col3.metric("Avg adoption", f"{low_adoption['adoption_pct'].mean():.0f}%")

                    detail = low_adoption[["name", "owner_name", "active_users", "paying_users",
                                            "adoption_pct", "trend_pct"]].copy()
                    detail.columns = ["Client", "AM", "Active", "Paying", "Adoption %", "Trend %"]
                    detail["Active"] = detail["Active"].astype(int)
                    detail["Paying"] = detail["Paying"].astype(int)
                    detail["Adoption %"] = detail["Adoption %"].apply(lambda x: f"{x:.0f}%")
                    detail["Trend %"] = detail["Trend %"].apply(lambda x: f"{x:+.0f}%" if pd.notna(x) else "-")
                    st.dataframe(detail, hide_index=True, use_container_width=True, height=400)
                else:
                    st.success(f"All accounts above {adoption_threshold}% adoption!")

        with alert_tab3:
            st.subheader("Active clients without recent contact")
            st.caption("Accounts with active users but no HubSpot contact in 60+ days")

            contact_days = st.slider("Days without contact", 30, 180, 60, step=15, key="contact")

            no_contact = ec_clients[
                (ec_clients["active_users"] > 0) &
                (ec_clients["days_no_contact"] > contact_days)
            ].sort_values("active_users", ascending=False)

            if not no_contact.empty:
                col1, col2 = st.columns(2)
                col1.metric("Accounts without contact", len(no_contact))
                col2.metric("Active Users without attention", int(no_contact["active_users"].sum()))

                detail = no_contact[["name", "owner_name", "active_users", "trend_pct",
                                      "days_no_contact", "last_contacted"]].copy()
                detail.columns = ["Client", "AM", "Active Users", "Trend %",
                                   "Days No Contact", "Last Contact"]
                detail["Active Users"] = detail["Active Users"].astype(int)
                detail["Trend %"] = detail["Trend %"].apply(lambda x: f"{x:+.0f}%" if pd.notna(x) else "-")
                detail["Last Contact"] = detail["Last Contact"].dt.strftime("%Y-%m-%d").fillna("-")
                st.dataframe(detail, hide_index=True, use_container_width=True, height=400)
            else:
                st.success("All active accounts have recent contact!")


# ═══════════════════════════════════════════════════════════════
# TAB 5: NPS (quarterly)
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.header("NPS — Make People Happy")
    st.caption("Net Promoter Score per client · Q1 2026 · Source: NPS survey (Ed Controls)")

    if nps_clients.empty:
        st.warning("No NPS data found for Q1 2026.")
    else:
        # Top metrics — response-weighted NPS (not simple average across clients)
        total_responses = int(nps_clients["nps_responses_q1_26"].sum())
        weights = nps_clients["nps_responses_q1_26"].fillna(1)
        avg_nps = (nps_clients[nps_cur_col] * weights).sum() / weights.sum()

        # NPS buckets
        promoters = len(nps_clients[nps_clients[nps_cur_col] > 50])
        detractors = len(nps_clients[nps_clients[nps_cur_col] < 0])

        col1, col2, col3, col4, col5 = st.columns(5)
        nps_target = 50 if product_filter != "FlexWhere" else 32
        col1.metric("NPS Q1 '26", f"{avg_nps:.0f}",
                     help=f"Response-weighted average. Target: {nps_target}")
        col2.metric("Clients with NPS", len(nps_clients))
        col3.metric("Responses Q1 '26", f"{total_responses:,}")
        col4.metric("Promoters (NPS > 50)", promoters, delta=f"{promoters} clients", delta_color="normal")
        col5.metric("Detractors (NPS < 0)", detractors, delta=f"-{detractors} clients", delta_color="inverse")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                nps_clients, x=nps_cur_col, nbins=20,
                title="NPS Distribution Q1 2026",
                labels={nps_cur_col: "NPS Score", "count": "Number of clients"},
                color_discrete_sequence=[BLUE],
            )
            fig.add_vline(x=0, line_dash="dash", line_color="grey", annotation_text="Neutral")
            nps_target = 50 if product_filter != "FlexWhere" else 32
            fig.add_vline(x=nps_target, line_dash="dash", line_color=GREEN,
                          annotation_text=f"Target: {nps_target}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Response-weighted NPS per AM
            am_nps_data = nps_clients[nps_clients["owner_name"].isin(sales_owners)].copy()
            am_nps_data["_w"] = am_nps_data["nps_responses_q1_26"].fillna(1)
            am_nps_data["_wnps"] = am_nps_data[nps_cur_col] * am_nps_data["_w"]
            nps_am = am_nps_data.groupby("owner_name").agg(
                _wnps_sum=("_wnps", "sum"),
                _w_sum=("_w", "sum"),
                clients=("id", "count"),
                responses=("nps_responses_q1_26", "sum"),
            ).reset_index()
            nps_am["avg_nps"] = nps_am["_wnps_sum"] / nps_am["_w_sum"]
            nps_am = nps_am.drop(columns=["_wnps_sum", "_w_sum"])
            nps_am = nps_am.sort_values("avg_nps", ascending=False)

            if not nps_am.empty:
                fig2 = px.bar(
                    nps_am, x="owner_name", y="avg_nps", text="clients",
                    title=f"Average NPS Q1 '26 per Account Manager",
                    labels={"owner_name": "", "avg_nps": "Avg NPS", "clients": ""},
                    color="avg_nps",
                    color_continuous_scale=[[0, RED], [0.5, ORANGE], [1, GREEN]],
                )
                fig2.update_traces(texttemplate="%{text} clients", textposition="outside")
                fig2.update_layout(height=400, xaxis_tickangle=-45, coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

        # NPS + Active Users combined view
        if not ec_clients.empty:
            st.subheader("NPS vs Active Users — Are happy clients also growing clients?")
            both = nps_clients.merge(
                ec_clients[["id", "active_users", "trend_pct"]].rename(columns={"id": "id"}),
                on="id", how="inner", suffixes=("", "_ec"),
            )
            if not both.empty and len(both) > 3:
                fig3 = px.scatter(
                    both, x=nps_cur_col, y="trend_pct",
                    size="active_users", color="owner_name",
                    hover_name="name",
                    title="NPS Q1 '26 vs User Growth Trend",
                    labels={nps_cur_col: "NPS Score Q1 '26", "trend_pct": "User Trend %", "owner_name": "AM"},
                )
                fig3.add_hline(y=0, line_dash="dash", line_color="grey")
                fig3.add_vline(x=0, line_dash="dash", line_color="grey")
                fig3.update_layout(height=450)
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Top right = happy & growing. Bottom left = unhappy & shrinking.")

        # Top & Bottom NPS tables
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Highest NPS Q1 '26")
            top_nps = nps_clients.nlargest(10, nps_cur_col)[["name", "owner_name", nps_cur_col, "nps_responses_q1_26"]].copy()
            top_nps.columns = ["Client", "AM", "NPS", "Responses"]
            top_nps["NPS"] = top_nps["NPS"].astype(int)
            st.dataframe(top_nps, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("Lowest NPS Q1 '26 — Needs attention")
            bottom_nps = nps_clients.nsmallest(10, nps_cur_col)[["name", "owner_name", nps_cur_col, "nps_responses_q1_26"]].copy()
            bottom_nps.columns = ["Client", "AM", "NPS", "Responses"]
            bottom_nps["NPS"] = bottom_nps["NPS"].astype(int)
            st.dataframe(bottom_nps, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: ACCOUNT DETAILS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("Account Details")

    selected_am = st.selectbox("Account Manager", ["All"] + sales_owners)

    df = companies.copy()
    if selected_am != "All":
        df = df[df["owner_name"] == selected_am]

    active_only = st.checkbox("Active clients only (revenue > 0)", value=True)
    if active_only:
        df = df[df["revenue_12m"] > 0]

    # Health-based risk scoring
    df["risk_score"] = 0
    df.loc[df["trend_pct"].fillna(0) < -10, "risk_score"] += 3
    df.loc[df["trend_pct"].fillna(0) < -20, "risk_score"] += 2
    df.loc[df[nps_cur_col].fillna(100) < 0, "risk_score"] += 2
    if product_filter != "Ed Controls":
        df.loc[df["adoption_pct"].fillna(100) < 30, "risk_score"] += 2
    df.loc[df["days_no_contact"] > 90, "risk_score"] += 1
    bill_interval = df["billing_interval_days"].fillna(30)
    overdue = df["days_no_invoice"] - bill_interval
    df.loc[overdue > 30, "risk_score"] += 1

    sort_col = st.radio(
        "Sort by",
        ["Risk (high→low)", "Active Users (high→low)", "Revenue (high→low)", "Name"],
        horizontal=True,
    )
    sort_map = {
        "Risk (high→low)": (["risk_score", "active_users"], [False, False]),
        "Active Users (high→low)": (["active_users"], [False]),
        "Revenue (high→low)": (["revenue_12m"], [False]),
        "Name": (["name"], [True]),
    }
    cols, asc = sort_map[sort_col]
    df = df.sort_values(cols, ascending=asc, na_position="last")

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Customers", len(df))
    has_au = df["active_users"].notna()
    col2.metric("With EdC data", int(has_au.sum()))
    col3.metric("Active Users", f"{df['active_users'].sum():,.0f}" if has_au.any() else "-")
    col4.metric("Revenue 12m", f"€{df['revenue_12m'].sum():,.0f}")
    col5.metric("At risk accounts", int((df["risk_score"] >= 3).sum()))

    # Table
    display = df[["name", "owner_name", "active_users", "paying_users", "trend_pct",
                   "adoption_pct", "health", nps_cur_col, "revenue_12m", "billing_freq",
                   "last_contacted", "risk_score"]].copy()
    adopt_label = "A+S %" if product_filter == "Ed Controls" else "Adoption %"
    pay_label = "Paying (A+S)" if product_filter == "Ed Controls" else "Paying"
    display.columns = ["Client", "AM", "Active", pay_label, "Trend %",
                        adopt_label, "Health", "NPS Q1", "Revenue 12m", "Freq",
                        "Last Contact", "Risk"]

    display["Active"] = display["Active"].fillna(-1).astype(int).astype(str).replace("-1", "-")
    display[pay_label] = display[pay_label].fillna(-1).astype(int).astype(str).replace("-1", "-")
    display["Trend %"] = display["Trend %"].apply(lambda x: f"{x:+.0f}%" if pd.notna(x) else "-")
    display[adopt_label] = display[adopt_label].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
    display["NPS Q1"] = display["NPS Q1"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
    display["Revenue 12m"] = display["Revenue 12m"].apply(lambda x: f"€{x:,.0f}")
    display["Last Contact"] = display["Last Contact"].dt.strftime("%Y-%m-%d").fillna("-")
    status_emoji = {"growing": "🟢", "stable": "🔵", "declining": "🔴", "inactive": "⚫", "unknown": "⚪"}
    display["Health"] = display["Health"].map(status_emoji).fillna("⚪")

    st.dataframe(display, hide_index=True, use_container_width=True, height=600)


# ═══════════════════════════════════════════════════════════════
# TAB 4: SALES ACTIVITY
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("Sales Activity — Means, not goal")
    st.caption("Activity is relevant when it leads to more happy users. Shown here for context.")

    if eng_df.empty:
        st.warning("No engagement data found.")
    else:
        team_df = eng_df[eng_df["owner_name"].isin(sales_owners)]

        # Filter to Q1 2026 only
        q1_team = team_df[team_df["month"] >= "2026-01-01"]

        col1, col2 = st.columns(2)

        with col1:
            q1_totals = (
                q1_team.groupby("owner_name")[["calls", "meetings", "tasks", "notes"]]
                .sum().reset_index()
                .sort_values("calls", ascending=False)
            )
            q1_totals["total"] = q1_totals["calls"] + q1_totals["meetings"] + q1_totals["tasks"]

            fig = px.bar(
                q1_totals.melt(id_vars="owner_name", value_vars=["calls", "meetings", "tasks"]),
                x="owner_name", y="value", color="variable",
                title="Activity per AM — Q1 2026",
                labels={"owner_name": "", "value": "Count", "variable": "Type"},
                color_discrete_map={"calls": BLUE, "meetings": DARK, "tasks": LIGHT},
            )
            fig.update_layout(barmode="stack", xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            monthly = (
                team_df.groupby(["month", "owner_name"])[["calls", "meetings", "tasks"]]
                .sum().reset_index()
            )
            monthly["total"] = monthly["calls"] + monthly["meetings"] + monthly["tasks"]

            fig2 = px.line(
                monthly, x="month", y="total", color="owner_name",
                title="Monthly trend (calls + meetings + tasks)",
                labels={"month": "", "total": "Count", "owner_name": "AM"},
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # ─ Effort vs Impact: scatter ─
        if not ec_clients.empty:
            st.subheader("Effort vs Impact — Does activity lead to more happy users?")
            st.caption(f"Q1 2026 effort vs user growth ({user_trend_col.upper()})")

            q1_totals_merged = (
                q1_team.groupby("owner_name")[["calls", "meetings", "tasks"]]
                .sum().reset_index()
            )
            q1_totals_merged["effort_q1"] = q1_totals_merged["calls"] + q1_totals_merged["meetings"] + q1_totals_merged["tasks"]

            ec_trend = ec_clients[ec_clients[user_trend_col].notna()]
            if not ec_trend.empty:
                am_health = ec_trend.groupby("owner_name").agg(
                    user_trend=(user_trend_col, "mean"),
                    total_active=("active_users", "sum"),
                    accounts=("id", "count"),
                ).reset_index()
                effort_impact = q1_totals_merged.merge(am_health, on="owner_name", how="inner")
                effort_impact = effort_impact[effort_impact["owner_name"].isin(sales_owners)]

                if not effort_impact.empty:
                    fig3 = px.scatter(
                        effort_impact, x="effort_q1", y="user_trend",
                        size="total_active", color="owner_name",
                        title=f"Sales effort Q1 2026 vs User growth ({user_trend_col.upper()})",
                        labels={
                            "effort_q1": "Activity Q1 2026 (calls+meetings+tasks)",
                            "user_trend": f"Avg user growth {user_trend_col.upper()} %",
                            "owner_name": "AM",
                        },
                    )
                    fig3.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="zero growth")
                    fig3.update_layout(height=450)
                    st.plotly_chart(fig3, use_container_width=True)
                    st.caption(
                        "Above the line = users growing vs last year. "
                        "Below the line = shrinking. Size = total active users in portfolio."
                    )
