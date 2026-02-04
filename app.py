import os
import streamlit as st
import pandas as pd
from databricks import sql
from dotenv import load_dotenv

# Ensure .env is loaded (so "streamlit run app.py" works reliably)
load_dotenv()

# ----------------------------
# Config (from .env)
# ----------------------------
CATALOG = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")
TABLE = "titanic"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

MIN_N = 30
SPREAD_THRESHOLD = 0.20  # Reflect 기준: spread가 크면 조합 분석까지 수행

# ----------------------------
# Databricks helpers
# ----------------------------
@st.cache_resource
def get_db_connection():
    hostname = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    missing = [k for k in ["DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"]
               if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Databricks connection env vars missing: {', '.join(missing)}\n"
            "Check your .env file and restart Streamlit."
        )

    return sql.connect(
        server_hostname=hostname,
        http_path=http_path,
        access_token=token
    )

def run_sql(query: str) -> pd.DataFrame:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)

# ----------------------------
# SQL builders (Databricks SQL)
# ----------------------------
def q_smoke_test():
    return "SELECT 1 AS ok"

def q_describe_table():
    return f"DESCRIBE TABLE {FULL_TABLE}"

def q_baseline():
    return f"""
    SELECT
      COUNT(*) AS n,
      AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    """

def q_group_rate(col: str):
    # For simple categorical dims: Sex, Pclass, Embarked ...
    return f"""
    SELECT
      {col} AS group_key,
      COUNT(*) AS n,
      AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    WHERE {col} IS NOT NULL
    GROUP BY {col}
    HAVING COUNT(*) >= {MIN_N}
    ORDER BY survival_rate DESC
    """

def q_age_group_rate():
    return f"""
    WITH base AS (
      SELECT
        Survived,
        CASE
          WHEN Age IS NULL THEN 'Unknown'
          WHEN Age < 10 THEN '0-9'
          WHEN Age < 20 THEN '10-19'
          WHEN Age < 30 THEN '20-29'
          WHEN Age < 40 THEN '30-39'
          WHEN Age < 50 THEN '40-49'
          WHEN Age < 60 THEN '50-59'
          ELSE '60+'
        END AS age_group
      FROM {FULL_TABLE}
    )
    SELECT
      age_group AS group_key,
      COUNT(*) AS n,
      AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM base
    GROUP BY age_group
    HAVING COUNT(*) >= {MIN_N}
    ORDER BY survival_rate DESC
    """

def q_combo_sex_pclass():
    return f"""
    SELECT
      Sex,
      Pclass,
      COUNT(*) AS n,
      AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    WHERE Sex IS NOT NULL AND Pclass IS NOT NULL
    GROUP BY Sex, Pclass
    HAVING COUNT(*) >= {MIN_N}
    ORDER BY survival_rate DESC
    """

# ----------------------------
# Agent logic (Plan → Execute → Reflect → Stop)
# ----------------------------
def compute_spread(df: pd.DataFrame) -> float:
    if df.empty or "survival_rate" not in df.columns:
        return 0.0
    s = pd.to_numeric(df["survival_rate"], errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float(s.max() - s.min())

def agent_run(dims: list[str]) -> dict:
    state = {
        "goal": "Find top 3 factors impacting survival rate",
        "plan": [],
        "results": {},
        "scores": {},
        "top3": [],
        "actions": [],
        "sql_log": [],
        "combo_done": False,
        "errors": [],
    }

    # 1) Plan
    state["plan"] = [
        "Run smoke test (connectivity check)",
        "Describe table schema (sanity check)",
        "Run baseline survival rate",
        "Compute survival rate by candidate dimensions",
        "Score dimensions by spread (max-min survival rate)",
        "Pick top 3 dimensions",
        "If top1 spread is high, run combo analysis (Sex+Pclass)",
        "Summarize insights with evidence + reproducible SQL"
    ]

    try:
        # 2) Execute - smoke test
        q0 = q_smoke_test()
        state["sql_log"].append(q0)
        state["results"]["smoke_test"] = run_sql(q0)

        # 3) Execute - describe table
        qd = q_describe_table()
        state["sql_log"].append(qd)
        state["results"]["schema"] = run_sql(qd)

        # 4) Execute - baseline
        qb = q_baseline()
        state["sql_log"].append(qb)
        baseline = run_sql(qb)
        state["results"]["baseline"] = baseline

        # 5) Execute - each dimension
        for d in dims:
            if d == "AgeGroup":
                q = q_age_group_rate()
            else:
                q = q_group_rate(d)

            state["sql_log"].append(q)
            df = run_sql(q)
            state["results"][d] = df
            state["scores"][d] = compute_spread(df)

        # 6) Reflect - rank by spread
        ranked = sorted(state["scores"].items(), key=lambda x: x[1], reverse=True)
        state["top3"] = ranked[:3]

        # 7) Reflect decision: combo?
        if state["top3"]:
            top1_dim, top1_spread = state["top3"][0]
            if top1_spread >= SPREAD_THRESHOLD:
                qc = q_combo_sex_pclass()
                state["sql_log"].append(qc)
                combo = run_sql(qc)
                state["results"]["Sex+Pclass"] = combo
                state["combo_done"] = True
                state["actions"].append(
                    f"Combo analysis executed because top spread={top1_spread:.3f} (>= {SPREAD_THRESHOLD})"
                )
            else:
                state["actions"].append(
                    f"Stopped without combo: top spread={top1_spread:.3f} (< {SPREAD_THRESHOLD})"
                )

    except Exception as e:
        state["errors"].append(str(e))

    return state

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Titanic Survival Rate Agent", layout="wide")
st.title("🛟 Titanic 생존율 비교 Agent (Streamlit + Databricks)")

with st.sidebar:
    st.subheader("Databricks")
    st.write(f"Catalog: `{CATALOG}`")
    st.write(f"Schema: `{SCHEMA}`")
    st.write(f"Table: `{TABLE}`")
    st.write(f"Full: `{FULL_TABLE}`")

    st.subheader("Agent settings")
    min_n = st.number_input("MIN_N (min group size)", min_value=1, max_value=1000, value=MIN_N, step=1)
    spread_th = st.number_input("SPREAD_THRESHOLD", min_value=0.0, max_value=1.0, value=SPREAD_THRESHOLD, step=0.05)

    # Update globals from UI (simple approach)
    MIN_N = int(min_n)
    SPREAD_THRESHOLD = float(spread_th)

    st.subheader("Dimensions to compare")
    dims_selected = st.multiselect(
        "Pick dimensions",
        ["Sex", "Pclass", "Embarked", "AgeGroup"],
        default=["Sex", "Pclass", "Embarked", "AgeGroup"]
    )

    run_btn = st.button("Run Agent", type="primary")

if run_btn:
    if not dims_selected:
        st.error("Please select at least one dimension.")
        st.stop()

    with st.spinner("Agent running... (Plan → Execute → Reflect → Stop)"):
        state = agent_run(dims_selected)

    if state["errors"]:
        st.error("Agent failed.")
        for err in state["errors"]:
            st.code(err)
        st.stop()

    st.subheader("✅ Plan")
    st.write("\n".join([f"{i+1}. {x}" for i, x in enumerate(state["plan"])]))

    st.subheader("🔌 Smoke test")
    st.dataframe(state["results"]["smoke_test"], use_container_width=True)

    st.subheader("🧾 Schema (DESCRIBE TABLE)")
    st.dataframe(state["results"]["schema"], use_container_width=True)

    st.subheader("📌 Baseline")
    st.dataframe(state["results"]["baseline"], use_container_width=True)

    st.subheader("📊 Dimension scores (spread)")
    score_df = pd.DataFrame(
        [{"dimension": k, "spread": v} for k, v in state["scores"].items()]
    ).sort_values("spread", ascending=False)

    c1, c2 = st.columns(2)
    c1.dataframe(score_df, use_container_width=True)
    if not score_df.empty:
        c1.bar_chart(score_df.set_index("dimension")["spread"])

    with c2:
        st.markdown("### Top 3 dimensions")
        for i, (dim, sp) in enumerate(state["top3"], start=1):
            st.write(f"{i}) **{dim}** — spread = `{sp:.3f}`")
        st.markdown("### Reflect actions")
        for a in state["actions"]:
            st.write(f"- {a}")

    st.subheader("🔎 Detailed results")
    for k, df in state["results"].items():
        if k in ("smoke_test", "schema", "baseline"):
            continue
        with st.expander(f"Show: {k}"):
            st.dataframe(df, use_container_width=True)

    st.subheader("🧾 SQL log (reproducible)")
    for i, q in enumerate(state["sql_log"], start=1):
        with st.expander(f"SQL {i}"):
            st.code(q, language="sql")

    st.subheader("🧠 Final summary")
    base = state["results"]["baseline"]
    if not base.empty:
        n = int(base.loc[0, "n"])
        sr = float(base.loc[0, "survival_rate"])
        st.write(f"- Overall survival rate: **{sr:.3f}** (n={n})")
    else:
        st.write("- Overall survival rate: N/A")

    if state["top3"]:
        st.write("- Top factors by spread: " + ", ".join([f"{d}({sp:.2f})" for d, sp in state["top3"]]))
    else:
        st.write("- Top factors by spread: N/A")

    if state["combo_done"]:
        st.write("- Combo analysis executed: **Sex + Pclass**")

else:
    st.info("왼쪽에서 **Run Agent**를 눌러 실행해봐.")
