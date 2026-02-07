import os
import json
import re
from typing import Literal
from typing_extensions import TypedDict
import pandas as pd
import streamlit as st
from databricks import sql
from dotenv import load_dotenv
import google.generativeai as genai

# --------------------------------------------------
# Load env
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Config
# --------------------------------------------------
CATALOG = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")
TABLE = "titanic"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

MIN_N = 30

# --------------------------------------------------
# Gemini setup
# --------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing in .env")

genai.configure(api_key=GOOGLE_API_KEY)

# ✅ 요청한 모델로 변경
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# --------------------------------------------------
# Gemini response schema (JSON)
# --------------------------------------------------
class ReflectDecision(TypedDict):
    decision: Literal["continue", "stop"]
    next_action: Literal["combo_analysis", "none"]
    target: Literal["Sex+Pclass", "none"]
    reason: str

# --------------------------------------------------
# Databricks helpers
# --------------------------------------------------
@st.cache_resource
def get_db_connection():
    host = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")

    missing = [k for k in ["DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN"] if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Databricks env vars missing: {', '.join(missing)}")

    return sql.connect(server_hostname=host, http_path=http_path, access_token=token)

def run_sql(query: str) -> pd.DataFrame:
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)

# --------------------------------------------------
# SQL builders
# --------------------------------------------------
def q_smoke():
    return "SELECT 1 AS ok"

def q_baseline():
    return f"""
    SELECT COUNT(*) AS n,
           AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    """

def q_group(col):
    return f"""
    SELECT {col} AS group_key,
           COUNT(*) AS n,
           AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    WHERE {col} IS NOT NULL
    GROUP BY {col}
    HAVING COUNT(*) >= {MIN_N}
    ORDER BY survival_rate DESC
    """

def q_age_group():
    return f"""
    WITH base AS (
      SELECT Survived,
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
    SELECT age_group AS group_key,
           COUNT(*) AS n,
           AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM base
    GROUP BY age_group
    HAVING COUNT(*) >= {MIN_N}
    ORDER BY survival_rate DESC
    """

def q_combo_sex_pclass():
    return f"""
    SELECT Sex, Pclass,
           COUNT(*) AS n,
           AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    WHERE Sex IS NOT NULL AND Pclass IS NOT NULL
    GROUP BY Sex, Pclass
    HAVING COUNT(*) >= {MIN_N}
    ORDER BY survival_rate DESC
    """

# --------------------------------------------------
# Robust JSON extraction
# --------------------------------------------------
def extract_json_from_text(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Empty LLM response text")

    t = text.strip()

    # 1) ```json ... ``` 코드블록 우선
    m = re.search(r"```json\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1).strip())

    # 2) ``` ... ``` 코드블록 안에 JSON이 있을 수도
    m2 = re.search(r"```\s*(\{.*?\})\s*```", t, flags=re.DOTALL)
    if m2:
        return json.loads(m2.group(1).strip())

    # 3) 텍스트 전체에서 첫 { 와 마지막 } 사이
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end + 1].strip()
        return json.loads(candidate)

    raise ValueError("No JSON object found in LLM response")

# --------------------------------------------------
# Gemini Reflect (with 1 retry)
# --------------------------------------------------
def _resp_text_and_reason(resp) -> tuple[str, str]:
    reason = ""
    try:
        cand = resp.candidates[0]
        reason_val = getattr(cand, "finish_reason", None)
        reason = "" if reason_val is None else str(reason_val)
    except Exception:
        reason = ""

    text = None
    try:
        text = resp.text
    except Exception:
        text = None

    if text:
        return text.strip(), reason

    try:
        parts = resp.candidates[0].content.parts
        joined = "".join([p.text for p in parts if hasattr(p, "text")])
        return (joined or "").strip(), reason
    except Exception:
        return "", reason

def call_gemini(prompt: str, *, json_mode: bool = False) -> tuple[str, str]:
    gen_cfg = {
        "temperature": 0.1,
        "max_output_tokens": 2048,
    }
    if json_mode:
        gen_cfg["response_mime_type"] = "application/json"
        gen_cfg["response_schema"] = ReflectDecision

    resp = GEMINI_MODEL.generate_content(
        prompt,
        generation_config=gen_cfg,
    )
    # resp.text 가 비어 있을 수 있어 안전 처리
    return _resp_text_and_reason(resp)

def call_gemini_reflect(prompt: str) -> tuple[dict, str, str, str, str, str]:
    # 1차 호출
    raw1, reason1 = call_gemini(prompt, json_mode=True)
    raw2 = ""
    err = ""
    try:
        return extract_json_from_text(raw1), raw1, raw2, err, reason1, ""
    except Exception as e1:
        # 2차 재시도: 더 강하게 "JSON만" 요구
        reprompt = f"""
Return ONLY valid JSON. No markdown. No explanation. No code fences.

JSON schema:
{{
  "decision": "continue" or "stop",
  "next_action": "combo_analysis" or "none",
  "target": "Sex+Pclass" or "none",
  "reason": "short"
}}

If unsure, output:
{{"decision":"stop","next_action":"none","target":"none","reason":"insufficient confidence"}}

Context:
{prompt}
"""
        raw2, reason2 = call_gemini(reprompt, json_mode=True)
        try:
            return extract_json_from_text(raw2), raw1, raw2, err, reason1, reason2
        except Exception as e2:
            err = f"{type(e1).__name__}: {e1} / {type(e2).__name__}: {e2}"
            fallback = {
                "decision": "stop",
                "next_action": "none",
                "target": "none",
                "reason": "llm_response_not_json"
            }
            return fallback, raw1, raw2, err, reason1, reason2

def build_reflect_prompt(goal, scores, executed):
    score_txt = "\n".join([f"- {k}: spread={v:.3f}" for k, v in scores.items()])
    done_txt = ", ".join(executed)

    return f"""
You are an agentic AI responsible ONLY for deciding the next analysis step.

Goal:
{goal}

Current analysis results (higher spread = stronger impact):
{score_txt}

Already executed:
{done_txt}

Rules:
- Prefer simple explanations.
- Avoid unnecessary computation.
- If results are sufficient, decide to stop.

Respond in JSON (no extra keys):
{{
  "decision": "continue | stop",
  "next_action": "combo_analysis | none",
  "target": "Sex+Pclass | none",
  "reason": "short explanation"
}}
"""

# --------------------------------------------------
# Agent logic
# --------------------------------------------------
def compute_spread(df):
    if df.empty:
        return 0.0
    s = pd.to_numeric(df["survival_rate"], errors="coerce").dropna()
    return float(s.max() - s.min()) if not s.empty else 0.0

def run_agent(dims):
    state = {
        "goal": "Identify top factors affecting Titanic survival",
        "results": {},
        "scores": {},
        "actions": [],
        "sql_log": [],
        "llm_decision": {},
        "llm_raw_1": "",
        "llm_raw_2": "",
        "llm_error": "",
        "llm_finish_reason_1": "",
        "llm_finish_reason_2": ""
    }

    # Smoke test
    qs = q_smoke()
    state["sql_log"].append(qs)
    _ = run_sql(qs)

    # Baseline
    qb = q_baseline()
    state["sql_log"].append(qb)
    state["results"]["baseline"] = run_sql(qb)

    # Dimension scan
    for d in dims:
        q = q_age_group() if d == "AgeGroup" else q_group(d)
        state["sql_log"].append(q)
        df = run_sql(q)
        state["results"][d] = df
        state["scores"][d] = compute_spread(df)

    # Reflect (Gemini decides)
    prompt = build_reflect_prompt(state["goal"], state["scores"], list(state["results"].keys()))
    decision, raw1, raw2, llm_error, reason1, reason2 = call_gemini_reflect(prompt)
    state["llm_decision"] = decision
    state["llm_raw_1"] = raw1
    state["llm_raw_2"] = raw2
    state["llm_error"] = llm_error
    state["llm_finish_reason_1"] = reason1
    state["llm_finish_reason_2"] = reason2
    if llm_error:
        state["actions"].append(f"LLM JSON parse failed; fallback used. {llm_error}")

    # Route decision
    if decision.get("decision") == "continue" and decision.get("next_action") == "combo_analysis":
        if decision.get("target") == "Sex+Pclass":
            qc = q_combo_sex_pclass()
            state["sql_log"].append(qc)
            state["results"]["Sex+Pclass"] = run_sql(qc)
            state["actions"].append("Gemini decided to run Sex+Pclass combo analysis")
        else:
            state["actions"].append(f"Gemini requested combo_analysis but target={decision.get('target')} not supported yet")
    else:
        state["actions"].append("Gemini decided to stop")

    return state

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("🧠 Agentic AI (Gemini 2.5 Flash) – Titanic Survival Analysis")

with st.sidebar:
    st.caption(f"Table: {FULL_TABLE}")
    dims = st.multiselect(
        "Dimensions",
        ["Sex", "Pclass", "Embarked", "AgeGroup"],
        default=["Sex", "Pclass", "Embarked", "AgeGroup"]
    )
    run = st.button("Run Agent", type="primary")

if run:
    with st.spinner("Agent running..."):
        state = run_agent(dims)

    st.subheader("📊 Dimension Scores (spread)")
    score_df = pd.DataFrame(
        [{"dimension": k, "spread": v} for k, v in state["scores"].items()]
    ).sort_values("spread", ascending=False)
    st.dataframe(score_df, use_container_width=True)
    if not score_df.empty:
        st.bar_chart(score_df.set_index("dimension")["spread"])

    st.subheader("🤖 Gemini Reflect Decision (parsed JSON)")
    st.json(state["llm_decision"])

    st.subheader("🧭 Actions")
    for a in state["actions"]:
        st.write(f"- {a}")

    with st.expander("🧩 Gemini raw response (debug)"):
        st.text(state.get("llm_raw_1", ""))
        if state.get("llm_finish_reason_1"):
            st.text(f"[finish_reason] {state.get('llm_finish_reason_1')}")
        if state.get("llm_raw_2"):
            st.text("\n--- retry ---\n")
            st.text(state.get("llm_raw_2", ""))
            if state.get("llm_finish_reason_2"):
                st.text(f"[finish_reason] {state.get('llm_finish_reason_2')}")
    if state.get("llm_error"):
        st.warning(state.get("llm_error"))

    st.subheader("🔎 Analysis Results")
    for k, df in state["results"].items():
        with st.expander(k):
            st.dataframe(df, use_container_width=True)

    st.subheader("🧾 SQL Log")
    for q in state["sql_log"]:
        st.code(q, language="sql")

else:
    st.info("Run Agent 버튼을 눌러 시작하세요.")
