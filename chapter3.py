# chapter3.py
import os
import json
import re
import ast
from typing import Literal, Any, Dict, List, Tuple
from typing_extensions import TypedDict

import pandas as pd
import streamlit as st
from databricks import sql
from dotenv import load_dotenv
import google.generativeai as genai

# ==================================================
# Load env
# ==================================================
load_dotenv()

# ==================================================
# Config
# ==================================================
CATALOG = os.getenv("DATABRICKS_CATALOG", "workspace")
SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")
TABLE = "titanic"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

DEFAULT_MIN_N = 30

# Hybrid fallback thresholds (only used if LLM fails)
FALLBACK_SPREAD_TRIGGER = 0.25

# Token budgets (important!)
MAX_TOKENS_CRITERIA = 1024     # Increased to prevent truncation even if model is verbose
MAX_TOKENS_DECISION = 2024    # ✅ requested

# ==================================================
# Gemini setup
# ==================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing in .env")

genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# ==================================================
# Chapter 3 JSON Schemas (two-step)
# ==================================================
class Criterion(TypedDict):
    name: str
    description: str


class CriteriaOnly(TypedDict):
    criteria: List[Criterion]


class DecisionOnly(TypedDict):
    decision: Literal["continue", "stop"]
    next_action: Literal["combo_analysis", "none"]
    target: Literal["Sex+Pclass", "none"]
    reason: str


# ==================================================
# Databricks helpers
# ==================================================
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


# ==================================================
# SQL builders
# ==================================================
def q_smoke():
    return "SELECT 1 AS ok"


def q_baseline():
    return f"""
    SELECT COUNT(*) AS n,
           AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    """


def q_group(col: str, min_n: int):
    return f"""
    SELECT {col} AS group_key,
           COUNT(*) AS n,
           AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    WHERE {col} IS NOT NULL
    GROUP BY {col}
    HAVING COUNT(*) >= {min_n}
    ORDER BY survival_rate DESC
    """


def q_age_group(min_n: int):
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
    HAVING COUNT(*) >= {min_n}
    ORDER BY survival_rate DESC
    """


def q_combo_sex_pclass(min_n: int):
    return f"""
    SELECT Sex, Pclass,
           COUNT(*) AS n,
           AVG(CAST(Survived AS DOUBLE)) AS survival_rate
    FROM {FULL_TABLE}
    WHERE Sex IS NOT NULL AND Pclass IS NOT NULL
    GROUP BY Sex, Pclass
    HAVING COUNT(*) >= {min_n}
    ORDER BY survival_rate DESC
    """


# ==================================================
# Robust JSON extraction
# ==================================================
def extract_json_from_text(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Empty LLM response text")

    t = text.strip()

    def _try_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            # allow single quotes / python-literal style as a last resort
            return ast.literal_eval(s)

    def _find_balanced_candidate(s: str):
        first_obj = s.find("{")
        first_arr = s.find("[")
        if first_obj == -1 and first_arr == -1:
            return None
        if first_arr == -1 or (first_obj != -1 and first_obj < first_arr):
            start = first_obj
        else:
            start = first_arr

        stack = []
        in_str = False
        str_char = ""
        escape = False
        last_complete = None

        for i in range(start, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue
            if in_str:
                if ch == "\\":
                    escape = True
                elif ch == str_char:
                    in_str = False
                continue

            if ch in ("\"", "'"):
                in_str = True
                str_char = ch
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    continue
                open_ch = stack.pop()
                if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                    # mismatched, abandon tracking
                    return None
                if not stack:
                    last_complete = i

        if last_complete is not None:
            return s[start:last_complete + 1]

        # Attempt a repair by closing unbalanced brackets/braces.
        candidate = s[start:]
        if in_str and str_char:
            while candidate.endswith("\\"):
                candidate = candidate[:-1]
            candidate += str_char
        for open_ch in reversed(stack):
            candidate += "}" if open_ch == "{" else "]"
        return candidate

    if t[0] in "{[" and t[-1] in "}]":
        try:
            return _try_load(t)
        except Exception:
            pass

    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return _try_load(m.group(1).strip())

    m2 = re.search(r"```\s*(\{.*?\}|\[.*?\])\s*```", t, flags=re.DOTALL)
    if m2:
        return _try_load(m2.group(1).strip())

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _try_load(t[start:end + 1].strip())
        except Exception:
            pass

    start = t.find("[")
    end = t.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return _try_load(t[start:end + 1].strip())
        except Exception:
            pass

    candidate = _find_balanced_candidate(t)
    if candidate:
        try:
            return _try_load(candidate.strip())
        except Exception:
            pass

    raise ValueError(f"No JSON object found in LLM response. Text: {t[:200]}...")


def _resp_text_and_reason(resp) -> Tuple[str, str]:
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


# ==================================================
# Gemini JSON call (schema + retry)
# ==================================================
def call_gemini_json(prompt: str, schema, *, max_tokens: int, temperature: float = 0.0) -> Tuple[str, str]:
    gen_cfg: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    resp = GEMINI_MODEL.generate_content(prompt, generation_config=gen_cfg)
    return _resp_text_and_reason(resp)


def try_parse_json(raw: str) -> dict:
    return extract_json_from_text(raw)


def validate_criteria_obj(obj: dict) -> bool:
    crit = obj.get("criteria")
    if not isinstance(crit, list) or not (1 <= len(crit) <= 6):
        return False
    for c in crit:
        if not isinstance(c, dict):
            return False
        name = c.get("name")
        desc = c.get("description")
        if not isinstance(name, str) or not isinstance(desc, str):
            return False
        if not name.strip() or not desc.strip():
            return False
        # RELAXED: allow long names but we'll truncate them in normalize_criteria
    return True


def validate_decision_obj(obj: dict, criteria: List[Dict[str, str]]) -> bool:
    if not isinstance(obj, dict):
        return False
    
    # Relaxed validation: case-insensitive
    decision = str(obj.get("decision", "")).lower().strip()
    if decision not in ("continue", "stop"):
        return False

    # REASON IS NOW OPTIONAL (auto-filled if missing) to prevent crashes
    return True


def normalize_criteria(criteria: Any) -> List[Criterion]:
    out: List[Criterion] = []
    if isinstance(criteria, list):
        for c in criteria[:6]:
            if isinstance(c, dict):
                name = c.get("name", "")
                desc = c.get("description", "")
                if isinstance(name, str) and isinstance(desc, str) and name.strip() and desc.strip():
                    # Normalize to short snake_case names to avoid overflow
                    name_norm = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
                    name_norm = re.sub(r"_+", "_", name_norm)[:20]
                    desc_clean = desc.strip()[:120]
                    if name_norm:
                        out.append({"name": name_norm, "description": desc_clean})
    if len(out) < 2:
        out = [
            {"name": "info_gain", "description": "Prefer actions that add new insight beyond current results."},
            {"name": "cost_control", "description": "Avoid extra computation unless benefit is clearly worth it."},
            {"name": "interaction_value", "description": "Run interaction check only if it adds explanatory value."},
        ]
    return out[:4]


def normalize_decision(d: Any) -> DecisionOnly:
    if not isinstance(d, dict):
        return {
            "decision": "stop",
            "next_action": "none",
            "target": "none",
            "reason": "invalid_decision_object",
        }

    decision = str(d.get("decision", "stop")).lower().strip()
    next_action = str(d.get("next_action", "none")).lower().strip()
    target = str(d.get("target", "none")).strip()
    reason = d.get("reason", "")

    if decision not in ("continue", "stop"):
        decision = "stop"
    if next_action not in ("combo_analysis", "none"):
        next_action = "none"
    
    if target != "Sex+Pclass":
        if "sex" in target.lower() and "pclass" in target.lower():
             target = "Sex+Pclass"
        else:
             target = "none"

    if not isinstance(reason, str) or not reason.strip():
        reason = "no_reason"

    return {
        "decision": decision, # type: ignore
        "next_action": next_action, # type: ignore
        "target": target, # type: ignore
        "reason": reason.strip(),
    }


def ensure_reason_mentions_criterion(reason: str, criteria: List[Dict[str, str]]) -> str:
    names = [c.get("name", "") for c in criteria if isinstance(c, dict)]
    names = [n for n in names if isinstance(n, str) and n.strip()]
    if not names:
        return reason
    if any(n in reason for n in names):
        return reason
    suffix = f" ({names[0]})"
    if reason.endswith("."):
        return reason + f" {names[0]}"
    return reason + suffix


def call_with_retry(
    prompt: str,
    schema,
    *,
    max_tokens: int,
    validator,
    validator_args: tuple,
    retry_hint: str,
    coerce=None,
    temperature: float = 0.0,
) -> Tuple[dict, str, str]:
    raw1, _ = call_gemini_json(prompt, schema, max_tokens=max_tokens, temperature=temperature)
    raw2 = ""

    try:
        obj = try_parse_json(raw1)
        if coerce:
            obj = coerce(obj)
        if validator(obj, *validator_args):
            return obj, raw1, raw2
        raise ValueError("validation_failed")
    except Exception:
        reprompt = f"""
Return ONLY valid JSON. No markdown. No explanation. No code fences.
{retry_hint}

Context:
{prompt}
"""
        raw2, _ = call_gemini_json(reprompt, schema, max_tokens=max_tokens, temperature=temperature)
        obj2 = try_parse_json(raw2)
        if coerce:
            obj2 = coerce(obj2)
        if not validator(obj2, *validator_args):
            raise ValueError(f"validation_failed_after_retry. Raw1: {raw1[:100]}... Raw2: {raw2[:100]}...")
        return obj2, raw1, raw2


def coerce_criteria_obj(obj: Any) -> dict:
    if isinstance(obj, list):
        items = []
        for it in obj:
            if isinstance(it, dict):
                name = it.get("name", "")
                desc = it.get("description", "auto")
                if isinstance(name, str) and name.strip():
                    items.append(
                        {
                            "name": name.strip()[:120],
                            "description": (desc if isinstance(desc, str) and desc.strip() else "auto")[:500],
                        }
                    )
            elif isinstance(it, str) and it.strip():
                items.append({"name": it.strip()[:120], "description": "auto"})
        return {"criteria": items}
    if isinstance(obj, dict):
        crit = obj.get("criteria")
        if isinstance(crit, dict):
            name = crit.get("name", "")
            desc = crit.get("description", "auto")
            obj["criteria"] = [
                {
                    "name": (name if isinstance(name, str) else str(name)).strip()[:120],
                    "description": (desc if isinstance(desc, str) and desc.strip() else "auto")[:500],
                }
            ]
        elif isinstance(crit, list):
            items = []
            for it in crit:
                if isinstance(it, dict):
                    name = it.get("name", "")
                    desc = it.get("description", "auto")
                    if isinstance(name, str) and name.strip():
                        items.append(
                            {
                                "name": name.strip()[:120],
                                "description": (desc if isinstance(desc, str) and desc.strip() else "auto")[:500],
                            }
                        )
                elif isinstance(it, str) and it.strip():
                    items.append({"name": it.strip()[:120], "description": "auto"})
            obj["criteria"] = items
        elif isinstance(crit, str):
            try:
                parsed = extract_json_from_text(crit)
                if isinstance(parsed, list):
                    items = []
                    for it in parsed:
                        if isinstance(it, dict):
                            name = it.get("name", "")
                            desc = it.get("description", "auto")
                            if isinstance(name, str) and name.strip():
                                items.append(
                                    {
                                        "name": name.strip()[:120],
                                        "description": (desc if isinstance(desc, str) and desc.strip() else "auto")[:500],
                                    }
                                )
                        elif isinstance(it, str) and it.strip():
                            items.append({"name": it.strip()[:120], "description": "auto"})
                    obj["criteria"] = items
                elif isinstance(parsed, dict) and "name" in parsed and "description" in parsed:
                    name = parsed.get("name", "")
                    desc = parsed.get("description", "auto")
                    obj["criteria"] = [
                        {
                            "name": (name if isinstance(name, str) else str(name)).strip()[:120],
                            "description": (desc if isinstance(desc, str) and desc.strip() else "auto")[:500],
                        }
                    ]
            except Exception:
                pass
        return obj
    return {"criteria": []}


def coerce_decision_obj(obj: Any) -> dict:
    if isinstance(obj, list) and obj:
        if isinstance(obj[0], dict):
            return obj[0]
    if isinstance(obj, dict):
        return obj
    return {}


# ==================================================
# Chapter 3 prompts (two-step)
# ==================================================
def build_criteria_prompt(goal: str, scores: Dict[str, float], executed: List[str]) -> str:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_txt = "\n".join([f"- {k}: {v:.3f}" for k, v in ranked[:4]])
    done_txt = ", ".join(executed)

    return f"""
Output 3 criteria as JSON only.

Format:
{{
  "criteria": [
    {{"name":"info_gain","description":"short text"}},
    {{"name":"interaction_value","description":"short text"}},
    {{"name":"cost_control","description":"short text"}}
  ]
}}

Each name: exactly info_gain, interaction_value, or cost_control
Each description: 5-8 words maximum

Data: {ranked_txt}
Done: {done_txt}
"""


def build_decision_prompt(
    goal: str,
    scores: Dict[str, float],
    executed: List[str],
    criteria: List[Criterion],
) -> str:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_txt = "\n".join([f"- {k}: spread={v:.3f}" for k, v in ranked[:4]])
    done_txt = ", ".join(executed)
    crit_txt = "\n".join([f"- {c['name']}: {c['description']}" for c in criteria])

    return f"""
Return ONLY JSON.

You are an analysis agent. Use ONLY the criteria below to decide the next step.

Goal:
{goal}

Criteria:
{crit_txt}

Evidence:
{ranked_txt}

Already executed:
{done_txt}

Choose ONE option:

Option A:
{{"decision":"continue","next_action":"combo_analysis","target":"Sex+Pclass","reason":"must mention at least one criterion name"}}

Option B:
{{"decision":"stop","next_action":"none","target":"none","reason":"must mention at least one criterion name"}}
"""


# ==================================================
# Agent logic
# ==================================================
def compute_spread(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    s = pd.to_numeric(df["survival_rate"], errors="coerce").dropna()
    return float(s.max() - s.min()) if not s.empty else 0.0


def rule_fallback(scores: Dict[str, float]) -> DecisionOnly:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_dim, top_spread = ranked[0] if ranked else ("none", 0.0)

    if top_spread >= FALLBACK_SPREAD_TRIGGER:
        return {
            "decision": "continue",
            "next_action": "combo_analysis",
            "target": "Sex+Pclass",
            "reason": f"fallback_rule uses info_gain: top_spread={top_spread:.3f} >= {FALLBACK_SPREAD_TRIGGER}",
        }
    return {
        "decision": "stop",
        "next_action": "none",
        "target": "none",
        "reason": f"fallback_rule uses cost_control: top_spread={top_spread:.3f} < {FALLBACK_SPREAD_TRIGGER}",
    }


def run_agent(dims: List[str], min_n: int, enable_hybrid: bool) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "goal": "Identify key factors affecting Titanic survival; decide whether interaction analysis is worth it.",
        "results": {},
        "scores": {},
        "actions": [],
        "sql_log": [],
        "criteria": [],
        "llm_decision": {},
        "llm_error": "",
        "raw_criteria_1": "",
        "raw_criteria_2": "",
        "raw_decision_1": "",
        "raw_decision_2": "",
        "used_hybrid_fallback": False,
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
        q = q_age_group(min_n) if d == "AgeGroup" else q_group(d, min_n)
        state["sql_log"].append(q)
        df = run_sql(q)
        state["results"][d] = df
        state["scores"][d] = compute_spread(df)

    executed = list(state["results"].keys())

    # -------- Step 1) Criteria generation --------
    c_prompt = build_criteria_prompt(state["goal"], state["scores"], executed)
    try:
        c_obj, raw_c1, raw_c2 = call_with_retry(
            c_prompt,
            CriteriaOnly,
            max_tokens=MAX_TOKENS_CRITERIA,
            validator=lambda o: validate_criteria_obj(o),
            validator_args=(),
            retry_hint='Output must match: {"criteria":[{"name":"info_gain","description":"..."},{"name":"interaction_value","description":"..."},{"name":"cost_control","description":"..."}]} (3 items).',
            coerce=coerce_criteria_obj,
            temperature=0.4,  # Higher temp to prevent repetition loops
        )
        state["raw_criteria_1"] = raw_c1
        state["raw_criteria_2"] = raw_c2
        state["criteria"] = normalize_criteria(c_obj.get("criteria"))
        state["actions"].append("LLM generated criteria (Step 1)")
    except Exception as e:
        state["llm_error"] = f"criteria_generation_failed: {type(e).__name__}: {e}"
        state["criteria"] = normalize_criteria([])
        state["actions"].append("LLM criteria generation failed; using default criteria")

    # -------- Step 2) Decision using criteria --------
    d_prompt = build_decision_prompt(state["goal"], state["scores"], executed, state["criteria"])
    try:
        d_obj, raw_d1, raw_d2 = call_with_retry(
            d_prompt,
            DecisionOnly,
            max_tokens=MAX_TOKENS_DECISION,  # ✅ 2024
            validator=lambda o, crit: validate_decision_obj(o, crit),
            validator_args=(state["criteria"],),
            retry_hint='Output must match JSON and include reason referencing one of: info_gain, interaction_value, cost_control.',
            coerce=coerce_decision_obj,
        )
        state["raw_decision_1"] = raw_d1
        state["raw_decision_2"] = raw_d2
        decision = normalize_decision(d_obj)
        decision["reason"] = ensure_reason_mentions_criterion(decision["reason"], state["criteria"])
        state["llm_decision"] = decision
        state["actions"].append("LLM decided next step using criteria (Step 2)")
    except Exception as e:
        state["llm_error"] = (state["llm_error"] + " | " if state["llm_error"] else "") + f"decision_failed: {type(e).__name__}: {e}"
        if enable_hybrid:
            decision = rule_fallback(state["scores"])
            state["llm_decision"] = decision
            state["used_hybrid_fallback"] = True
            state["actions"].append("LLM decision failed; hybrid fallback applied (rule-based)")
        else:
            state["llm_decision"] = {"decision": "stop", "next_action": "none", "target": "none", "reason": "llm_decision_failed"}
            state["actions"].append("LLM decision failed; stopped safely")

    # -------- Route action --------
    decision = state["llm_decision"]
    if decision.get("decision") == "continue" and decision.get("next_action") == "combo_analysis":
        if decision.get("target") == "Sex+Pclass":
            qc = q_combo_sex_pclass(min_n)
            state["sql_log"].append(qc)
            state["results"]["Sex+Pclass"] = run_sql(qc)
            state["actions"].append("Action: ran Sex+Pclass combo analysis")
        else:
            state["actions"].append(f"Requested combo_analysis but target={decision.get('target')} not supported yet")
    else:
        state["actions"].append("Action: stop")

    return state


# ==================================================
# Streamlit UI
# ==================================================
st.set_page_config(layout="wide")
st.title("🧠 Chapter 3 – Criteria → Decision (2-step) | Gemini 2.5 Flash + Databricks")

with st.sidebar:
    st.caption(f"Table: {FULL_TABLE}")
    dims = st.multiselect(
        "Dimensions",
        ["Sex", "Pclass", "Embarked", "AgeGroup"],
        default=["Sex", "Pclass", "Embarked", "AgeGroup"],
    )
    min_n = st.number_input("MIN_N (min group size)", min_value=1, max_value=2000, value=DEFAULT_MIN_N, step=1)
    enable_hybrid = st.checkbox("Enable Hybrid fallback (LLM fail → rule-based)", value=True)
    run = st.button("Run Agent", type="primary")

if run:
    with st.spinner("Agent running..."):
        state = run_agent(dims=dims, min_n=int(min_n), enable_hybrid=enable_hybrid)

    st.subheader("📊 Dimension Scores (spread)")
    score_df = pd.DataFrame(
        [{"dimension": k, "spread": v} for k, v in state["scores"].items()]
    ).sort_values("spread", ascending=False)
    st.dataframe(score_df, use_container_width=True)
    if not score_df.empty:
        st.bar_chart(score_df.set_index("dimension")["spread"])

    st.subheader("🧠 Generated Criteria (Chapter 3 핵심)")
    st.table(pd.DataFrame(state["criteria"]))

    st.subheader("🤖 Decision (based on criteria)")
    st.json(state["llm_decision"])
    if state.get("used_hybrid_fallback"):
        st.info("Hybrid fallback was applied (LLM output not usable).")

    st.subheader("🧭 Actions")
    for a in state["actions"]:
        st.write(f"- {a}")

    with st.expander("🧩 Gemini raw response (debug)"):
        st.markdown("**[Criteria step #1]**")
        st.text(state.get("raw_criteria_1", ""))
        if state.get("raw_criteria_2"):
            st.text("\n--- retry ---\n")
            st.text(state.get("raw_criteria_2", ""))

        st.markdown("**[Decision step #2]**")
        st.text(state.get("raw_decision_1", ""))
        if state.get("raw_decision_2"):
            st.text("\n--- retry ---\n")
            st.text(state.get("raw_decision_2", ""))

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
