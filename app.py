# ==========================================================
# BHB Study Finder — Dynamic Column Filters + Enrichr
# (robust Enrichr, recs, formatted EA, GMT term sizes, adjP-first,
#  drop old p columns, Gene set size → Overlap # → Overlap %, real reset)
# + Display truncation (~50 chars per cell) in AgGrid
# + Ultra-robust CSV reading (encoding + delimiter sniff + bad-line skip)
# ==========================================================
from io import BytesIO, StringIO
import os, re, sys, traceback, numpy as np, pandas as pd, streamlit as st
from pathlib import Path
import warnings, platform, json
import requests
import socket
from typing import Optional

# ---------- Page ----------
st.set_page_config(page_title="BHB Study Finder", page_icon="🔬", layout="wide")
st.title("🔬 BHB Study Finder")

st.markdown("""
This is a tool that lets you filter BHB studies according to several criteria (model, tissue/organ(elle), method of increasing BHB, etc.). 
Use the left sidebar to combine filters; results update live.

**About targets → enrichment:**  
After filtering, the app extracts the set of **BHB targets** from the filtered studies (column *BHB Target*), standardizes them to official symbols, and can send them to **Enrichr** for pathway/GO enrichment via the public REST API.
""")

# ---------- AgGrid import with diagnostics ----------
AGGRID_IMPORT_ERR = None
HAVE_AGGRID = False
ColumnsAutoSizeMode = None
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    try:
        from st_aggrid.shared import ColumnsAutoSizeMode
    except Exception:
        ColumnsAutoSizeMode = None  # older versions don't expose this
    HAVE_AGGRID = True
    print("[AgGrid] Import successful.")
except Exception as e:
    AGGRID_IMPORT_ERR = e
    HAVE_AGGRID = False
    AgGrid = GridOptionsBuilder = None
    print("[AgGrid] Import failed:", repr(e))
    print("[AgGrid] Traceback:\n", traceback.format_exc())

# ---------- Config ----------
DELIMS_PATTERN = r"[;,+/|]"  # for multi-value columns (incl. BHB Target)
MAX_MULTISELECT_OPTIONS = 200
BOOL_TRUE  = {"true","1","yes","y","t"}
BOOL_FALSE = {"false","0","no","n","f"}
APP_DIR = Path(__file__).resolve().parent

# ---------- Tips ----------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+","", s.lower())

TIPS = {
    _norm("BHB Filter"): "Filter via official gene names like **NLRP3**, **UCP1**, etc.",
    _norm("Model Raw"): "Study model as extracted from the abstract, e.g. **Wistar rat**, **healthy human adults**.",
    _norm("Model Global"): "Select **in vitro**, **animal**, or **human**.",
    _norm("Model Canonical"): "Categorized broad models such as **Ex vivo**, **Mouse**, **In silico**, etc.",
    _norm("Model Canonical"): "Categorized broad models such as **Ex vivo**, **Mouse**, **In silico**, etc.",
    _norm("Mechanism Raw"): "Studied mechanism as extracted from the abstract, e.g. **lipolysis**, **mitochondrial complex II activation**.",
    _norm("Mechanism Canonical"): "Broader categories such as **Mitochondrial Function & Bioenergetics** or **Metabolic Regulation**.",
    _norm("Mechanism Canonical"): "Broader categories such as **Mitochondrial Function & Bioenergetics** or **Metabolic Regulation**.",
}

# ---------- Helpers ----------
def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

def is_id_like(colname: str) -> bool:
    n = normalize(colname)
    return n in {"abstractid","pmcid","id","aid"} or n.endswith("id")

def is_pmid_col(colname: str) -> bool:
    return normalize(colname) == "pmid"

def sanitize_key(s: str) -> str:
    return "flt_" + re.sub(r"[^a-zA-Z0-9_]", "_", s)

def tip_for(colname: str) -> str | None:
    return TIPS.get(_norm(colname))

def tokenize_options(series: pd.Series) -> list:
    vals = set()
    for v in series.dropna().astype(str):
        for tok in re.split(DELIMS_PATTERN, v):
            tok = tok.strip()
            if tok:
                vals.add(tok)
    return sorted(vals)

def match_tokens(cell, selected_set):
    if not selected_set:
        return True
    if pd.isna(cell):
        return False
    toks = {t.strip() for t in re.split(DELIMS_PATTERN, str(cell)) if t.strip()}
    return bool(toks & selected_set)

def to_excel_bytes(df: pd.DataFrame):
    bio = BytesIO()
    df.to_excel(bio, index=False)
    return bio.getvalue()

def is_numeric_series(series: pd.Series, min_frac_numeric: float = 0.8) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    frac = s.notna().mean() if len(s) else 0.0
    return frac >= min_frac_numeric

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def guess_datetime_format(series: pd.Series, sample_size: int = 500) -> str | None:
    candidates = [
        "%Y-%m-%d","%Y/%m/%d","%d/%m/%Y","%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S","%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S","%Y-%m-%dT%H:%M:%S.%f",
    ]
    s = series.dropna().astype(str)
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=0)
    best_fmt, best_rate = None, 0.0
    for fmt in candidates:
        parsed = pd.to_datetime(s, errors="coerce", format=fmt, utc=False)
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_rate, best_fmt = rate, fmt
    return best_fmt if best_rate >= 0.8 else None

def safe_to_datetime(series: pd.Series, fmt: str | None) -> pd.Series:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*infer_datetime_format.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
        return pd.to_datetime(series, errors="coerce", format=fmt, utc=False)

def is_datetime_series(series: pd.Series, min_frac_dt: float = 0.8) -> bool:
    fmt = guess_datetime_format(series)
    s = safe_to_datetime(series, fmt)
    frac = s.notna().mean() if len(s) else 0.0
    return frac >= min_frac_dt

def is_booleanish_series(series: pd.Series, min_frac_bool: float = 0.9) -> bool:
    s = series.dropna().astype(str).str.strip().str.lower()
    ok = s.isin(BOOL_TRUE | BOOL_FALSE)
    frac = ok.mean() if len(s) else 0.0
    return frac >= min_frac_bool

def coerce_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map(lambda x: True if x in BOOL_TRUE else (False if x in BOOL_FALSE else np.nan))

def parse_id_equals_any(text: str) -> set:
    if not text:
        return set()
    parts = re.split(r"[;\s,]+", text)
    return {p.strip() for p in parts if p.strip()}

# ---------- Data discovery ----------
APP_DIR = Path(__file__).resolve().parent

def discover_repo_csv() -> Path | None:
    ds = None
    try:
        ds = st.secrets.get("DATASET_PATH", None)  # type: ignore[attr-defined]
    except Exception:
        ds = None

    if ds:
        p = (APP_DIR / str(ds)).resolve()
        if p.exists():
            return p

    ds_env = os.environ.get("DATASET_PATH")
    if ds_env:
        p = (APP_DIR / ds_env).resolve()
        if p.exists():
            return p

    for name in ("bhb_s.csv", "studies.csv", "dataset.csv"):
        p = (APP_DIR / name).resolve()
        if p.exists():
            return p

    candidates = list(APP_DIR.glob("*.csv"))
    data_dir = APP_DIR / "data"
    if not candidates and data_dir.exists():
        candidates = list(data_dir.glob("*.csv"))
    return candidates[0].resolve() if candidates else None

# ---------- Ultra-robust CSV read (encoding + delimiter sniff + bad-line skip) ----------

def read_csv_robust(path: Path) -> pd.DataFrame:
    """Try multiple encodings and parsers; sniff delimiter; skip bad lines; never crash."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    # Pass 1: C engine, standard comma
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    # Pass 2: Python engine with delimiter sniffing
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    # Pass 3: Python engine, sniff + skip malformed rows
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc, sep=None, engine="python", on_bad_lines='skip')
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    # Final fallback: read as text with replacement + strip NUL bytes
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        # If even read_text fails with utf-8, try cp1252 replacement
        data = Path(path).read_bytes()
        text = data.decode("cp1252", errors="replace")
    text = text.replace("\x00", "")
    return pd.read_csv(StringIO(text), low_memory=False, sep=None, engine="python", on_bad_lines='skip')

# ---------- HTTP debug (optional) ----------

def http_debug(resp):
    try:
        st.write(f"**Request:** {resp.request.method} {resp.url}")
    except Exception:
        pass
    st.write("**Status:**", resp.status_code, " • **Elapsed:**", getattr(resp, "elapsed", "n/a"))
    try:
        st.write("**Response headers (subset):**", {k: v for k, v in resp.headers.items() if k.lower() in {"content-type","server","date"}})
    except Exception:
        pass
    body_preview = (resp.text or "")[:1000]
    st.code(body_preview if body_preview else "(empty body)")

def _http_debug(resp):
    try:
        http_debug(resp)
    except Exception:
        pass

# ---------- Robust Enrichr session ----------

def secrets_bool(key: str, default: bool = False) -> bool:
    try:
        return bool(st.secrets.get(key, default))  # type: ignore[attr-defined]
    except Exception:
        v = os.environ.get(key)
        if v is None:
            return default
        return v.strip().lower() in {"1", "true", "yes", "y"}

ENRICHR_BASES = [
    "https://maayanlab.cloud/Enrichr",
    "http://maayanlab.cloud/Enrichr",
    "https://amp.pharm.mssm.edu/Enrichr",  # legacy hostname
]

def make_session(force_ipv4: bool = False) -> requests.Session:
    if force_ipv4:
        try:
            import urllib3.util.connection as urllib3_cn
            def allowed_gai_family():
                return socket.AF_INET
            urllib3_cn.allowed_gai_family = allowed_gai_family
        except Exception:
            pass
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    s.headers.update({"User-Agent": "BHBStudyFinder/0.1"})
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def _first_ok_base(session: requests.Session, debug: bool = False) -> Optional[str]:
    for base in ENRICHR_BASES:
        try:
            r = session.get(f"{base}/datasetStatistics", timeout=15)
            if debug: _http_debug(r)
            if r.ok:
                return base
        except requests.RequestException:
            pass
    return None

def _endpoints(session: requests.Session, debug: bool = False):
    base = _first_ok_base(session, debug=debug)
    if not base:
        raise RuntimeError("Could not reach any Enrichr endpoint (network/TLS/DNS issue).")
    return (f"{base}/addList", f"{base}/enrich", f"{base}/geneSetLibrary", f"{base}/datasetStatistics")

# ---------- Target helpers ----------
TARGET_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,19}$")
BAD_TOKENS = {"NA", "N/A", "NONE", "NULL", "-"}

def find_bhb_target_col(columns: list[str]) -> str | None:
    for c in columns:
        if normalize(c) == "bhbtarget":
            return c
    for c in columns:
        if "target" in normalize(c):
            return c
    return None

def extract_targets_from_df(df: pd.DataFrame) -> list[str]:
    col = find_bhb_target_col(list(df.columns))
    if not col:
        return []
    genes = []
    for cell in df[col].dropna().astype(str):
        parts = [t.strip() for t in re.split(DELIMS_PATTERN, cell) if t.strip()]
        for tok in parts:
            tok_up = tok.upper()
            if " " in tok_up: continue
            if tok_up in BAD_TOKENS: continue
            if TARGET_TOKEN_RE.match(tok_up):
                genes.append(tok_up)
    seen, uniq = set(), []
    for g in genes:
        if g not in seen:
            uniq.append(g); seen.add(g)
    return uniq

# ---------- Enrichr REST ----------
DEFAULT_LIBRARIES = [
    "GO_Biological_Process_2023",
    "GO_Cellular_Component_2023",
    "GO_Molecular_Function_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "WikiPathways_2023_Human",
    "MSigDB_Hallmark_2020",
    "TRRUST_Transcription_Factors_2019",
]

@st.cache_data(show_spinner=False)
def enrichr_add_list(genes: list[str], description: str = "BHB Study Finder selection", debug: bool = False) -> dict:
    s = make_session(force_ipv4=secrets_bool("FORCE_IPV4", False))
    ENRICHR_ADD, *_ = _endpoints(s, debug=debug)
    payload = {"list": "\n".join(genes), "description": description}
    try:
        r = s.post(ENRICHR_ADD, data=payload, timeout=30)
        if debug: _http_debug(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        files = {"list": (None, "\n".join(genes)), "description": (None, description)}
        r2 = s.post(ENRICHR_ADD, files=files, timeout=30)
        if debug: _http_debug(r2)
        r2.raise_for_status()
        return r2.json()

@st.cache_data(show_spinner=False)
def enrichr_libraries(debug: bool = False) -> list[str]:
    s = make_session(force_ipv4=secrets_bool("FORCE_IPV4", False))
    *_, ENRICHR_STATS = _endpoints(s, debug=debug)
    r = s.get(ENRICHR_STATS, timeout=30); 
    if debug: _http_debug(r)
    r.raise_for_status()
    data = r.json()
    items = data.get("statistics", data) if isinstance(data, dict) else data
    libs = []
    for x in items:
        if isinstance(x, dict):
            name = x.get("libraryName") or x.get("library")
            if name: libs.append(name)
    return sorted(set(libs))

def _detect_overlap_ratio(rows: list[list]) -> list[Optional[str]]:
    out = []
    pat = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
    for r in rows:
        found = None
        for el in r:
            if isinstance(el, str) and pat.match(el):
                found = el.strip().replace(" ", "")
                break
        out.append(found)
    return out

def _coerce_rows_to_df(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Term","P-value","Adjusted P-value","Z-score","Combined Score","Overlap Genes"])
    overlap_ratio = _detect_overlap_ratio(rows)
    first = rows[0]
    if isinstance(first[0], (int, float)) and not isinstance(first[1], (int, float)):
        cols = ["Rank","Term","P-value","Z-score","Combined Score","Overlap Genes","Adjusted P-value","Old P-value","Old Adjusted P-value","Odds Ratio","Extra"]
        df = pd.DataFrame(rows, columns=cols[:len(first)])
    else:
        cols = ["Term","P-value","Z-score","Combined Score","Overlap Genes","Adjusted P-value","Old P-value","Old Adjusted P-value","Odds Ratio","Extra"]
        df = pd.DataFrame(rows, columns=cols[:len(first)])
        if "Rank" not in df.columns:
            df.insert(0, "Rank", range(1, len(df)+1))
    if "Overlap Genes" in df.columns:
        df["Overlap Genes"] = df["Overlap Genes"].astype(str).str.replace(r"[|]", ", ", regex=True)
    if any(overlap_ratio):
        df["Overlap"] = overlap_ratio

    # Drop legacy columns
    df = df.drop(columns=["Old P-value", "Old Adjusted P-value"], errors="ignore")

    sort_col = "Adjusted P-value" if "Adjusted P-value" in df.columns else "P-value"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df[sort_col] = pd.to_numeric(df[sort_col], errors="coerce")
    df = df.sort_values(by=sort_col, ascending=True, kind="mergesort").reset_index(drop=True)
    return df

def enrichr_enrich(user_list_id: int, library: str, debug: bool = False) -> pd.DataFrame:
    s = make_session(force_ipv4=secrets_bool("FORCE_IPV4", False))
    _, ENRICHR_ENR, *_ = _endpoints(s, debug=debug)
    params = {"userListId": user_list_id, "backgroundType": library}
    r = s.get(ENRICHR_ENR, params=params, timeout=60)
    if debug: _http_debug(r)
    r.raise_for_status()
    data = r.json()
    rows = next(iter(data.values())) if isinstance(data, dict) else data
    return _coerce_rows_to_df(rows)

# ----- P-value formatting + term-size lookup (GMT) -----

def _format_p(value) -> str:
    """4 d.p. normally; <1e-4 in compact scientific with real value."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v): return ""
    if v < 1e-4:
        s = f"{v:.2E}"
        s = re.sub(r"(\.\d*?[1-9])0+E", r"\1E", s)
        s = re.sub(r"\.0+E", "E", s)
        return s
    return f"{v:.4f}"

def _count_overlap_genes(s: str) -> int:
    if s is None or (isinstance(s, float) and np.isnan(s)): return 0
    return sum(1 for g in str(s).split(",") if g.strip())

def _normalize_term_key(term: str) -> str:
    t = str(term)
    t = re.sub(r"\s*[-–]\s*Homo sapiens$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\((?:Homo sapiens|GO:\d+|R-[A-Z]+-\d+|WP\d+)\)\s*$", "", t, flags=re.IGNORECASE)
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t

@st.cache_data(show_spinner=False)
def enrichr_termsizes(library: str, debug: bool = False) -> dict:
    """
    Return {term -> gene set size} using GMT text for consistency across libraries.
    Also stores a normalized-key map under '_norm_map_'.
    """
    s = make_session(force_ipv4=secrets_bool("FORCE_IPV4", False))
    *_, ENRICHR_GMT, _ = _endpoints(s, debug=debug)
    params = {"mode": "text", "libraryName": library}
    r = s.get(ENRICHR_GMT, params=params, timeout=60)
    if debug: _http_debug(r)
    r.raise_for_status()
    text = r.text

    sizes: dict = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            term = parts[0]
            sizes[term] = len(parts) - 2
    sizes["_norm_map_"] = { _normalize_term_key(t): sz for t, sz in sizes.items() if t != "_norm_map_" }
    return sizes

def _augment_enrichr_table(df: pd.DataFrame, library: str):
    """
    Returns (df_numeric, df_display):
      - df_numeric: numeric table for downloads (includes Gene set size, Overlap #, Overlap %)
      - df_display: formatted table (p-values, Overlap %) and nice column order
    Gene set size (K):
      1) Parse 'Overlap' ratio 'k/K' if present
      2) Else map via GMT sizes with normalized keys
    """
    df_num = df.copy()

    # Overlap counts (k)
    if "Overlap Genes" in df_num.columns:
        k = df_num["Overlap Genes"].apply(_count_overlap_genes)
    else:
        k = pd.Series([np.nan] * len(df_num), index=df_num.index)

    # From Overlap ratio if provided (k/K)
    K_series = pd.Series([np.nan] * len(df_num), index=df_num.index, dtype="float")
    if "Overlap" in df_num.columns:
        parts = df_num["Overlap"].astype(str).str.extract(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            num = pd.to_numeric(parts[0], errors="coerce")
            den = pd.to_numeric(parts[1], errors="coerce")
        k = k.fillna(num)
        K_series = den

    # Fallback via GMT
    if K_series.isna().all():
        try:
            sizes = enrichr_termsizes(library, debug=False)
        except Exception:
            sizes = {}
        if sizes:
            norm_map = sizes.get("_norm_map_", {})
            if not norm_map:
                norm_map = { _normalize_term_key(t): sz for t, sz in sizes.items() if isinstance(t, str) }
            K_series = df_num["Term"].map(lambda t: norm_map.get(_normalize_term_key(str(t)), np.nan))

    # Compose numeric outputs
    df_num["Gene set size"] = pd.to_numeric(K_series, errors="coerce").round(0).astype("Int64")
    df_num["Overlap #"]     = pd.to_numeric(k, errors="coerce").round(0).astype("Int64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denom = df_num["Gene set size"].astype(float)
        nume  = df_num["Overlap #"].astype(float)
        df_num["Overlap %"] = np.where(denom.notna() & (denom > 0), (nume / denom) * 100.0, np.nan)

    # Pretty copy for display
    df_disp = df_num.copy()
    for col in ("P-value", "Adjusted P-value"):
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(_format_p)
    if "Overlap %" in df_disp.columns:
        df_disp["Overlap %"] = df_disp["Overlap %"].apply(lambda v: "" if pd.isna(v) else f"{v:.1f}")

    # Reorder columns — adjP first; Gene set size → Overlap # → Overlap %
    desired = [
        "Rank", "Term",
        "Adjusted P-value", "P-value",
        "Z-score", "Combined Score",
        "Gene set size", "Overlap #", "Overlap %",
        "Overlap Genes"
    ]
    def _reorder(df_):
        first = [c for c in desired if c in df_.columns]
        rest  = [c for c in df_.columns if c not in first]
        return df_[first + rest]
    df_num  = _reorder(df_num)
    df_disp = _reorder(df_disp)

    return df_num, df_disp

# ---------- Data source ----------
csv_path = None
try:
    csv_path = discover_repo_csv()
    if not csv_path:
        st.error("No dataset found. Put a CSV in the repo root (e.g., `bhb_s.csv`) or set `DATASET_PATH` in Secrets.")
        st.stop()
except Exception as e:
    st.error(f"Failed to locate dataset: {e}")
    st.stop()

# >>> ultra-robust CSV read here <<<
df = read_csv_robust(csv_path)

st.caption(f"Loaded dataset: `{csv_path.name}` • {len(df):,} rows, {df.shape[1]} columns")

# ---------- Reset logic ----------

def clear_all_filters():
    # Flag a one-shot reset and clear individual filter widget states
    st.session_state["__do_reset__"] = True
    for k in list(st.session_state.keys()):
        if k.startswith("flt_"):
            st.session_state.pop(k, None)
    # <- no st.rerun() here


# ---------- Sidebar: dynamic filters (with true reset-to-“Any”) ----------
filters_meta = []
with st.sidebar:
    st.header("🔎 Column Filters")
    st.caption("Filters apply cumulatively.")
    st.button("🔁 Reset all filters", on_click=clear_all_filters)

    # Was a reset requested this run?
    RESET_NOW = st.session_state.pop("__do_reset__", False)

    for col in df.columns:
        if is_pmid_col(col):
            continue
        series = df[col]
        keybase = sanitize_key(col)
        st.markdown(f"**{col}**")
        hint = tip_for(col)
        if hint:
            st.caption(hint)

        # ID-like columns → equals-any text box
        if is_id_like(col):
            key = keybase + "_idany"
            default = ""
            if RESET_NOW:
                st.session_state[key] = default
            txt = st.text_input("Equals any of …",
                                value=st.session_state.get(key, default),
                                key=key)
            filters_meta.append({"col": col, "type": "id_any", "value": txt})
            st.divider()
            continue

        # Type detection
        try_numeric = is_numeric_series(series)
        try_dt = False if try_numeric else is_datetime_series(series)
        if try_dt and not safe_to_datetime(series, guess_datetime_format(series)).notna().any():
            try_dt = False
        try_bool = False if (try_numeric or try_dt) else is_booleanish_series(series)

        if try_numeric:
            s_num = coerce_numeric(series)
            if s_num.notna().any():
                vmin = float(np.nanmin(s_num)); vmax = float(np.nanmax(s_num))
            else:
                vmin = 0.0; vmax = 0.0
            rng_key  = keybase + "_range"
            excl_key = keybase + "_exclna"
            default_rng  = (float(vmin), float(vmax))
            default_excl = False
            if RESET_NOW:
                st.session_state[rng_key]  = default_rng
                st.session_state[excl_key] = default_excl
            rng = st.slider("Range",
                            min_value=float(vmin), max_value=float(vmax),
                            value=st.session_state.get(rng_key, default_rng),
                            key=rng_key)
            excl_na = st.checkbox("Exclude missing",
                                  value=st.session_state.get(excl_key, default_excl),
                                  key=excl_key)
            filters_meta.append({"col": col, "type": "range", "value": rng, "excl_na": excl_na})

        elif try_dt:
            dt_fmt = guess_datetime_format(series)
            s_dt = safe_to_datetime(series, dt_fmt)
            dmin = s_dt.min().date() if s_dt.notna().any() else None
            dmax = s_dt.max().date() if s_dt.notna().any() else None
            if dmin and dmax:
                dr_key      = keybase + "_daterange"
                excl_dt_key = keybase + "_exclna_dt"
                default_dr       = (dmin, dmax)
                default_excl_dt  = False
                if RESET_NOW:
                    st.session_state[dr_key]      = default_dr
                    st.session_state[excl_dt_key] = default_excl_dt
                date_range = st.date_input("Date range",
                                           st.session_state.get(dr_key, default_dr),
                                           key=dr_key)
                excl_na = st.checkbox("Exclude missing",
                                      value=st.session_state.get(excl_dt_key, default_excl_dt),
                                      key=excl_dt_key)
                filters_meta.append({"col": col, "type": "date_range", "value": date_range,
                                     "excl_na": excl_na, "dt_format": dt_fmt})
            else:
                st.caption("_No valid dates detected_")

        elif try_bool:
            bool_key = keybase + "_bool"
            default_bool = "Any"
            if RESET_NOW:
                st.session_state[bool_key] = default_bool
            options = ["Any", "True", "False"]
            choice = st.selectbox("Value", options,
                                  index=options.index(st.session_state.get(bool_key, default_bool)),
                                  key=bool_key)
            filters_meta.append({"col": col, "type": "bool", "value": choice})

        else:
            tokens = tokenize_options(series.astype(str))
            if 0 < len(tokens) <= MAX_MULTISELECT_OPTIONS:
                multi_key = keybase + "_multi"
                default_multi = ["Any"]
                if RESET_NOW:
                    st.session_state[multi_key] = default_multi
                sel = st.multiselect("Select",
                                     ["Any"] + tokens,
                                     default=st.session_state.get(multi_key, default_multi),
                                     key=multi_key)
                filters_meta.append({"col": col, "type": "multi", "value": sel})
            else:
                txt_key = keybase + "_contains"
                default_txt = ""
                if RESET_NOW:
                    st.session_state[txt_key] = default_txt
                query = st.text_input("Contains any of …",
                                      value=st.session_state.get(txt_key, default_txt),
                                      key=txt_key)
                filters_meta.append({"col": col, "type": "contains_any", "value": query})
        st.divider()

# ---------- Apply filters ----------
mask = pd.Series([True] * len(df))
for f in filters_meta:
    col = f["col"]; typ = f["type"]; val = f["value"]
    if typ == "id_any":
        ids = parse_id_equals_any(val)
        if ids:
            mask &= df[col].astype(str).isin(ids)
    elif typ == "range":
        lo, hi = val
        s_num = coerce_numeric(df[col])
        cond = s_num.between(lo, hi)
        if not f.get("excl_na", False): cond = cond | s_num.isna()
        mask &= cond
    elif typ == "date_range":
        fmt = f.get("dt_format")
        s_dt = safe_to_datetime(df[col], fmt)
        if isinstance(val, tuple) and len(val) == 2:
            lo, hi = pd.to_datetime(val[0]), pd.to_datetime(val[1])
            cond = s_dt.between(lo, hi)
            if not f.get("excl_na", False): cond = cond | s_dt.isna()
            mask &= cond
    elif typ == "bool":
        if val in ("True", "False"):
            s_b = coerce_bool(df[col]); want = (val == "True")
            mask &= (s_b == want)
    elif typ == "multi":
        sel = [s for s in val if s != "Any"]
        if sel:
            sel_set = set(sel)
            mask &= df[col].apply(lambda v: match_tokens(v, sel_set))
    elif typ == "contains_any":
        query = str(val).strip()
        if query:
            needles = [q.strip() for q in query.split(";") if q.strip()]
            patt = "|".join(re.escape(q) for q in needles)
            mask &= df[col].astype(str).str.contains(patt, case=False, na=False, regex=True)

result = df.loc[mask].copy()

# ---------- Results + downloads ----------
st.subheader(f"📑 {len(result)} row{'s' if len(result)!=1 else ''} match your filters")

PAGE_SIZE = 20
GRID_HEIGHT = 600
TRUNC = 50  # ~50 characters preview per cell

if HAVE_AGGRID:
    gob = GridOptionsBuilder.from_dataframe(result)
    try:
        gob.configure_pagination(paginationAutoPageSize=False, paginationPageSize=PAGE_SIZE)
    except TypeError:
        gob.configure_pagination(paginationPageSize=PAGE_SIZE)
    gob.configure_default_column(filter=True, sortable=True, resizable=True)
    gob.configure_grid_options(domLayout="normal")

    # --- truncate every column to ~50 chars in the grid ---
    for c in result.columns:
        gob.configure_column(
            c,
            tooltipField=c,  # show full value on hover
            valueFormatter=f"""
                function(params) {{
                  if (params.value == null) return '';
                  var s = params.value.toString();
                  return s.length > {TRUNC} ? s.slice(0, {TRUNC}) + ' …' : s;
                }}
            """,
            cellStyle={"white-space": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
        )

    grid_opts = gob.build()
    AgGrid(
        result,
        gridOptions=grid_opts,
        height=GRID_HEIGHT,
        theme="alpine",
        fit_columns_on_grid_load=False,
        columns_auto_size_mode=None,  # disable auto-size-to-content (keeps truncation effective)
    )
else:
    st.info("Interactive grid unavailable (streamlit-aggrid not installed). Showing a simple table instead.")
    # Fallback: show a shortened copy so large text doesn't overwhelm
    def _short(v, n=TRUNC):
        s = "" if pd.isna(v) else str(v)
        return s if len(s) <= n else s[:n] + " …"
    short = result.applymap(_short)
    st.dataframe(short, use_container_width=True, height=GRID_HEIGHT)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("💾 Excel of filtered rows", to_excel_bytes(result), "filtered_rows.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with col_dl2:
    st.download_button("🗒️ CSV of filtered rows", result.to_csv(index=False), "filtered_rows.csv", mime="text/csv")

# ---------- Enrichment Analysis ----------
st.markdown("---")
st.subheader("🧬 Enrichment Analysis")

# Load ALL libraries up-front for multiselect options
libs_all, libs_err = [], None
try:
    libs_all = enrichr_libraries(debug=False)
except Exception as e:
    libs_err = e

# Quick recs
st.markdown("""
**Recommended libraries (quick guide)**  
Use these to capture BHB’s main biology (bioenergetics, metabolism, inflammation, regulation):
- **MSigDB_Hallmark_2020** — compact, non-redundant high-level pathways for a clean overview.
- **GO_Biological_Process_2023** — process-level signal (e.g., autophagy, fatty-acid metabolism).
- **GO_Cellular_Component_2023** — subcellular context (mitochondrion, peroxisome, chromatin).
- **GO_Molecular_Function_2023** — activities (dehydrogenase, transporter, oxidoreductase).
- **Reactome_2022** — curated signaling & metabolic routes (PDH/TCA, immune signaling).
- **KEGG_2021_Human** — metabolism maps (ketone bodies, TCA, insulin, AMPK) & disease links.
- **WikiPathways_2023_Human** — community pathways; often niche metabolic/mitochondrial maps.
- **TRRUST_Transcription_Factors_2019** — upstream TFs (e.g., HIF1A, NFKB1) driving your targets.
""")

# List all libraries
lib_list_container = st.container()
if lib_list_container.button("📚 List available Enrichr libraries"):
    try:
        if not libs_all and libs_err:
            raise libs_err
        st.success(f"Fetched {len(libs_all)} libraries from Enrichr")
        st.dataframe(pd.DataFrame({"library": libs_all}), use_container_width=True, height=400)
    except Exception as e:
        st.exception(e)

targets = extract_targets_from_df(result)
n_targets = len(targets)

if n_targets == 0:
    st.info("No targets found in the filtered rows. Make sure your dataset has a column named **BHB Target** (or any column containing 'target').")
else:
    st.markdown(f"**Filtered target set:** {n_targets} unique symbols")
    preview = ", ".join(targets[:50]) + (" …" if n_targets > 50 else "")
    st.code(preview or "(empty)")

    with st.expander("Run enrichment (Enrichr)", expanded=True):
        st.caption("Choose Enrichr libraries (type to search).")

        RECOMMENDED_LIBS = [
            "MSigDB_Hallmark_2020",
            "GO_Biological_Process_2023",
            "GO_Cellular_Component_2023",
            "GO_Molecular_Function_2023",
            "Reactome_2022",
            "KEGG_2021_Human",
            "WikiPathways_2023_Human",
            "TRRUST_Transcription_Factors_2019",
        ]

        if libs_all:
            defaults = [lib for lib in RECOMMENDED_LIBS if lib in libs_all] or libs_all[:3]
            options = libs_all
        else:
            defaults = RECOMMENDED_LIBS[:3]
            options = DEFAULT_LIBRARIES

        libs = st.multiselect(
            "Gene set libraries (Enrichr ‘backgroundType’ names)",
            options=options,
            default=defaults
        )
        topn = st.slider("Show top N terms per library", 5, 50, 20, step=5)
        run_btn = st.button("🚀 Run Enrichment")

        if run_btn:
            user_list_id = None
            try:
                with st.spinner("Uploading gene list to Enrichr…"):
                    add_res = enrichr_add_list(targets, description="BHB Study Finder selection")
                    user_list_id = add_res.get("userListId")
                    if not user_list_id:
                        st.error(f"Unexpected response from Enrichr:\n{json.dumps(add_res, indent=2)}")
                        st.stop()
            except requests.HTTPError as he:
                st.error(f"Upload to Enrichr failed (HTTP): {he}")
                st.stop()
            except Exception as e:
                st.error(f"Upload to Enrichr failed: {e}")
                st.stop()

            if not libs:
                st.warning("Please select at least one library.")
            else:
                tabs = st.tabs(libs)
                for lib, tab in zip(libs, tabs):
                    with tab:
                        try:
                            with st.spinner(f"Enriching against **{lib}**…"):
                                df_lib = enrichr_enrich(user_list_id, lib, debug=False)
                            if df_lib.empty:
                                st.info("No significant terms returned.")
                            else:
                                df_num, df_disp = _augment_enrichr_table(df_lib, lib)
                                st.dataframe(df_disp.head(topn), use_container_width=True)

                                if df_num["Gene set size"].isna().all():
                                    st.caption("ℹ️ Gene set sizes not available/matchable for this library; Overlap % left blank.")

                                c1, c2 = st.columns(2)
                                with c1:
                                    st.download_button(
                                        f"⬇️ CSV — {lib}",
                                        df_num.to_csv(index=False),
                                        file_name=f"enrichr_{lib}.csv",
                                        mime="text/csv"
                                    )
                                with c2:
                                    st.download_button(
                                        f"⬇️ Excel — {lib}",
                                        to_excel_bytes(df_num),
                                        file_name=f"enrichr_{lib}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                        except requests.HTTPError as he:
                            st.error(f"Enrichr returned an HTTP error for `{lib}`: {he}")
                        except Exception as e:
                            st.error(f"Failed to fetch enrichment for `{lib}`: {e}")

# ---------- Debug panel ----------
with st.sidebar.expander("🪲 Debug", expanded=False):
    st.write("**Python**:", sys.version.split()[0], platform.platform())
    try:
        import importlib.metadata as ilm
        ag_ver = ilm.version("streamlit-aggrid")
        st.write("**streamlit-aggrid**:", ag_ver)
        print("[AgGrid] Detected streamlit-aggrid version:", ag_ver)
    except Exception as e:
        st.write("**streamlit-aggrid**: not installed or not detected:", repr(e))
        print("[AgGrid] streamlit-aggrid not detected:", repr(e))
    st.write("**Streamlit**:", st.__version__)
    st.write("**HAVE_AGGRID**:", HAVE_AGGRID)
    if AGGRID_IMPORT_ERR:
        st.write("**AgGrid import error**:", repr(AGGRID_IMPORT_ERR))
        st.code(traceback.format_exc(), language="text")

# Optional network self-test
with st.sidebar.expander("🌐 Network self-test", expanded=False):
    st.caption("If connections fail in production, consider setting secret/env `FORCE_IPV4=true`.")
    if st.button("Test Enrichr reachability"):
        try:
            s = make_session(force_ipv4=secrets_bool("FORCE_IPV4", False))
            base = _first_ok_base(s, debug=False)
            if base:
                st.success(f"Reachable: {base}")
            else:
                st.error("No Enrichr base reachable (network/TLS/DNS issue).")
        except Exception as e:
            st.exception(e)
