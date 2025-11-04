#!/usr/bin/env python3
# ================== PACKAGES ==================
import os, json, re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from datetime import timedelta
import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import random
import json

# ================== ENV / CREDS ==================
def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

SA_JSON_RAW      = _req("GOOGLE_SERVICE_ACCOUNT_JSON")
DRIVE_FOLDER_ID  = _req("DRIVE_FOLDER_ID")
SCOPES = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/drive.file"]
CREDS  = Credentials.from_service_account_info(json.loads(SA_JSON_RAW), scopes=SCOPES)

def drive_client():
    return build("drive", "v3", credentials=CREDS)

def drive_create_subfolder(drive, parent_id: str, name: str) -> str:
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    return drive.files().create(body=meta, fields="id").execute()["id"]

def drive_upload_csv(drive, folder_id: str, path: Path) -> str:
    meta  = {"name": path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(path), mimetype="text/csv", resumable=True)
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]

# ================== CONFIG YOU EDIT ==================
# 1) Row-level filters: all conditions must pass.
#    Supported ops: eq, ne, in, nin, contains, regex, notnull, null (case-insensitive on strings)
FILTERS: Dict[str, Dict[str, Any]] = {
    # examples — adjust to your columns
    # "Grad Year": {"eq": "2027"},
    # "ACS Rank":  {"in": ["High Priority", "Priority", "Potential/WatchMore"]},
}

# 2) Ranking → which rows are “top” (put best first). Edit to your needs.
#    If the column is categorical, provide desired order in ORDER_MAPS.

TOP_N = 50  # keep this many

DISTANCE_ORDER = ["far", "never", "recent"]  # priority order

def add_contact_distance(df: pd.DataFrame, last_contact_col: str = "Last Contact") -> pd.DataFrame:
    """
    Build contact_distance from Last Contact:
      - NA/empty => 'never'
      - > 365 days ago => 'far'
      - within last 365 days => 'recent'
    """
    df = df.copy()
    # parse dates; tolerate blanks & weird formats
    lc = pd.to_datetime(df[last_contact_col].replace({"": None}), errors="coerce", utc=True)

    today = pd.Timestamp.utcnow().normalize()
    one_year_ago = today - pd.Timedelta(days=365)

    # classify
    cond_never = lc.isna()
    cond_recent = lc >= one_year_ago
    # anything else is "far"

    dist = pd.Series("far", index=df.index)
    dist[cond_never] = "never"
    dist[cond_recent & ~cond_never] = "recent"

    df["contact_distance"] = dist
    return df

def sort_by_contact_distance(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {k: i for i, k in enumerate(DISTANCE_ORDER)}  # far=0, never=1, recent=2
    key = df["contact_distance"].map(order_map).fillna(9999)
    return df.assign(_dist_rank=key).sort_values("_dist_rank", kind="stable").drop(columns="_dist_rank")
# ================== HELPERS ==================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _discover_run_id() -> str:
    """Find RUN_ID from env, a last_run_id.txt, or the newest manifest in any supported layout."""
    rid = os.getenv("RUN_ID", "").strip()
    if rid:
        return rid

    # 1) last_run_id.txt in output/ or any run-*/ folder
    try:
        return Path(_find_last_run_id_path()).read_text().strip()
    except FileNotFoundError:
        pass

    # 2) newest manifest in any of these bases
    patterns = [
        "output/run-*/raw/manifest_*.json",
        "run-*/raw/manifest_*.json",
        "output/raw/manifest_*.json",
        "raw/manifest_*.json",
    ]
    cands = []
    for pat in patterns:
        cands.extend(Path(".").glob(pat))
    if cands:
        latest = max(cands, key=lambda p: p.stat().st_mtime)
        try:
            m = json.loads(latest.read_text())
            return m.get("run_id") or latest.stem.replace("manifest_", "")
        except Exception:
            return latest.stem.replace("manifest_", "")

    raise RuntimeError(
        "Cannot determine RUN_ID. Set RUN_ID env var, provide output/last_run_id.txt, "
        "or ensure a manifest_* exists under output/run-*/raw or run-*/raw."
    )
def _find_raw_csv(run_id: str) -> Path:
    """
    Find the raw CSV for a given run_id across multiple supported layouts.

    Supported directories (first match wins):
      1) output/run-<run_id>/raw/
      2) output/raw/
      3) run-<run_id>/raw/
      4) raw/
    """
    # Search bases in priority order
    bases = [
        Path(f"output/run-{run_id}/raw"),
        Path("output/raw"),
        Path(f"run-{run_id}/raw"),
        Path("raw"),
    ]

    # Try manifest, then canonical name, then any *run_id* csv, then newest csv in each base
    for base in bases:
        if not base.exists():
            continue

        # 1) manifest
        mani = base / f"manifest_{run_id}.json"
        if mani.exists():
            try:
                saved = json.loads(mani.read_text()).get("saved_csv", "")
                p = Path(saved)
                if p.is_file():
                    return p
                # If manifest only contains filename, assume it's relative to base
                if saved and (base / saved).is_file():
                    return base / saved
            except Exception:
                pass

        # 2) canonical filename
        p = base / f"export_{run_id}.csv"
        if p.is_file():
            return p

        # 3) any csv containing the run_id
        cands = sorted(base.glob(f"*{run_id}*.csv"))
        if cands:
            return cands[0]

        # 4) newest CSV in this base
        any_csvs = sorted(base.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if any_csvs:
            print(f"[warn] No exact match for run_id={run_id} in {base}; using newest: {any_csvs[0].name}")
            return any_csvs[0]

    raise FileNotFoundError(
        f"No raw CSV found for run_id={run_id} in any of: " + ", ".join(str(b) for b in bases)
    )

def _find_last_run_id_path() -> Path:
    candidates = [Path("output/last_run_id.txt")] + list(Path(".").glob("run-*/last_run_id.txt"))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("last_run_id.txt not found in output/ or run-*/")
    
def _normalize_str(x: Any) -> str:
    return str(x).strip()

def apply_filters(df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if not rules:
        return df
    m = pd.Series([True] * len(df), index=df.index)
    for col, cond in rules.items():
        if col not in df.columns:
            # missing column: treat as all False for that condition
            m &= False
            continue
        s = df[col].astype(str).fillna("").map(_normalize_str)
        for op, val in cond.items():
            op = op.lower()
            if op == "eq":
                m &= (s.str.lower() == str(val).strip().lower())
            elif op == "ne":
                m &= (s.str.lower() != str(val).strip().lower())
            elif op == "in":
                vals = [str(v).strip().lower() for v in (val if isinstance(val, list) else [val])]
                m &= s.str.lower().isin(vals)
            elif op == "nin":
                vals = [str(v).strip().lower() for v in (val if isinstance(val, list) else [val])]
                m &= ~s.str.lower().isin(vals)
            elif op == "contains":
                m &= s.str.contains(str(val), case=False, na=False)
            elif op == "regex":
                m &= s.str.contains(val, flags=re.I, regex=True, na=False)
            elif op == "notnull":
                m &= s.str.len() > 0
            elif op == "null":
                m &= s.str.len() == 0
            else:
                raise ValueError(f"Unsupported filter op: {op} for column {col}")
    return df[m]


import re
import random

TYLER_CAP = 30
ALI_CAP   = 10
KIZ_CAP   = 10

def _find_last_call_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Last Call With", "last_call_with", "LastCallWith", "Last Call", "last_call"]
    for c in candidates:
        if c in df.columns:
            return c
    # fuzzy fallback
    for c in df.columns:
        if re.fullmatch(r"\s*last\s*call.*", c, flags=re.I):
            return c
    return None

def _name_hits(s: str, pattern: str) -> bool:
    if not isinstance(s, str):
        return False
    return re.search(pattern, s, flags=re.I) is not None

def split_top50_into_tyler_ali_kizmahr(df50: pd.DataFrame, run_id: str) -> Dict[str, pd.DataFrame]:
    """
    Returns dict with keys: 'tyler','ali','kizmahr' (DataFrames).
    Caps: tyler=30, ali=10, kizmahr=10. Uses RUN_ID for deterministic fill.
    """
    col = _find_last_call_col(df50)
    if col is None:
        # No column → everything goes to leftovers to be round-robined by caps
        leftovers_idx = list(df50.index)
        tyler_idx, ali_idx, kiz_idx = [], [], []
    else:
        # Pre-candidate pools (indices)
        tyler_idx = [i for i, v in df50[col].items()
                     if _name_hits(v, r"(meg|tyler|alex|cy)")]
        ali_idx   = [i for i, v in df50[col].items()
                     if _name_hits(v, r"ali")]
        kiz_idx   = [i for i, v in df50[col].items()
                     if _name_hits(v, r"kizmahr")]

        # If any overlap (rare), prefer tyler-group > ali > kiz
        ali_idx   = [i for i in ali_idx if i not in tyler_idx]
        kiz_idx   = [i for i in kiz_idx if i not in tyler_idx and i not in ali_idx]

        matched = set(tyler_idx) | set(ali_idx) | set(kiz_idx)
        leftovers_idx = [i for i in df50.index if i not in matched]

    # Prioritize within tyler: meg > tyler > alex > cy
    def tyler_priority(i: int) -> int:
        if col is None:
            return 999
        v = str(df50.at[i, col])
        if _name_hits(v, r"\btyler"): return 0
        if _name_hits(v, r"\bmeg"):   return 1
        if _name_hits(v, r"\balex"):  return 2
        if _name_hits(v, r"\bcy"):    return 3
        return 9

    tyler_idx_sorted = sorted(tyler_idx, key=tyler_priority)
    tyler_take = tyler_idx_sorted[:TYLER_CAP]
    ali_take   = ali_idx[:ALI_CAP]
    kiz_take   = kiz_idx[:KIZ_CAP]
    
    matched_all = set(tyler_idx) | set(ali_idx) | set(kiz_idx)
    taken_all   = set(tyler_take) | set(ali_take) | set(kiz_take)

    # Base leftovers = everyone not matched (true leftovers)
    leftovers_idx = [i for i in df50.index if i not in matched_all]

    # Plus the overflow from matched groups that didn't fit in caps
    overflow = [i for i in matched_all if i not in taken_all]
    leftovers_idx.extend(overflow)

    # Compute remaining capacity
    need_tyler = TYLER_CAP - len(tyler_take)
    need_ali   = ALI_CAP   - len(ali_take)
    need_kiz   = KIZ_CAP   - len(kiz_take)

    # Deterministic shuffle of leftovers based on RUN_ID
    seed = 0
    try:
        seed = int(re.sub(r"\D", "", run_id)[:9] or "0")
    except Exception:
        pass
    rng = random.Random(seed)
    leftovers_shuffled = leftovers_idx[:]
    rng.shuffle(leftovers_shuffled)

    # Fill remaining capacities with leftovers in order
    def take_from_leftovers(n: int) -> List[int]:
        taken = leftovers_shuffled[:n]
        del leftovers_shuffled[:n]
        return taken

    if need_tyler > 0:
        tyler_take += take_from_leftovers(need_tyler)
    if need_ali > 0:
        ali_take   += take_from_leftovers(need_ali)
    if need_kiz > 0:
        kiz_take   += take_from_leftovers(need_kiz)

    # Build DFs (preserve column order)
    cols = list(df50.columns)
    out = {
        "tyler":   df50.loc[tyler_take, cols].reset_index(drop=True),
        "ali":     df50.loc[ali_take,   cols].reset_index(drop=True),
        "kizmahr": df50.loc[kiz_take,   cols].reset_index(drop=True),
    }
    return out



def date_folder_from_run_id(run_id: str) -> str:
    # RUN_ID like YYYYMMDD_HHMMSS -> turn into YYYY-MM-DD
    try:
        d = datetime.strptime(run_id.split("_")[0], "%Y%m%d").date()
        return d.isoformat()
    except Exception:
        return datetime.utcnow().date().isoformat()

# ================== MAIN ==================
def main():
    run_id = _discover_run_id()
    raw_csv = _find_raw_csv(run_id)
    print(f"[info] RUN_ID={run_id}")
    print(f"[info] raw CSV: {raw_csv}")

    # Output dirs
    out_dir = Path(f"output/processed/{run_id}")
    _ensure_dir(out_dir)

    # Load & light clean
    df = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # --- build contact_distance & order by it (far -> never -> recent) ---
    df = add_contact_distance(df, last_contact_col="Last Contact")
    df = sort_by_contact_distance(df)

    # --- keep top 50 after ordering ---
    df_top = df.head(TOP_N).reset_index(drop=True)
    print(f"[info] rows total={len(df)}, top {TOP_N} kept by contact_distance priority")

    # Split to 30/10/10 using "Last Call With"
    buckets = split_top50_into_tyler_ali_kizmahr(df_top, run_id)

    # Save locally
    top_csv   = out_dir / "top50.csv"
    ty_csv    = out_dir / "tyler.csv"
    ali_csv   = out_dir / "ali.csv"
    kiz_csv   = out_dir / "kizmahr.csv"

    df_top.to_csv(top_csv, index=False)
    buckets["tyler"].to_csv(ty_csv, index=False)
    buckets["ali"].to_csv(ali_csv, index=False)
    buckets["kizmahr"].to_csv(kiz_csv, index=False)

    manifest = {
        "run_id": run_id,
        "source_csv": str(raw_csv),
        "top_n": TOP_N,
        "distance_order": DISTANCE_ORDER,
        "outputs": [str(top_csv), str(ty_csv), str(ali_csv), str(kiz_csv)],
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Upload to Drive (date folder / run subfolder)
    drive = drive_client()
    date_folder = date_folder_from_run_id(run_id)
    parent = drive_create_subfolder(drive, DRIVE_FOLDER_ID, date_folder)
    rid_folder = drive_create_subfolder(drive, parent, f"run_{run_id}")

    for p in [top_csv, ty_csv, ali_csv, kiz_csv]:
        fid = drive_upload_csv(drive, rid_folder, p)
        print(f"[drive] uploaded {p.name} → id {fid}")


    print("[done] process & upload complete.")

if __name__ == "__main__":
    main()
