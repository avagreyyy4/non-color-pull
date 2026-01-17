#!/usr/bin/env python3
# ================== PACKAGES ==================
import os, json, re, random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials as UserCreds

# ================== ENV / CREDS ==================
def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

DRIVE_FOLDER_ID = _req("DRIVE_FOLDER_ID")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def drive_client():
    rtok = os.getenv("GDRIVE_REFRESH_TOKEN")
    if not rtok:
        raise RuntimeError("Missing GDRIVE_REFRESH_TOKEN.")
    creds = UserCreds(
        token=None,
        refresh_token=rtok,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GDRIVE_CLIENT_ID"),
        client_secret=os.getenv("GDRIVE_CLIENT_SECRET"),
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds)

def drive_create_subfolder(drive, parent_id: str, name: str) -> str:
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return drive.files().create(
        body=meta, fields="id", supportsAllDrives=True
    ).execute()["id"]

def drive_upload_csv(drive, folder_id: str, path: Path) -> str:
    meta = {"name": path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(path), mimetype="text/csv", resumable=False)
    return drive.files().create(
        body=meta, media_body=media, fields="id", supportsAllDrives=True
    ).execute()["id"]

# ================== CONFIG ==================
FILTERS: Dict[str, Dict[str, Any]] = {
    "Mobile Phone": {"notnull": True},
    # Optional:
    # "texted_distance": {"ne": "recent"},
}

TOP_N = 50
DISTANCE_ORDER = ["far", "never", "recent"]

TYLER_CAP = 30
ALLIE_CAP = 10
KIZ_CAP = 10

# ================== HELPERS ==================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _normalize_str(x: Any) -> str:
    return str(x).strip()

def apply_filters(df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if not rules:
        return df
    m = pd.Series(True, index=df.index)
    for col, cond in rules.items():
        if col not in df.columns:
            m &= False
            continue
        s = df[col].astype(str).fillna("").map(_normalize_str)
        for op, val in cond.items():
            op = op.lower()
            if op == "eq":
                m &= s.str.lower() == str(val).lower()
            elif op == "ne":
                m &= s.str.lower() != str(val).lower()
            elif op == "contains":
                m &= s.str.contains(str(val), case=False, na=False)
            elif op == "notnull":
                m &= s.str.len() > 0
            elif op == "null":
                m &= s.str.len() == 0
            else:
                raise ValueError(f"Unsupported filter op: {op}")
    return df[m]

# ================== RECENCY LOGIC ==================
def add_recency_bucket(
    df: pd.DataFrame,
    date_col: str,
    prefix: str,
    recent_days: int = 365,
) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col].replace({"": None}), errors="coerce", utc=True)

    today = pd.Timestamp.now(tz="UTC").normalize()
    cutoff = today - pd.Timedelta(days=recent_days)

    dist = pd.Series("far", index=df.index)
    dist[dt.isna()] = "never"
    dist[dt >= cutoff] = "recent"

    df[f"{prefix}_dt"] = dt
    df[f"{prefix}_distance"] = dist
    return df

def sort_by_contact_and_texted(df: pd.DataFrame) -> pd.DataFrame:
    order = {k: i for i, k in enumerate(DISTANCE_ORDER)}
    today = pd.Timestamp.now(tz="UTC").normalize()

    return (
        df.assign(
            _contact_rank=df["contact_distance"].map(order).fillna(999),
            _texted_rank=df["texted_distance"].map(order).fillna(999),
            _contact_days=(today - df["contact_dt"]).dt.days.fillna(9999),
            _texted_days=(today - df["texted_dt"]).dt.days.fillna(9999),
        )
        .sort_values(
            by=[
                "_contact_rank",
                "_texted_rank",
                "_contact_days",
                "_texted_days",
            ],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .drop(
            columns=[
                "_contact_rank",
                "_texted_rank",
                "_contact_days",
                "_texted_days",
            ]
        )
    )

# ================== SPLITTING ==================
def _find_last_call_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.fullmatch(r"\s*last\s*call.*", c, flags=re.I):
            return c
    return None

def _name_hits(s: str, pattern: str) -> bool:
    return isinstance(s, str) and re.search(pattern, s, flags=re.I)

def split_top50_into_tyler_allie_kizmahr(df50: pd.DataFrame, run_id: str):
    col = _find_last_call_col(df50)
    idx = list(df50.index)

    tyler, allie, kiz = [], [], []

    if col:
        for i in idx:
            v = df50.at[i, col]
            if _name_hits(v, r"(meg|tyler|grossman|cy)"):
                tyler.append(i)
            elif _name_hits(v, r"(allie|alexandra)"):
                allie.append(i)
            elif _name_hits(v, r"kizmahr"):
                kiz.append(i)

    used = set(tyler + allie + kiz)
    leftovers = [i for i in idx if i not in used]

    def cap(xs, n): return xs[:n]

    tyler = cap(tyler, TYLER_CAP)
    allie = cap(allie, ALLIE_CAP)
    kiz = cap(kiz, KIZ_CAP)

    seed = int(re.sub(r"\D", "", run_id)[:8] or "0")
    rng = random.Random(seed)
    rng.shuffle(leftovers)

    def fill(xs, cap):
        need = cap - len(xs)
        take = leftovers[:need]
        del leftovers[:need]
        return xs + take

    tyler = fill(tyler, TYLER_CAP)
    allie = fill(allie, ALLIE_CAP)
    kiz = fill(kiz, KIZ_CAP)

    return {
        "tyler": df50.loc[tyler].reset_index(drop=True),
        "allie": df50.loc[allie].reset_index(drop=True),
        "kizmahr": df50.loc[kiz].reset_index(drop=True),
    }

# ================== MAIN ==================
def main():
    run_id = os.getenv("RUN_ID")
    if not run_id:
        raise RuntimeError("RUN_ID not set")

    raw_csv = Path(f"output/raw/export_{run_id}.csv")
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)

    out_dir = Path(f"output/processed/{run_id}")
    _ensure_dir(out_dir)

    df = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # Filters
    df = apply_filters(df, FILTERS)

    # Add recency logic
    df = add_recency_bucket(df, "Last Contact", prefix="contact")
    df = add_recency_bucket(df, "Last Texted", prefix="texted")

    # Sort + top N
    df = sort_by_contact_and_texted(df)
    df_top = df.head(TOP_N).reset_index(drop=True)

    # Split
    buckets = split_top50_into_tyler_allie_kizmahr(df_top, run_id)

    # Save
    paths = {
        "top50": out_dir / "top50.csv",
        "tyler": out_dir / "tyler.csv",
        "allie": out_dir / "allie.csv",
        "kizmahr": out_dir / "kizmahr.csv",
    }

    df_top.to_csv(paths["top50"], index=False)
    for k in ["tyler", "allie", "kizmahr"]:
        buckets[k].to_csv(paths[k], index=False)

    # Upload
    drive = drive_client()
    date_folder = datetime.utcnow().date().isoformat()
    parent = drive_create_subfolder(drive, DRIVE_FOLDER_ID, date_folder)
    rid_folder = drive_create_subfolder(drive, parent, f"run_{run_id}")

    for p in paths.values():
        drive_upload_csv(drive, rid_folder, p)

    print("[done] process & upload complete.")

if __name__ == "__main__":
    main()
