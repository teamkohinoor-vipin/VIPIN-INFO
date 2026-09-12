import os
import json
import threading
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config (defaults baked in — env vars optional) ──────────────────────────
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/Kzr0xx/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")

INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

REMOTE_INDEXES = {
    "phone":  [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet"  for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── DuckDB singleton ────────────────────────────────────────────────────────
_conn: duckdb.DuckDBPyConnection | None = None
_conn_lock = threading.Lock()


def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet([{lst}])"
        )
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        with _conn_lock:
            if _conn is None:
                _conn = _new_conn()
    return _conn


# ── Dedup & Connected ───────────────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()


def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected


def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out


# ── Search Logic ────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")

    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"):
            view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"):
            view = "people_aadhar"
        else:
            return {"field": field, "value": value, "mode": mode,
                    "count": 0, "results": []}
        sql = (f"SELECT * FROM {view} WHERE {field} = '{v}' "
               f"LIMIT {limit * DUPLICATE_CAP + 20}")
    elif mode == "contains":
        if field == "name":
            return {"field": field, "value": value, "mode": mode,
                    "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' "
               f"ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode,
            "count": len(results), "results": results}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    is_num = q.isdigit() and len(q) >= 8
    if not is_num:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    all_rows, searched = [], []
    if _idx_ready("phone"):
        r = _run_field_search("phoneNumber", q, "exact", limit)
        all_rows.extend(r["results"])
        searched.append("phoneNumber")
    if not all_rows and _idx_ready("aadhar"):
        r = _run_field_search("aadharNumber", q, "exact", limit)
        all_rows.extend(r["results"])
        searched.append("aadharNumber")

    all_rows = _cap_duplicates(all_rows)[:limit]
    return {"query": q, "searched_fields": searched,
            "count": len(all_rows), "results": all_rows}


# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="ICMR + HITEK Search API")


class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10


@app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "records": 2_504_793_870,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "developer": "@kzr0x | channel @api_wallah",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "raw_database_required": False,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE,
    }


@app.get("/search")
async def search(
    q: str | None = Query(None),
    mobile: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile")

    if field:
        data = _run_field_search(field, q_val, mode, limit)
    else:
        data = _unified_search(q_val, limit)

    result = {"success": bool(data["count"]), **data,
              "number": q_val, "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")

    results = [
        _run_field_search(
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    return Response(
        content=json.dumps({"searches": len(req.queries), "results": results},
                           indent=2, ensure_ascii=False),
        media_type="application/json",
    )
