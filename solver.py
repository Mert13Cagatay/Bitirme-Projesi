"""
Tail Assignment Solvers — H1, H2, and LP (conflict-based).
Logic ported verbatim from the user's notebooks (Höristikler.ipynb + LP_copy.ipynb).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import pulp


# ---------- time parsing ----------
def parse_time_robust(date_val, time_val):
    d = pd.to_datetime(date_val).date()
    if pd.isna(time_val):
        return None
    if hasattr(time_val, "hour") and hasattr(time_val, "minute") and not isinstance(time_val, str):
        return datetime.combine(d, time_val)
    if isinstance(time_val, (int, float, np.number)) and not pd.isna(time_val):
        hh = int(float(time_val))
        mm = int(round((float(time_val) - hh) * 100))
        return datetime.combine(d, datetime.min.time()) + timedelta(hours=hh, minutes=mm)
    s = str(time_val).strip()
    add_day = 0
    if s.endswith("+1"):
        add_day = 1
        s = s[:-2].strip()
    if ":" in s:
        parts = s.split(":")
        return datetime.combine(d, datetime.min.time()) + timedelta(
            days=add_day, hours=int(parts[0]), minutes=int(parts[1])
        )
    if s.isdigit():
        if len(s) == 4:
            hh, mm = int(s[:2]), int(s[2:])
        elif len(s) == 3:
            hh, mm = int(s[0]), int(s[1:])
        else:
            hh, mm = int(s), 0
        return datetime.combine(d, datetime.min.time()) + timedelta(days=add_day, hours=hh, minutes=mm)
    raise ValueError(f"Invalid time format: {time_val}")


# ---------- input loader ----------
def load_input(excel_path: str, require_base: bool = True):
    flights = pd.read_excel(excel_path, sheet_name="Flights").copy()
    aircraft = pd.read_excel(excel_path, sheet_name="Aircraft").copy()

    req_f = {"flight_id", "dep", "arr", "req_type", "distance_k"}
    req_a = {"tail", "type", "cons"} | ({"b_i"} if require_base else set())

    miss_f = req_f - set(flights.columns)
    miss_a = req_a - set(aircraft.columns)
    if miss_f:
        raise ValueError(f"Flights sheet missing columns: {miss_f}")
    if miss_a:
        raise ValueError(f"Aircraft sheet missing columns: {miss_a}")

    flights["flight_id"] = flights["flight_id"].astype(int)
    flights["distance_k"] = flights["distance_k"].astype(float)
    flights["req_type"] = flights["req_type"].astype(str).str.strip()
    flights["dep"] = flights["dep"].astype(str).str.strip()
    flights["arr"] = flights["arr"].astype(str).str.strip()

    aircraft["tail"] = aircraft["tail"].astype(str).str.strip()
    aircraft["type"] = aircraft["type"].astype(str).str.strip()
    aircraft["cons"] = aircraft["cons"].astype(float)
    if "b_i" in aircraft.columns:
        aircraft["b_i"] = aircraft["b_i"].astype(str).str.strip()

    if "dep_dt" in flights.columns and "arr_dt" in flights.columns:
        flights["dep_dt"] = pd.to_datetime(flights["dep_dt"])
        flights["arr_dt"] = pd.to_datetime(flights["arr_dt"])
    else:
        needed = {"date", "dep_time", "arr_time"}
        if not needed.issubset(flights.columns):
            raise ValueError("Flights must have (dep_dt, arr_dt) OR (date, dep_time, arr_time).")
        flights["dep_dt"] = [parse_time_robust(d, t) for d, t in zip(flights["date"], flights["dep_time"])]
        flights["arr_dt"] = [parse_time_robust(d, t) for d, t in zip(flights["date"], flights["arr_time"])]

    flights = flights[["flight_id", "dep", "arr", "req_type", "dep_dt", "arr_dt", "distance_k"]].copy()
    cols_a = ["tail", "type", "cons"] + (["b_i"] if "b_i" in aircraft.columns else [])
    aircraft = aircraft[cols_a].copy()
    return flights, aircraft


# ---------- finalize ----------
def _finalize(flights, aircraft, assignments):
    assign_df = pd.DataFrame(assignments)
    if assign_df.empty:
        sol = flights.copy()
        sol["assigned_tail"] = pd.NA
        sol["cons"] = pd.NA
        if "b_i" in aircraft.columns:
            sol["b_i"] = pd.NA
        sol["fuel_cost"] = pd.NA
    else:
        sol = flights.merge(assign_df, on="flight_id", how="left")
        sol = sol.rename(columns={"tail": "assigned_tail"})
        sol = sol.merge(aircraft.rename(columns={"tail": "assigned_tail"}), on="assigned_tail", how="left")
        sol["fuel_cost"] = sol["distance_k"] * sol["cons"]
    total_cost = float(sol["fuel_cost"].fillna(0).sum())
    unassigned = sorted(sol.loc[sol["assigned_tail"].isna(), "flight_id"].tolist())
    return sol.sort_values(["dep_dt", "flight_id"]).reset_index(drop=True), total_cost, unassigned


def build_rotations(sol: pd.DataFrame) -> pd.DataFrame:
    rows = []
    assigned = sol.dropna(subset=["assigned_tail"]).copy()
    for tail, grp in assigned.groupby("assigned_tail"):
        g = grp.sort_values("dep_dt").reset_index(drop=True)
        for k, r in g.iterrows():
            rows.append({
                "tail": tail, "seq": k + 1, "flight_id": int(r["flight_id"]),
                "dep": r["dep"], "arr": r["arr"],
                "dep_dt": r["dep_dt"], "arr_dt": r["arr_dt"],
                "req_type": r["req_type"],
                "fuel_cost": float(r["fuel_cost"]) if pd.notna(r["fuel_cost"]) else np.nan,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["tail", "seq"]).reset_index(drop=True)


def validate_solution(sol: pd.DataFrame) -> List[str]:
    errors = []
    assigned = sol.dropna(subset=["assigned_tail"]).copy()
    bad_type = assigned[assigned["req_type"].astype(str) != assigned["type"].astype(str)]
    for _, r in bad_type.iterrows():
        errors.append(
            f"Type mismatch: flight {r['flight_id']} requires {r['req_type']}, "
            f"assigned aircraft {r['assigned_tail']} is {r['type']}"
        )
    for tail, grp in assigned.groupby("assigned_tail"):
        g = grp.sort_values("dep_dt").reset_index(drop=True)
        if "b_i" in g.columns and not g.empty:
            first = g.iloc[0]
            if pd.notna(first.get("b_i")) and str(first["dep"]) != str(first["b_i"]):
                errors.append(
                    f"Start-base violation: aircraft {tail} starts at {first['b_i']} "
                    f"but first assigned flight {first['flight_id']} departs from {first['dep']}"
                )
        for i in range(len(g) - 1):
            f1, f2 = g.iloc[i], g.iloc[i + 1]
            if str(f1["arr"]) != str(f2["dep"]):
                errors.append(
                    f"Airport continuity violation for {tail}: "
                    f"flight {f1['flight_id']} arrives {f1['arr']}, next {f2['flight_id']} departs {f2['dep']}"
                )
            if f1["arr_dt"] > f2["dep_dt"]:
                errors.append(
                    f"Time violation for {tail}: flight {f1['flight_id']} arrives {f1['arr_dt']}, "
                    f"next {f2['flight_id']} departs {f2['dep_dt']}"
                )
    return errors


# =========================================================
# H1 — aircraft-first with base constraint
# =========================================================
def heuristic_h1(excel_path: str, T_minutes: int = 0):
    flights, aircraft = load_input(excel_path, require_base=True)
    T = timedelta(minutes=T_minutes)
    aircraft = aircraft.sort_values(["cons", "tail"]).reset_index(drop=True)
    flights = flights.sort_values(["dep_dt", "flight_id"]).reset_index(drop=True)

    unassigned = set(flights["flight_id"].tolist())
    assignments = []
    flights_by_id = flights.set_index("flight_id")

    for _, ac in aircraft.iterrows():
        tail, ac_type, base = ac["tail"], ac["type"], ac["b_i"]
        first_candidates = flights[
            (flights["flight_id"].isin(unassigned)) &
            (flights["req_type"] == ac_type) &
            (flights["dep"] == base)
        ].sort_values(["dep_dt", "flight_id"])
        if first_candidates.empty:
            continue
        cur = int(first_candidates.iloc[0]["flight_id"])
        assignments.append({"tail": tail, "flight_id": cur})
        unassigned.remove(cur)
        while True:
            last = flights_by_id.loc[cur]
            nxt = flights[
                (flights["flight_id"].isin(unassigned)) &
                (flights["req_type"] == ac_type) &
                (flights["dep"] == last["arr"]) &
                (flights["dep_dt"] >= last["arr_dt"] + T)
            ].sort_values(["dep_dt", "flight_id"])
            if nxt.empty:
                break
            cur = int(nxt.iloc[0]["flight_id"])
            assignments.append({"tail": tail, "flight_id": cur})
            unassigned.remove(cur)
    sol, total, una = _finalize(flights, aircraft, assignments)
    return sol, total, una


# =========================================================
# H2 — longest-flight-first with base constraint
# =========================================================
def heuristic_h2(excel_path: str, T_minutes: int = 0):
    flights, aircraft = load_input(excel_path, require_base=True)
    T = timedelta(minutes=T_minutes)
    flights_long = flights.sort_values(
        ["distance_k", "dep_dt", "flight_id"], ascending=[False, True, True]
    ).reset_index(drop=True)
    aircraft = aircraft.sort_values(["cons", "tail"]).reset_index(drop=True)

    assignments = []
    ac_state = {
        row["tail"]: {"type": row["type"], "cons": row["cons"], "b_i": row["b_i"],
                      "used": False, "last_arr": None, "last_arr_dt": None}
        for _, row in aircraft.iterrows()
    }

    for _, fl in flights_long.iterrows():
        fid = int(fl["flight_id"])
        feasible = []
        for tail, st in ac_state.items():
            if st["type"] != fl["req_type"]:
                continue
            if not st["used"]:
                if fl["dep"] == st["b_i"]:
                    feasible.append((tail, st["cons"]))
            else:
                if fl["dep"] == st["last_arr"] and fl["dep_dt"] >= st["last_arr_dt"] + T:
                    feasible.append((tail, st["cons"]))
        if not feasible:
            continue
        chosen = sorted(feasible, key=lambda x: (x[1], x[0]))[0][0]
        assignments.append({"tail": chosen, "flight_id": fid})
        st = ac_state[chosen]
        st["used"] = True
        st["last_arr"] = fl["arr"]
        st["last_arr_dt"] = fl["arr_dt"]

    sol, total, una = _finalize(flights, aircraft, assignments)
    return sol, total, una


# =========================================================
# LP — conflict-based PuLP/CBC
# =========================================================
def solve_lp(excel_path: str, T_minutes: int = 0, time_limit_sec: int = 180):
    flights, aircraft = load_input(excel_path, require_base=True)
    T = timedelta(minutes=T_minutes)

    F = flights["flight_id"].tolist()
    I = aircraft["tail"].tolist()
    dep_air = dict(zip(flights["flight_id"], flights["dep"]))
    arr_air = dict(zip(flights["flight_id"], flights["arr"]))
    dep_dt = dict(zip(flights["flight_id"], flights["dep_dt"]))
    arr_dt = dict(zip(flights["flight_id"], flights["arr_dt"]))
    req_type = dict(zip(flights["flight_id"], flights["req_type"]))
    tail_type = dict(zip(aircraft["tail"], aircraft["type"]))
    base = dict(zip(aircraft["tail"], aircraft["b_i"]))
    cons = dict(zip(aircraft["tail"], aircraft["cons"]))
    dist = dict(zip(flights["flight_id"], flights["distance_k"]))

    type_ok = {(i, f): int(tail_type[i] == req_type[f]) for i in I for f in F}
    start_ok = {(i, f): int(base[i] == dep_air[f]) for i in I for f in F}

    precedence = set()
    for f in F:
        for g in F:
            if f == g:
                continue
            if arr_air[f] == dep_air[g] and arr_dt[f] + T <= dep_dt[g]:
                precedence.add((f, g))

    incompatible_pairs = []
    for idx_f in range(len(F)):
        f = F[idx_f]
        for idx_g in range(idx_f + 1, len(F)):
            g = F[idx_g]
            if req_type[f] != req_type[g]:
                continue
            if (f, g) not in precedence and (g, f) not in precedence:
                incompatible_pairs.append((f, g))

    fuel = {(i, f): dist[f] * cons[i] for i in I for f in F}

    global_sources = set()
    for t in sorted(flights["req_type"].unique()):
        Ft = flights[flights["req_type"] == t]
        indeg = {fid: 0 for fid in Ft["flight_id"].tolist()}
        rows = Ft[["flight_id", "dep", "arr", "dep_dt", "arr_dt"]].to_dict("records")
        for fr in rows:
            for gr in rows:
                if fr["flight_id"] == gr["flight_id"]:
                    continue
                if str(fr["arr"]) == str(gr["dep"]) and fr["arr_dt"] + T <= gr["dep_dt"]:
                    indeg[gr["flight_id"]] += 1
        global_sources |= {fid for fid, d in indeg.items() if d == 0}

    prob = pulp.LpProblem("TAP_ConflictBased", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (I, F), lowBound=0, upBound=1, cat=pulp.LpBinary)
    prob += pulp.lpSum(fuel[(i, f)] * x[i][f] for i in I for f in F)
    for f in F:
        prob += pulp.lpSum(x[i][f] for i in I) == 1, f"cover_{f}"
    for i in I:
        for f in F:
            if type_ok[(i, f)] == 0:
                prob += x[i][f] == 0, f"type_{i}_{f}"
    for i in I:
        itype = tail_type[i]
        for (f, g) in incompatible_pairs:
            if req_type[f] == itype:
                prob += x[i][f] + x[i][g] <= 1, f"conflict_{i}_{f}_{g}"
    for i in I:
        for f in global_sources:
            if type_ok[(i, f)] == 1 and start_ok[(i, f)] == 0:
                prob += x[i][f] == 0, f"source_base_{i}_{f}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    prob.solve(solver)
    status = pulp.LpStatus.get(prob.status, str(prob.status))

    assignments = []
    for f in F:
        for i in I:
            v = pulp.value(x[i][f])
            if v is not None and v > 0.5:
                assignments.append({"tail": i, "flight_id": f})
                break
    sol, total, una = _finalize(flights, aircraft, assignments)
    return sol, total, una, status


# ---------- serialization ----------
def sol_to_payload(sol: pd.DataFrame, total_cost: float, unassigned: List[int],
                   model: str, status: str | None = None) -> Dict[str, Any]:
    rotations = build_rotations(sol)
    errors = validate_solution(sol)

    sol_out = sol.copy()
    for c in ("dep_dt", "arr_dt"):
        if c in sol_out.columns:
            sol_out[c] = pd.to_datetime(sol_out[c]).dt.strftime("%Y-%m-%d %H:%M")
    rot_out = rotations.copy()
    for c in ("dep_dt", "arr_dt"):
        if c in rot_out.columns:
            rot_out[c] = pd.to_datetime(rot_out[c]).dt.strftime("%Y-%m-%d %H:%M")

    by_tail = (
        sol.dropna(subset=["assigned_tail"])
        .groupby("assigned_tail")
        .agg(n_flights=("flight_id", "count"),
             total_distance_k=("distance_k", "sum"),
             total_fuel_cost=("fuel_cost", "sum"))
        .reset_index()
        .sort_values("total_fuel_cost", ascending=False)
    )

    return {
        "model": model,
        "status": status or "OK",
        "summary": {
            "total_flights": int(len(sol)),
            "assigned_flights": int(sol["assigned_tail"].notna().sum()),
            "unassigned_flights": int(len(unassigned)),
            "total_fuel_cost": float(total_cost),
            "tails_used": int(sol["assigned_tail"].dropna().nunique()),
            "validation_error_count": int(len(errors)),
        },
        "unassigned_flight_ids": unassigned,
        "validation_errors": errors[:200],
        "solution": sol_out.replace({np.nan: None}).to_dict(orient="records"),
        "rotations": rot_out.replace({np.nan: None}).to_dict(orient="records") if not rot_out.empty else [],
        "by_tail": by_tail.replace({np.nan: None}).to_dict(orient="records"),
    }