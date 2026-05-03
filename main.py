"""
FastAPI server exposing tail-assignment solvers (H1, H2, LP).

Run locally:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000
"""
import io
import os
import tempfile
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import solver as S

app = FastAPI(title="Tail Assignment Solver API")

# CORS — allow the Lovable frontend (and local dev) to call us.
ALLOWED = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _save_upload(file: UploadFile) -> str:
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Please upload a .xlsx file")
    suffix = ".xlsx"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(file.file.read())
    return path


@app.post("/solve")
async def solve(
    model: Literal["H1", "H2", "LP"] = Form(...),
    t_minutes: int = Form(0),
    time_limit_sec: int = Form(180),
    file: UploadFile = File(...),
):
    path = _save_upload(file)
    try:
        if model == "H1":
            sol, total, una = S.heuristic_h1(path, T_minutes=t_minutes)
            return S.sol_to_payload(sol, total, una, "H1")
        if model == "H2":
            sol, total, una = S.heuristic_h2(path, T_minutes=t_minutes)
            return S.sol_to_payload(sol, total, una, "H2")
        if model == "LP":
            sol, total, una, status = S.solve_lp(path, T_minutes=t_minutes, time_limit_sec=time_limit_sec)
            return S.sol_to_payload(sol, total, una, "LP", status=status)
        raise HTTPException(400, "Unknown model")
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/export")
async def export(
    model: Literal["H1", "H2", "LP"] = Form(...),
    t_minutes: int = Form(0),
    time_limit_sec: int = Form(180),
    file: UploadFile = File(...),
):
    """Run solver and return an Excel workbook with solution + rotations."""
    path = _save_upload(file)
    try:
        if model == "H1":
            sol, total, una = S.heuristic_h1(path, T_minutes=t_minutes)
        elif model == "H2":
            sol, total, una = S.heuristic_h2(path, T_minutes=t_minutes)
        else:
            sol, total, una, _ = S.solve_lp(path, T_minutes=t_minutes, time_limit_sec=time_limit_sec)

        rotations = S.build_rotations(sol)
        errors = S.validate_solution(sol)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            sol.to_excel(w, sheet_name="Solution", index=False)
            (rotations if not rotations.empty else pd.DataFrame()).to_excel(
                w, sheet_name="Rotations", index=False
            )
            pd.DataFrame([{
                "model": model,
                "total_fuel_cost": total,
                "assigned": int(sol["assigned_tail"].notna().sum()),
                "unassigned": len(una),
                "validation_errors": len(errors),
            }]).to_excel(w, sheet_name="Summary", index=False)
            if errors:
                pd.DataFrame({"error": errors}).to_excel(w, sheet_name="Errors", index=False)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="solution_{model}.xlsx"'},
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass