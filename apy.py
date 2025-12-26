# app.py
"""
Intelligent Study Planner & Academic Performance Optimizer (Streamlit)
Single-file, production-oriented app:
- SQLite persistence
- Adaptive schedule generation
- Editable schedules
- Progress tracking + analytics
- Predictive regression + what-if + risk alerts
- PDF report download

Run:
  python -m venv .venv
  source .venv/bin/activate  (Windows: .venv\\Scripts\\activate)
  pip install -U streamlit pandas numpy scikit-learn plotly reportlab
  streamlit run app.py
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Intelligent Study Planner & Academic Performance Optimizer"
DB_PATH_DEFAULT = "study_planner.sqlite3"
DATE_FMT = "%Y-%m-%d"

TIME_BLOCKS = ["Morning", "Afternoon", "Evening", "Night", "Any"]

CHAPTER_STATUSES = ["not_started", "in_progress", "done"]
SCHEDULE_STATUSES = ["planned", "done", "skipped"]

QUALITY_SCALE = [1, 2, 3, 4, 5]


# -----------------------------
# Utilities
# -----------------------------
def today_local() -> date:
    return date.today()


def iso(d: date) -> str:
    return d.strftime(DATE_FMT)


def parse_iso(s: str) -> date:
    return datetime.strptime(s, DATE_FMT).date()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return int(v)
    except Exception:
        return default


def weekday_name(i: int) -> str:
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i]


def normalize_0_1(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def round_to_half_hour(hours: float) -> float:
    return round(hours * 2) / 2.0


# -----------------------------
# Database
# -----------------------------
def db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS subjects (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE,
          target_score REAL NOT NULL DEFAULT 75
        );

        CREATE TABLE IF NOT EXISTS chapters (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          subject_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          difficulty INTEGER NOT NULL DEFAULT 3,
          est_hours REAL NOT NULL DEFAULT 2.0,
          status TEXT NOT NULL DEFAULT 'not_started',
          completion REAL NOT NULL DEFAULT 0.0,
          last_updated TEXT NOT NULL DEFAULT (DATE('now')),
          UNIQUE(subject_id, name),
          FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS exams (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          subject_id INTEGER NOT NULL,
          name TEXT NOT NULL,
          exam_date TEXT NOT NULL,
          weightage REAL NOT NULL DEFAULT 1.0,
          priority INTEGER NOT NULL DEFAULT 3,
          notes TEXT,
          FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS availability (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          weekday INTEGER NOT NULL UNIQUE,
          hours REAL NOT NULL DEFAULT 2.0,
          preferred_block TEXT NOT NULL DEFAULT 'Any'
        );

        CREATE TABLE IF NOT EXISTS performance (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          subject_id INTEGER NOT NULL,
          assessment_name TEXT NOT NULL,
          assessment_date TEXT NOT NULL,
          score REAL NOT NULL,
          max_score REAL NOT NULL DEFAULT 100,
          kind TEXT NOT NULL DEFAULT 'quiz',
          linked_exam_id INTEGER,
          notes TEXT,
          FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
          FOREIGN KEY(linked_exam_id) REFERENCES exams(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS study_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          log_date TEXT NOT NULL,
          subject_id INTEGER NOT NULL,
          chapter_id INTEGER,
          planned_hours REAL NOT NULL DEFAULT 0.0,
          actual_hours REAL NOT NULL DEFAULT 0.0,
          quality INTEGER NOT NULL DEFAULT 3,
          notes TEXT,
          FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
          FOREIGN KEY(chapter_id) REFERENCES chapters(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS schedule_items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          item_date TEXT NOT NULL,
          subject_id INTEGER NOT NULL,
          chapter_id INTEGER,
          planned_hours REAL NOT NULL DEFAULT 1.0,
          time_block TEXT NOT NULL DEFAULT 'Any',
          status TEXT NOT NULL DEFAULT 'planned',
          notes TEXT,
          source TEXT NOT NULL DEFAULT 'auto',
          created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
          FOREIGN KEY(subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
          FOREIGN KEY(chapter_id) REFERENCES chapters(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chapter_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          chapter_id INTEGER NOT NULL,
          event_date TEXT NOT NULL,
          completion REAL NOT NULL,
          status TEXT NOT NULL,
          FOREIGN KEY(chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_schedule_date ON schedule_items(item_date);
        CREATE INDEX IF NOT EXISTS idx_logs_date ON study_logs(log_date);
        CREATE INDEX IF NOT EXISTS idx_perf_date ON performance(assessment_date);
        CREATE INDEX IF NOT EXISTS idx_chapter_events_date ON chapter_events(event_date);
        """
    )

    # Seed availability rows
    existing = conn.execute("SELECT COUNT(*) AS c FROM availability").fetchone()["c"]
    if existing == 0:
        for wd in range(7):
            conn.execute(
                "INSERT INTO availability(weekday, hours, preferred_block) VALUES(?,?,?)",
                (wd, 2.0 if wd < 5 else 3.0, "Any"),
            )
    # Seed settings
    _set_setting_if_missing(conn, "db_version", "1")
    _set_setting_if_missing(conn, "planning_horizon_days", "30")
    _set_setting_if_missing(conn, "min_chunk_hours", "0.5")
    _set_setting_if_missing(conn, "auto_overwrite_future_auto", "true")
    _set_setting_if_missing(conn, "urgency_alpha", "1.4")
    _set_setting_if_missing(conn, "weight_difficulty", "0.35")
    _set_setting_if_missing(conn, "weight_weakness", "0.35")
    _set_setting_if_missing(conn, "weight_completion_gap", "0.30")
    _set_setting_if_missing(conn, "risk_buffer_factor", "0.90")
    conn.commit()


def _set_setting_if_missing(conn: sqlite3.Connection, key: str, value: str) -> None:
    row = conn.execute("SELECT key FROM settings WHERE key=?", (key,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO settings(key,value) VALUES(?,?)", (key, value))


def get_setting(conn: sqlite3.Connection, key: str, default: str) -> str:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


# -----------------------------
# Data Access Layer
# -----------------------------
def df_query(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def fetch_subjects(conn: sqlite3.Connection) -> pd.DataFrame:
    return df_query(conn, "SELECT * FROM subjects ORDER BY name")


def fetch_chapters(conn: sqlite3.Connection) -> pd.DataFrame:
    return df_query(
        conn,
        """
        SELECT c.*, s.name AS subject_name
        FROM chapters c
        JOIN subjects s ON s.id = c.subject_id
        ORDER BY s.name, c.name
        """,
    )


def fetch_exams(conn: sqlite3.Connection) -> pd.DataFrame:
    return df_query(
        conn,
        """
        SELECT e.*, s.name AS subject_name
        FROM exams e
        JOIN subjects s ON s.id = e.subject_id
        ORDER BY e.exam_date, s.name
        """,
    )


def fetch_availability(conn: sqlite3.Connection) -> pd.DataFrame:
    return df_query(conn, "SELECT * FROM availability ORDER BY weekday")


def fetch_schedule(conn: sqlite3.Connection, start: date, end: date) -> pd.DataFrame:
    return df_query(
        conn,
        """
        SELECT si.*, s.name AS subject_name, c.name AS chapter_name
        FROM schedule_items si
        JOIN subjects s ON s.id = si.subject_id
        LEFT JOIN chapters c ON c.id = si.chapter_id
        WHERE si.item_date BETWEEN ? AND ?
        ORDER BY si.item_date, si.time_block, s.name, c.name
        """,
        (iso(start), iso(end)),
    )


def fetch_logs(conn: sqlite3.Connection, start: date, end: date) -> pd.DataFrame:
    return df_query(
        conn,
        """
        SELECT l.*, s.name AS subject_name, c.name AS chapter_name
        FROM study_logs l
        JOIN subjects s ON s.id = l.subject_id
        LEFT JOIN chapters c ON c.id = l.chapter_id
        WHERE l.log_date BETWEEN ? AND ?
        ORDER BY l.log_date DESC
        """,
        (iso(start), iso(end)),
    )


def fetch_performance(conn: sqlite3.Connection) -> pd.DataFrame:
    return df_query(
        conn,
        """
        SELECT p.*, s.name AS subject_name
        FROM performance p
        JOIN subjects s ON s.id = p.subject_id
        ORDER BY p.assessment_date DESC
        """,
    )


def upsert_subject(conn: sqlite3.Connection, name: str, target_score: float) -> None:
    conn.execute(
        "INSERT INTO subjects(name,target_score) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET target_score=excluded.target_score",
        (name.strip(), float(target_score)),
    )
    conn.commit()


def delete_subject(conn: sqlite3.Connection, subject_id: int) -> None:
    conn.execute("DELETE FROM subjects WHERE id=?", (subject_id,))
    conn.commit()


def upsert_chapter(
    conn: sqlite3.Connection,
    subject_id: int,
    name: str,
    difficulty: int,
    est_hours: float,
    status: str,
    completion: float,
) -> None:
    name = name.strip()
    completion = clamp(float(completion), 0.0, 1.0)
    status = status if status in CHAPTER_STATUSES else "not_started"
    difficulty = int(clamp(int(difficulty), 1, 5))
    est_hours = max(0.5, float(est_hours))

    conn.execute(
        """
        INSERT INTO chapters(subject_id,name,difficulty,est_hours,status,completion,last_updated)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(subject_id,name) DO UPDATE SET
          difficulty=excluded.difficulty,
          est_hours=excluded.est_hours,
          status=excluded.status,
          completion=excluded.completion,
          last_updated=excluded.last_updated
        """,
        (subject_id, name, difficulty, est_hours, status, completion, iso(today_local())),
    )
    # Create/append event for this chapter change
    chap = conn.execute(
        "SELECT id, completion, status FROM chapters WHERE subject_id=? AND name=?",
        (subject_id, name),
    ).fetchone()
    if chap:
        conn.execute(
            "INSERT INTO chapter_events(chapter_id,event_date,completion,status) VALUES(?,?,?,?)",
            (chap["id"], iso(today_local()), float(chap["completion"]), str(chap["status"])),
        )
    conn.commit()


def update_chapter_progress(conn: sqlite3.Connection, chapter_id: int, status: str, completion: float) -> None:
    completion = clamp(float(completion), 0.0, 1.0)
    status = status if status in CHAPTER_STATUSES else "not_started"
    conn.execute(
        "UPDATE chapters SET status=?, completion=?, last_updated=? WHERE id=?",
        (status, completion, iso(today_local()), chapter_id),
    )
    conn.execute(
        "INSERT INTO chapter_events(chapter_id,event_date,completion,status) VALUES(?,?,?,?)",
        (chapter_id, iso(today_local()), completion, status),
    )
    conn.commit()


def delete_chapter(conn: sqlite3.Connection, chapter_id: int) -> None:
    conn.execute("DELETE FROM chapters WHERE id=?", (chapter_id,))
    conn.commit()


def upsert_exam(
    conn: sqlite3.Connection,
    subject_id: int,
    name: str,
    exam_date: date,
    weightage: float,
    priority: int,
    notes: str,
) -> None:
    conn.execute(
        """
        INSERT INTO exams(subject_id,name,exam_date,weightage,priority,notes)
        VALUES(?,?,?,?,?,?)
        """,
        (subject_id, name.strip(), iso(exam_date), float(weightage), int(priority), notes.strip() or None),
    )
    conn.commit()


def delete_exam(conn: sqlite3.Connection, exam_id: int) -> None:
    conn.execute("DELETE FROM exams WHERE id=?", (exam_id,))
    conn.commit()


def upsert_availability(conn: sqlite3.Connection, weekday: int, hours: float, preferred_block: str) -> None:
    preferred_block = preferred_block if preferred_block in TIME_BLOCKS else "Any"
    conn.execute(
        """
        INSERT INTO availability(weekday,hours,preferred_block)
        VALUES(?,?,?)
        ON CONFLICT(weekday) DO UPDATE SET hours=excluded.hours, preferred_block=excluded.preferred_block
        """,
        (int(weekday), float(hours), preferred_block),
    )
    conn.commit()


def add_study_log(
    conn: sqlite3.Connection,
    log_date: date,
    subject_id: int,
    chapter_id: Optional[int],
    planned_hours: float,
    actual_hours: float,
    quality: int,
    notes: str,
) -> None:
    conn.execute(
        """
        INSERT INTO study_logs(log_date,subject_id,chapter_id,planned_hours,actual_hours,quality,notes)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            iso(log_date),
            subject_id,
            chapter_id,
            float(planned_hours),
            float(actual_hours),
            int(quality),
            notes.strip() or None,
        ),
    )
    conn.commit()


def delete_log(conn: sqlite3.Connection, log_id: int) -> None:
    conn.execute("DELETE FROM study_logs WHERE id=?", (log_id,))
    conn.commit()


def add_performance(
    conn: sqlite3.Connection,
    subject_id: int,
    assessment_name: str,
    assessment_date: date,
    score: float,
    max_score: float,
    kind: str,
    linked_exam_id: Optional[int],
    notes: str,
) -> None:
    conn.execute(
        """
        INSERT INTO performance(subject_id,assessment_name,assessment_date,score,max_score,kind,linked_exam_id,notes)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            subject_id,
            assessment_name.strip(),
            iso(assessment_date),
            float(score),
            float(max_score),
            kind.strip() or "quiz",
            linked_exam_id,
            notes.strip() or None,
        ),
    )
    conn.commit()


def delete_performance(conn: sqlite3.Connection, perf_id: int) -> None:
    conn.execute("DELETE FROM performance WHERE id=?", (perf_id,))
    conn.commit()


def bulk_update_schedule(conn: sqlite3.Connection, updates: pd.DataFrame) -> None:
    """
    updates must contain:
      id, planned_hours, time_block, status, notes
    """
    cur = conn.cursor()
    for _, r in updates.iterrows():
        cur.execute(
            """
            UPDATE schedule_items
            SET planned_hours=?, time_block=?, status=?, notes=?
            WHERE id=?
            """,
            (
                float(r["planned_hours"]),
                str(r["time_block"]),
                str(r["status"]),
                (str(r["notes"]).strip() if pd.notna(r["notes"]) else None),
                int(r["id"]),
            ),
        )
    conn.commit()


def insert_schedule_item(
    conn: sqlite3.Connection,
    item_date: date,
    subject_id: int,
    chapter_id: Optional[int],
    planned_hours: float,
    time_block: str,
    status: str,
    notes: str,
    source: str,
) -> None:
    time_block = time_block if time_block in TIME_BLOCKS else "Any"
    status = status if status in SCHEDULE_STATUSES else "planned"
    conn.execute(
        """
        INSERT INTO schedule_items(item_date,subject_id,chapter_id,planned_hours,time_block,status,notes,source)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (
            iso(item_date),
            subject_id,
            chapter_id,
            float(planned_hours),
            time_block,
            status,
            notes.strip() or None,
            source,
        ),
    )


def delete_future_auto_schedule(conn: sqlite3.Connection, start: date, end: date) -> None:
    conn.execute(
        """
        DELETE FROM schedule_items
        WHERE item_date BETWEEN ? AND ? AND source='auto' AND status='planned'
        """,
        (iso(start), iso(end)),
    )
    conn.commit()


# -----------------------------
# Planning / Scoring
# -----------------------------
@dataclass(frozen=True)
class PlannerWeights:
    urgency_alpha: float
    w_difficulty: float
    w_weakness: float
    w_completion_gap: float
    min_chunk_hours: float


def load_weights(conn: sqlite3.Connection) -> PlannerWeights:
    return PlannerWeights(
        urgency_alpha=safe_float(get_setting(conn, "urgency_alpha", "1.4"), 1.4),
        w_difficulty=safe_float(get_setting(conn, "weight_difficulty", "0.35"), 0.35),
        w_weakness=safe_float(get_setting(conn, "weight_weakness", "0.35"), 0.35),
        w_completion_gap=safe_float(get_setting(conn, "weight_completion_gap", "0.30"), 0.30),
        min_chunk_hours=max(0.25, safe_float(get_setting(conn, "min_chunk_hours", "0.5"), 0.5)),
    )


def subject_recent_score_norm(perf_df: pd.DataFrame, subject_id: int, lookback_n: int = 5) -> float:
    if perf_df.empty:
        return 0.65
    sdf = perf_df[perf_df["subject_id"] == subject_id].copy()
    if sdf.empty:
        return 0.65
    sdf["pct"] = (sdf["score"] / sdf["max_score"]).clip(0, 1) * 100.0
    sdf = sdf.sort_values("assessment_date", ascending=False).head(lookback_n)
    return float(sdf["pct"].mean() / 100.0)


def nearest_exam_for_subject(exams_df: pd.DataFrame, subject_id: int, ref: date) -> Optional[pd.Series]:
    sdf = exams_df[exams_df["subject_id"] == subject_id].copy()
    if sdf.empty:
        return None
    sdf["days_to"] = sdf["exam_date"].apply(lambda s: (parse_iso(s) - ref).days)
    sdf = sdf[sdf["days_to"] >= 0].sort_values("days_to", ascending=True)
    if sdf.empty:
        return None
    return sdf.iloc[0]


def compute_chapter_priority(
    ref: date,
    chapter_row: pd.Series,
    exams_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    subject_target: float,
    weights: PlannerWeights,
) -> float:
    """
    Returns a bounded score in [0, ~5] (not strictly capped but stabilized).
    """
    subject_id = int(chapter_row["subject_id"])

    completion = float(chapter_row["completion"])
    status = str(chapter_row["status"])
    if status == "done" or completion >= 0.999:
        return 0.0

    nearest_exam = nearest_exam_for_subject(exams_df, subject_id, ref)
    if nearest_exam is None:
        days_to_exam = 60
        weightage = 1.0
        exam_priority = 3
    else:
        days_to_exam = int(nearest_exam["days_to"])
        weightage = float(nearest_exam["weightage"])
        exam_priority = int(nearest_exam["priority"])

    urgency = 1.0 / float(days_to_exam + 1)
    urgency = urgency ** float(weights.urgency_alpha)

    difficulty = normalize_0_1(float(chapter_row["difficulty"]), 1.0, 5.0)
    completion_gap = 1.0 - clamp(completion, 0.0, 1.0)

    recent_norm = subject_recent_score_norm(perf_df, subject_id)
    weakness = 1.0 - clamp(recent_norm, 0.0, 1.0)

    # Weightage/priority boosts (bounded)
    weightage_boost = 1.0 + clamp((weightage - 1.0) / 4.0, 0.0, 0.6)  # weightage 1..5 -> up to +0.6
    priority_boost = 1.0 + clamp((exam_priority - 3.0) / 5.0, -0.3, 0.4)  # 1..5 -> +/- bounded

    # Target score affects boost slightly: higher target => more aggressive planning
    target_boost = 1.0 + clamp((float(subject_target) - 75.0) / 100.0, -0.2, 0.3)

    base = (
        weights.w_difficulty * difficulty
        + weights.w_weakness * weakness
        + weights.w_completion_gap * completion_gap
    )

    score = urgency * base * weightage_boost * priority_boost * target_boost

    # Stabilize scale for usability
    return float(clamp(score * 6.0, 0.0, 6.0))


def estimate_remaining_hours(chapter_row: pd.Series) -> float:
    est = float(chapter_row["est_hours"])
    completion = float(chapter_row["completion"])
    remaining = max(0.0, est * (1.0 - completion))
    return float(max(0.25, remaining))


def availability_hours_map(av_df: pd.DataFrame) -> Dict[int, Tuple[float, str]]:
    m: Dict[int, Tuple[float, str]] = {}
    for _, r in av_df.iterrows():
        m[int(r["weekday"])] = (float(r["hours"]), str(r["preferred_block"]))
    for wd in range(7):
        m.setdefault(wd, (2.0, "Any"))
    return m


def generate_schedule_auto(
    conn: sqlite3.Connection,
    start: date,
    end: date,
    overwrite_future_auto: bool,
) -> Dict[str, Any]:
    subjects = fetch_subjects(conn)
    chapters = fetch_chapters(conn)
    exams = fetch_exams(conn)
    perf = fetch_performance(conn)
    av = fetch_availability(conn)

    weights = load_weights(conn)
    av_map = availability_hours_map(av)

    if subjects.empty or chapters.empty:
        return {"created": 0, "message": "Add subjects and chapters before generating a schedule."}

    chapters_active = chapters.copy()
    chapters_active = chapters_active[(chapters_active["status"] != "done") & (chapters_active["completion"] < 0.999)]
    if chapters_active.empty:
        return {"created": 0, "message": "All chapters are completed. Great job — nothing to plan!"}

    if overwrite_future_auto:
        delete_future_auto_schedule(conn, start, end)

    subject_target = {int(r["id"]): float(r["target_score"]) for _, r in subjects.iterrows()}

    # Precompute chapter priority & remaining hours
    rows = []
    for _, ch in chapters_active.iterrows():
        s_id = int(ch["subject_id"])
        score = compute_chapter_priority(
            ref=start,
            chapter_row=ch,
            exams_df=exams,
            perf_df=perf,
            subject_target=subject_target.get(s_id, 75.0),
            weights=weights,
        )
        rows.append(
            {
                "chapter_id": int(ch["id"]),
                "subject_id": s_id,
                "chapter_name": str(ch["name"]),
                "subject_name": str(ch["subject_name"]),
                "priority": float(score),
                "remaining_hours": float(estimate_remaining_hours(ch)),
                "difficulty": int(ch["difficulty"]),
            }
        )
    cand = pd.DataFrame(rows)
    if cand.empty:
        return {"created": 0, "message": "No active chapters found."}

    cand["priority"] = cand["priority"].clip(0, 6)
    cand = cand.sort_values(["priority", "difficulty"], ascending=[False, False])

    # Track remaining hours per chapter
    remaining = {int(r["chapter_id"]): float(r["remaining_hours"]) for _, r in cand.iterrows()}

    created = 0
    cur = conn.cursor()

    day = start
    while day <= end:
        wd = day.weekday()
        day_hours, pref_block = av_map.get(wd, (2.0, "Any"))
        day_hours = max(0.0, float(day_hours))
        if day_hours <= 0.0:
            day += timedelta(days=1)
            continue

        # Recompute urgency relative to each day (not just start)
        # (cheap approximation for responsiveness)
        cand2 = cand.copy()
        if not exams.empty:
            # Faster: compute subject-level urgency multiplier for this day
            urg_mult: Dict[int, float] = {}
            for s_id in cand2["subject_id"].unique():
                ex = nearest_exam_for_subject(exams, int(s_id), day)
                if ex is None:
                    urg_mult[int(s_id)] = 1.0
                else:
                    days_to = int(ex["days_to"])
                    urgency = (1.0 / float(days_to + 1)) ** float(weights.urgency_alpha)
                    urg_mult[int(s_id)] = float(clamp(urgency * 4.0, 0.6, 3.0))
            cand2["priority_day"] = cand2["subject_id"].apply(lambda s: float(urg_mult.get(int(s), 1.0))) * cand2[
                "priority"
            ]
        else:
            cand2["priority_day"] = cand2["priority"]

        # Remove items already completed in remaining tracker
        cand2 = cand2[cand2["chapter_id"].apply(lambda cid: remaining.get(int(cid), 0.0) > 0.0)]
        cand2 = cand2.sort_values("priority_day", ascending=False)

        if cand2.empty:
            day += timedelta(days=1)
            continue

        hours_left = float(day_hours)
        min_chunk = float(weights.min_chunk_hours)

        for _, r in cand2.iterrows():
            if hours_left < min_chunk:
                break
            ch_id = int(r["chapter_id"])
            s_id = int(r["subject_id"])
            rem = float(remaining.get(ch_id, 0.0))
            if rem <= 0.0:
                continue

            # Allocation: proportional chunk with a cap
            prop = float(r["priority_day"]) / float(max(cand2["priority_day"].sum(), 1e-9))
            chunk = max(min_chunk, hours_left * prop)
            chunk = min(chunk, rem, hours_left)
            chunk = round_to_half_hour(chunk)
            if chunk < min_chunk:
                continue

            insert_schedule_item(
                conn=conn,
                item_date=day,
                subject_id=s_id,
                chapter_id=ch_id,
                planned_hours=chunk,
                time_block=pref_block,
                status="planned",
                notes="Auto-planned",
                source="auto",
            )
            created += 1
            remaining[ch_id] = max(0.0, rem - chunk)
            hours_left = max(0.0, hours_left - chunk)

            if hours_left < min_chunk:
                break

        day += timedelta(days=1)

    conn.commit()
    return {"created": created, "message": f"Created {created} schedule items."}


# -----------------------------
# Progress & Risk
# -----------------------------
def subject_completion_now(chapters_df: pd.DataFrame, subject_id: int) -> float:
    sdf = chapters_df[chapters_df["subject_id"] == subject_id]
    if sdf.empty:
        return 0.0
    return float(sdf["completion"].clip(0, 1).mean())


def completion_at_date(conn: sqlite3.Connection, subject_id: int, at: date) -> float:
    """
    Uses chapter_events (last event <= at) per chapter; falls back to 0 if none.
    """
    chapters = df_query(
        conn, "SELECT id FROM chapters WHERE subject_id=?", (subject_id,)
    )
    if chapters.empty:
        return 0.0

    vals = []
    for ch_id in chapters["id"].tolist():
        row = conn.execute(
            """
            SELECT completion FROM chapter_events
            WHERE chapter_id=? AND event_date<=?
            ORDER BY event_date DESC, id DESC
            LIMIT 1
            """,
            (int(ch_id), iso(at)),
        ).fetchone()
        vals.append(float(row["completion"]) if row else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def required_daily_hours_for_exam(
    conn: sqlite3.Connection,
    chapters_df: pd.DataFrame,
    subject_id: int,
    exam_date: date,
) -> Tuple[float, float, int]:
    """
    Returns (required_daily_hours, remaining_hours, days_left)
    """
    sdf = chapters_df[chapters_df["subject_id"] == subject_id].copy()
    if sdf.empty:
        return 0.0, 0.0, (exam_date - today_local()).days

    sdf["remaining"] = sdf.apply(estimate_remaining_hours, axis=1)
    # Ignore done chapters
    sdf = sdf[(sdf["status"] != "done") & (sdf["completion"] < 0.999)]
    remaining_hours = float(sdf["remaining"].sum())
    days_left = max(0, (exam_date - today_local()).days)
    if days_left <= 0:
        return float("inf") if remaining_hours > 0 else 0.0, remaining_hours, days_left
    return remaining_hours / float(days_left), remaining_hours, days_left


def avg_daily_available_hours(av_df: pd.DataFrame) -> float:
    if av_df.empty:
        return 2.0
    return float(av_df["hours"].mean())


def build_risk_alerts(
    conn: sqlite3.Connection,
    subjects_df: pd.DataFrame,
    chapters_df: pd.DataFrame,
    exams_df: pd.DataFrame,
    av_df: pd.DataFrame,
    predicted_scores: Optional[pd.DataFrame],
) -> List[str]:
    alerts: List[str] = []
    buffer_factor = safe_float(get_setting(conn, "risk_buffer_factor", "0.90"), 0.90)
    avail_daily = avg_daily_available_hours(av_df) * float(buffer_factor)

    for _, ex in exams_df.iterrows():
        ex_date = parse_iso(str(ex["exam_date"]))
        if ex_date < today_local():
            continue
        s_id = int(ex["subject_id"])
        subj_name = str(ex["subject_name"])
        req_daily, rem_hours, days_left = required_daily_hours_for_exam(conn, chapters_df, s_id, ex_date)

        if rem_hours <= 0.1:
            continue

        if days_left <= 0:
            alerts.append(f"🚨 {subj_name}: Exam is today or past, but remaining syllabus is not complete.")
            continue

        if req_daily > avail_daily:
            alerts.append(
                f"⚠️ {subj_name}: Needs ~{req_daily:.2f}h/day to finish remaining work ({rem_hours:.1f}h) before {iso(ex_date)}. "
                f"Your average available is ~{avail_daily:.2f}h/day."
            )

        # Predicted-score risk
        if predicted_scores is not None and not predicted_scores.empty:
            p = predicted_scores[predicted_scores["subject_id"] == s_id]
            if not p.empty:
                pred = float(p.iloc[0]["predicted_score"])
                target = float(subjects_df[subjects_df["id"] == s_id]["target_score"].iloc[0])
                if pred + 1e-6 < target:
                    alerts.append(
                        f"📉 {subj_name}: Predicted score {pred:.1f} is below target {target:.1f}. Consider increasing study hours or revising weak chapters."
                    )

    return alerts


# -----------------------------
# ML: Prediction
# -----------------------------
def build_training_dataset(conn: sqlite3.Connection) -> pd.DataFrame:
    perf = fetch_performance(conn)
    if perf.empty:
        return pd.DataFrame()

    chapters = fetch_chapters(conn)
    if chapters.empty:
        return pd.DataFrame()

    # Difficulty mean per subject
    diff_mean = chapters.groupby("subject_id")["difficulty"].mean().to_dict()

    rows = []
    for _, p in perf.iterrows():
        s_id = int(p["subject_id"])
        ad = parse_iso(str(p["assessment_date"]))

        # Hours studied last 14 days
        start = ad - timedelta(days=14)
        logs = df_query(
            conn,
            """
            SELECT SUM(actual_hours) AS h
            FROM study_logs
            WHERE subject_id=? AND log_date BETWEEN ? AND ?
            """,
            (s_id, iso(start), iso(ad - timedelta(days=1))),
        )
        h14 = float(logs.iloc[0]["h"]) if not logs.empty and logs.iloc[0]["h"] is not None else 0.0

        comp = completion_at_date(conn, s_id, ad)
        difficulty = float(diff_mean.get(s_id, 3.0))

        # Optional days_to_exam if linked
        linked_exam_id = p["linked_exam_id"]
        days_to_exam = 0
        if linked_exam_id is not None and not (isinstance(linked_exam_id, float) and np.isnan(linked_exam_id)):
            ex = df_query(conn, "SELECT exam_date FROM exams WHERE id=?", (int(linked_exam_id),))
            if not ex.empty:
                exd = parse_iso(str(ex.iloc[0]["exam_date"]))
                days_to_exam = max(0, (exd - ad).days)

        pct = float((p["score"] / p["max_score"]) * 100.0) if float(p["max_score"]) > 0 else 0.0
        rows.append(
            {
                "subject_id": s_id,
                "hours_14d": h14,
                "completion": comp,
                "difficulty_mean": difficulty,
                "days_to_exam": days_to_exam,
                "y": clamp(pct, 0.0, 100.0),
            }
        )
    return pd.DataFrame(rows)


def train_model(train_df: pd.DataFrame) -> Optional[Pipeline]:
    if train_df is None or train_df.empty or len(train_df) < 10:
        return None

    X = train_df[["subject_id", "hours_14d", "completion", "difficulty_mean", "days_to_exam"]]
    y = train_df["y"]

    pre = ColumnTransformer(
        transformers=[
            ("subj", OneHotEncoder(handle_unknown="ignore"), ["subject_id"]),
            ("num", StandardScaler(), ["hours_14d", "completion", "difficulty_mean", "days_to_exam"]),
        ]
    )
    model = Ridge(alpha=1.0, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)
    return pipe


def predict_upcoming_scores(
    conn: sqlite3.Connection,
    pipe: Pipeline,
    what_if_multiplier: float = 1.0,
) -> pd.DataFrame:
    subjects = fetch_subjects(conn)
    chapters = fetch_chapters(conn)
    exams = fetch_exams(conn)

    if subjects.empty:
        return pd.DataFrame()

    # Difficulty mean per subject
    diff_mean = chapters.groupby("subject_id")["difficulty"].mean().to_dict() if not chapters.empty else {}

    rows = []
    for _, s in subjects.iterrows():
        s_id = int(s["id"])
        # Choose nearest exam date for horizon features
        ex = nearest_exam_for_subject(exams, s_id, today_local())
        if ex is None:
            days_to_exam = 30
            horizon_end = today_local() + timedelta(days=30)
        else:
            exd = parse_iso(str(ex["exam_date"]))
            days_to_exam = max(0, (exd - today_local()).days)
            horizon_end = exd

        # Project hours in next 14 days: use schedule planned hours + availability if missing
        h_start = today_local()
        h_end = min(horizon_end, today_local() + timedelta(days=14))
        sch = df_query(
            conn,
            """
            SELECT SUM(planned_hours) AS h
            FROM schedule_items
            WHERE subject_id=? AND item_date BETWEEN ? AND ?
            """,
            (s_id, iso(h_start), iso(h_end)),
        )
        planned = float(sch.iloc[0]["h"]) if not sch.empty and sch.iloc[0]["h"] is not None else 0.0

        # Fallback: estimate planned from availability
        if planned <= 0.01:
            av = fetch_availability(conn)
            avg_daily = avg_daily_available_hours(av)
            days = max(1, (h_end - h_start).days + 1)
            planned = avg_daily * float(days) / max(1, len(subjects))

        planned *= float(what_if_multiplier)

        comp = subject_completion_now(chapters, s_id) if not chapters.empty else 0.0
        difficulty = float(diff_mean.get(s_id, 3.0))

        rows.append(
            {
                "subject_id": s_id,
                "subject_name": str(s["name"]),
                "hours_14d": planned,
                "completion": comp,
                "difficulty_mean": difficulty,
                "days_to_exam": days_to_exam,
            }
        )

    Xp = pd.DataFrame(rows)
    if Xp.empty:
        return pd.DataFrame()
    yp = pipe.predict(Xp[["subject_id", "hours_14d", "completion", "difficulty_mean", "days_to_exam"]])
    Xp["predicted_score"] = np.clip(yp, 0, 100)
    return Xp.sort_values("predicted_score", ascending=False)


# -----------------------------
# PDF Report
# -----------------------------
def draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    leading: float = 14,
    font_name: str = "Helvetica",
    font_size: int = 10,
) -> float:
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if stringWidth(test, font_name, font_size) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def generate_pdf_report(
    title: str,
    generated_on: datetime,
    schedule_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    alerts: List[str],
    predictions_df: Optional[pd.DataFrame],
) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin = 2 * cm
    x = margin
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Generated: {generated_on.strftime('%Y-%m-%d %H:%M')}")
    y -= 18

    # Alerts
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Risk Alerts")
    y -= 14
    if not alerts:
        c.setFont("Helvetica", 10)
        c.drawString(x, y, "No risk alerts detected.")
        y -= 14
    else:
        for a in alerts[:12]:
            y = draw_wrapped_text(c, f"- {a}", x, y, width - 2 * margin, leading=12, font_size=9)
            if y < margin + 100:
                c.showPage()
                y = height - margin

    # Progress
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Progress Summary")
    y -= 14
    if progress_df is not None and not progress_df.empty:
        show = progress_df.copy().head(12)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x, y, "Subject")
        c.drawString(x + 240, y, "Completion")
        c.drawString(x + 340, y, "Hours (30d)")
        y -= 12
        c.setFont("Helvetica", 9)
        for _, r in show.iterrows():
            c.drawString(x, y, str(r["subject_name"])[:30])
            c.drawString(x + 240, y, f"{float(r['completion_pct']):.0f}%")
            c.drawString(x + 340, y, f"{float(r['hours_30d']):.1f}")
            y -= 12
            if y < margin + 120:
                c.showPage()
                y = height - margin
    else:
        c.setFont("Helvetica", 10)
        c.drawString(x, y, "No progress data available yet.")
        y -= 14

    # Predictions
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Predictions")
    y -= 14
    if predictions_df is None or predictions_df.empty:
        c.setFont("Helvetica", 10)
        c.drawString(x, y, "Not enough historical data to train a prediction model (need ~10+ records).")
        y -= 14
    else:
        show = predictions_df.copy().sort_values("predicted_score", ascending=False).head(10)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x, y, "Subject")
        c.drawString(x + 260, y, "Predicted Score")
        c.drawString(x + 380, y, "Completion")
        y -= 12
        c.setFont("Helvetica", 9)
        for _, r in show.iterrows():
            c.drawString(x, y, str(r["subject_name"])[:30])
            c.drawString(x + 260, y, f"{float(r['predicted_score']):.1f}")
            c.drawString(x + 380, y, f"{float(r['completion'])*100:.0f}%")
            y -= 12
            if y < margin + 120:
                c.showPage()
                y = height - margin

    # Schedule
    c.showPage()
    y = height - margin
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Next 7 Days Schedule")
    y -= 14
    if schedule_df is None or schedule_df.empty:
        c.setFont("Helvetica", 10)
        c.drawString(x, y, "No schedule items found for the next 7 days.")
        y -= 14
    else:
        show = schedule_df.copy().head(40)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x, y, "Date")
        c.drawString(x + 70, y, "Time")
        c.drawString(x + 120, y, "Subject")
        c.drawString(x + 260, y, "Chapter")
        c.drawString(x + 430, y, "Hours")
        y -= 10
        c.setFont("Helvetica", 8)
        for _, r in show.iterrows():
            c.drawString(x, y, str(r["item_date"]))
            c.drawString(x + 70, y, str(r["time_block"]))
            c.drawString(x + 120, y, str(r["subject_name"])[:18])
            c.drawString(x + 260, y, (str(r["chapter_name"]) if pd.notna(r["chapter_name"]) else "-")[:24])
            c.drawString(x + 430, y, f"{float(r['planned_hours']):.1f}")
            y -= 10
            if y < margin + 40:
                c.showPage()
                y = height - margin

    c.save()
    return buf.getvalue()


# -----------------------------
# UI Helpers
# -----------------------------
def select_subject(subjects_df: pd.DataFrame, label: str, key: str) -> Optional[int]:
    if subjects_df.empty:
        st.warning("Add at least one subject first.")
        return None
    names = subjects_df["name"].tolist()
    idx = st.selectbox(label, list(range(len(names))), format_func=lambda i: names[i], key=key)
    return int(subjects_df.iloc[int(idx)]["id"])


def ensure_db_in_session() -> str:
    if "db_path" not in st.session_state:
        st.session_state.db_path = DB_PATH_DEFAULT
    return st.session_state.db_path


# -----------------------------
# Streamlit App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")

    st.title("📚 Intelligent Study Planner & Academic Performance Optimizer")
    st.caption("Production-grade, adaptive study planning + analytics + prediction with persistent storage (SQLite).")

    db_path = ensure_db_in_session()
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Setup", "Planner", "Analytics", "Predict", "Reports", "Settings"],
            index=1,
        )
        st.divider()
        st.subheader("Database")
        db_path_input = st.text_input("SQLite DB Path", value=db_path, help="Use a file path; will be created if missing.")
        if db_path_input.strip() and db_path_input != db_path:
            st.session_state.db_path = db_path_input.strip()
            st.rerun()

        st.caption("Tip: Keep your DB file in your project folder for easy backup/versioning.")

    conn = db_connect(st.session_state.db_path)
    db_init(conn)

    if page == "Setup":
        ui_setup(conn)
    elif page == "Planner":
        ui_planner(conn)
    elif page == "Analytics":
        ui_analytics(conn)
    elif page == "Predict":
        ui_predict(conn)
    elif page == "Reports":
        ui_reports(conn)
    elif page == "Settings":
        ui_settings(conn)


def ui_setup(conn: sqlite3.Connection) -> None:
    st.subheader("Setup: Subjects, Chapters, Exams, Availability, Performance")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Subjects", "Chapters", "Exams", "Availability", "Performance"]
    )

    with tab1:
        subjects = fetch_subjects(conn)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Add / Update Subject")
            with st.form("add_subject", clear_on_submit=True):
                name = st.text_input("Subject Name", placeholder="e.g., Mathematics")
                target = st.slider("Target Score", 40, 100, 80)
                submitted = st.form_submit_button("Save Subject")
            if submitted:
                if not name.strip():
                    st.error("Subject name is required.")
                else:
                    upsert_subject(conn, name, float(target))
                    st.success("Saved.")
                    st.rerun()

        with c2:
            st.markdown("### Existing Subjects")
            subjects = fetch_subjects(conn)
            if subjects.empty:
                st.info("No subjects yet.")
            else:
                st.dataframe(subjects, use_container_width=True, hide_index=True)
                del_id = st.selectbox(
                    "Delete subject (cascades chapters/exams/logs)",
                    [None] + subjects["id"].tolist(),
                    format_func=lambda x: "Select..." if x is None else subjects[subjects["id"] == x]["name"].iloc[0],
                )
                if st.button("Delete Selected Subject", type="secondary", disabled=del_id is None):
                    delete_subject(conn, int(del_id))
                    st.success("Deleted.")
                    st.rerun()

    with tab2:
        subjects = fetch_subjects(conn)
        chapters = fetch_chapters(conn)
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("### Add / Update Chapter")
            s_id = select_subject(subjects, "Subject", key="chap_subj")
            if s_id is not None:
                with st.form("add_chapter", clear_on_submit=True):
                    name = st.text_input("Chapter / Topic Name", placeholder="e.g., Quadratic Equations")
                    difficulty = st.slider("Difficulty (1 easy → 5 hard)", 1, 5, 3)
                    est_hours = st.number_input("Estimated Hours to Master", min_value=0.5, max_value=200.0, value=4.0, step=0.5)
                    status = st.selectbox("Status", CHAPTER_STATUSES, index=0)
                    completion = st.slider("Completion", 0, 100, 0) / 100.0
                    submitted = st.form_submit_button("Save Chapter")
                if submitted:
                    if not name.strip():
                        st.error("Chapter name is required.")
                    else:
                        upsert_chapter(conn, s_id, name, difficulty, est_hours, status, completion)
                        st.success("Saved.")
                        st.rerun()

        with c2:
            st.markdown("### Chapter Manager")
            chapters = fetch_chapters(conn)
            if chapters.empty:
                st.info("No chapters yet.")
            else:
                # Quick inline progress update
                subject_filter = st.selectbox(
                    "Filter by subject",
                    ["All"] + sorted(chapters["subject_name"].unique().tolist()),
                )
                view = chapters.copy()
                if subject_filter != "All":
                    view = view[view["subject_name"] == subject_filter]

                display = view[["id", "subject_name", "name", "difficulty", "est_hours", "status", "completion", "last_updated"]].copy()
                display.rename(columns={"name": "chapter"}, inplace=True)
                st.dataframe(display, use_container_width=True, hide_index=True)

                st.markdown("#### Update Progress")
                ch_id = st.selectbox(
                    "Select chapter",
                    [None] + view["id"].tolist(),
                    format_func=lambda x: "Select..." if x is None else f"{view[view['id'] == x]['subject_name'].iloc[0]} — {view[view['id'] == x]['name'].iloc[0]}",
                )
                if ch_id is not None:
                    row = view[view["id"] == ch_id].iloc[0]
                    new_status = st.selectbox("Status", CHAPTER_STATUSES, index=CHAPTER_STATUSES.index(str(row["status"])))
                    new_completion = st.slider("Completion (%)", 0, 100, int(float(row["completion"]) * 100))
                    if st.button("Save Progress Update"):
                        update_chapter_progress(conn, int(ch_id), new_status, float(new_completion) / 100.0)
                        st.success("Updated.")
                        st.rerun()

                del_ch_id = st.selectbox(
                    "Delete chapter",
                    [None] + view["id"].tolist(),
                    format_func=lambda x: "Select..." if x is None else view[view["id"] == x]["name"].iloc[0],
                    key="del_ch",
                )
                if st.button("Delete Selected Chapter", type="secondary", disabled=del_ch_id is None):
                    delete_chapter(conn, int(del_ch_id))
                    st.success("Deleted.")
                    st.rerun()

    with tab3:
        subjects = fetch_subjects(conn)
        exams = fetch_exams(conn)

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("### Add Exam")
            s_id = select_subject(subjects, "Subject", key="exam_subj")
            if s_id is not None:
                with st.form("add_exam", clear_on_submit=True):
                    name = st.text_input("Exam Name", placeholder="e.g., Midterm / Final / Board")
                    ex_date = st.date_input("Exam Date", value=today_local() + timedelta(days=14))
                    weightage = st.slider("Weightage (importance)", 1.0, 5.0, 3.0, 0.5)
                    priority = st.slider("Priority", 1, 5, 3)
                    notes = st.text_area("Notes (optional)")
                    submitted = st.form_submit_button("Add Exam")
                if submitted:
                    if not name.strip():
                        st.error("Exam name is required.")
                    else:
                        upsert_exam(conn, s_id, name, ex_date, float(weightage), int(priority), notes)
                        st.success("Added.")
                        st.rerun()

        with c2:
            st.markdown("### Existing Exams")
            exams = fetch_exams(conn)
            if exams.empty:
                st.info("No exams yet.")
            else:
                show = exams.copy()
                show["exam_date"] = show["exam_date"].astype(str)
                st.dataframe(show[["id", "subject_name", "name", "exam_date", "weightage", "priority", "notes"]], use_container_width=True, hide_index=True)
                del_id = st.selectbox(
                    "Delete exam",
                    [None] + exams["id"].tolist(),
                    format_func=lambda x: "Select..." if x is None else f"{exams[exams['id'] == x]['subject_name'].iloc[0]} — {exams[exams['id'] == x]['name'].iloc[0]}",
                )
                if st.button("Delete Selected Exam", type="secondary", disabled=del_id is None):
                    delete_exam(conn, int(del_id))
                    st.success("Deleted.")
                    st.rerun()

    with tab4:
        st.markdown("### Weekly Availability & Preferred Study Time")
        av = fetch_availability(conn)
        if av.empty:
            st.info("No availability rows found (will be seeded automatically).")
        else:
            ed = av.copy()
            ed["weekday"] = ed["weekday"].apply(lambda x: weekday_name(int(x)))
            st.dataframe(ed, use_container_width=True, hide_index=True)

        st.markdown("#### Edit Availability")
        av = fetch_availability(conn)
        for _, r in av.iterrows():
            wd = int(r["weekday"])
            with st.expander(f"{weekday_name(wd)} settings", expanded=False):
                hours = st.number_input(
                    f"Hours available ({weekday_name(wd)})",
                    min_value=0.0,
                    max_value=24.0,
                    value=float(r["hours"]),
                    step=0.5,
                    key=f"av_h_{wd}",
                )
                pref = st.selectbox(
                    f"Preferred block ({weekday_name(wd)})",
                    TIME_BLOCKS,
                    index=TIME_BLOCKS.index(str(r["preferred_block"])) if str(r["preferred_block"]) in TIME_BLOCKS else TIME_BLOCKS.index("Any"),
                    key=f"av_p_{wd}",
                )
                if st.button(f"Save {weekday_name(wd)}", key=f"save_av_{wd}"):
                    upsert_availability(conn, wd, float(hours), pref)
                    st.success("Saved.")
                    st.rerun()

    with tab5:
        subjects = fetch_subjects(conn)
        exams = fetch_exams(conn)

        st.markdown("### Add Performance Record (Marks/Grades)")
        if subjects.empty:
            st.warning("Add subjects first.")
            return
        s_id = select_subject(subjects, "Subject", key="perf_subj")
        linked_exam_options = [None] + exams["id"].tolist()
        linked_exam_id = st.selectbox(
            "Link to an exam (optional, recommended)",
            linked_exam_options,
            format_func=lambda x: "None" if x is None else f"{exams[exams['id'] == x]['subject_name'].iloc[0]} — {exams[exams['id'] == x]['name'].iloc[0]} ({exams[exams['id']==x]['exam_date'].iloc[0]})",
        )

        with st.form("add_perf", clear_on_submit=True):
            assessment_name = st.text_input("Assessment Name", placeholder="e.g., Quiz 2 / Mock Test")
            assessment_date = st.date_input("Date", value=today_local())
            c1, c2 = st.columns(2)
            with c1:
                score = st.number_input("Score", min_value=0.0, max_value=1000.0, value=78.0, step=1.0)
            with c2:
                max_score = st.number_input("Max Score", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
            kind = st.selectbox("Type", ["quiz", "assignment", "midterm", "mock", "final", "other"])
            notes = st.text_area("Notes (optional)")
            submitted = st.form_submit_button("Add Performance Record")
        if submitted:
            if not assessment_name.strip():
                st.error("Assessment name is required.")
            else:
                add_performance(
                    conn=conn,
                    subject_id=int(s_id),
                    assessment_name=assessment_name,
                    assessment_date=assessment_date,
                    score=float(score),
                    max_score=float(max_score),
                    kind=kind,
                    linked_exam_id=int(linked_exam_id) if linked_exam_id is not None else None,
                    notes=notes,
                )
                st.success("Added.")
                st.rerun()

        st.markdown("### Performance History")
        perf = fetch_performance(conn)
        if perf.empty:
            st.info("No performance records yet.")
        else:
            perf_show = perf.copy()
            perf_show["pct"] = (perf_show["score"] / perf_show["max_score"]).clip(0, 1) * 100.0
            st.dataframe(
                perf_show[["id", "subject_name", "assessment_name", "assessment_date", "score", "max_score", "pct", "kind", "linked_exam_id", "notes"]],
                use_container_width=True,
                hide_index=True,
            )
            del_id = st.selectbox("Delete performance record", [None] + perf["id"].tolist(), format_func=lambda x: "Select..." if x is None else f"#{x}")
            if st.button("Delete Selected Performance", type="secondary", disabled=del_id is None):
                delete_performance(conn, int(del_id))
                st.success("Deleted.")
                st.rerun()


def ui_planner(conn: sqlite3.Connection) -> None:
    st.subheader("Planner: Generate, Edit, and Track Your Study Schedule")

    subjects = fetch_subjects(conn)
    chapters = fetch_chapters(conn)
    exams = fetch_exams(conn)
    av = fetch_availability(conn)

    if subjects.empty or chapters.empty:
        st.info("Start in **Setup**: add Subjects → Chapters → Exams → Availability.")
        return

    cA, cB = st.columns([1.1, 1.0])
    with cA:
        st.markdown("### Auto Plan Generator")
        horizon_default = safe_int(get_setting(conn, "planning_horizon_days", "30"), 30)
        start = st.date_input("Start date", value=today_local(), key="plan_start")
        end = st.date_input("End date", value=today_local() + timedelta(days=horizon_default), key="plan_end")
        overwrite = st.checkbox(
            "Overwrite future auto-planned items (safe regen)",
            value=(get_setting(conn, "auto_overwrite_future_auto", "true").lower() == "true"),
        )

        if st.button("⚙️ Generate / Refresh Schedule", type="primary"):
            if end < start:
                st.error("End date must be after start date.")
            else:
                result = generate_schedule_auto(conn, start, end, overwrite_future_auto=overwrite)
                st.success(result["message"])
                st.rerun()

        st.markdown("### Daily Study Log")
        with st.form("add_log", clear_on_submit=True):
            log_date = st.date_input("Log Date", value=today_local())
            s_id = select_subject(subjects, "Subject (for log)", key="log_subj")
            chapters_for_subject = chapters[chapters["subject_id"] == s_id] if s_id is not None else pd.DataFrame()
            chap_options = [None] + (chapters_for_subject["id"].tolist() if not chapters_for_subject.empty else [])
            ch_id = st.selectbox(
                "Chapter (optional)",
                chap_options,
                format_func=lambda x: "None" if x is None else chapters_for_subject[chapters_for_subject["id"] == x]["name"].iloc[0],
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                planned = st.number_input("Planned Hours", min_value=0.0, max_value=24.0, value=0.0, step=0.5)
            with c2:
                actual = st.number_input("Actual Hours", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
            with c3:
                quality = st.selectbox("Quality (1–5)", QUALITY_SCALE, index=2)
            notes = st.text_area("Notes (optional)")
            submitted = st.form_submit_button("Add Study Log")
        if submitted:
            if s_id is None:
                st.error("Select a subject.")
            else:
                add_study_log(conn, log_date, int(s_id), int(ch_id) if ch_id is not None else None, planned, actual, int(quality), notes)
                st.success("Logged.")
                st.rerun()

    with cB:
        st.markdown("### This Week: Editable Schedule")
        week_start = st.date_input("Week starting", value=(today_local() - timedelta(days=today_local().weekday())))
        week_end = week_start + timedelta(days=6)

        sched = fetch_schedule(conn, week_start, week_end)
        if sched.empty:
            st.info("No schedule items in this week. Generate a schedule or add items manually in Settings (advanced).")
        else:
            editable = sched.copy()
            # Keep id for updates (hidden via column config)
            editable["planned_hours"] = editable["planned_hours"].astype(float)
            editable["notes"] = editable["notes"].fillna("")

            edited = st.data_editor(
                editable[["id", "item_date", "time_block", "subject_name", "chapter_name", "planned_hours", "status", "notes", "source"]],
                use_container_width=True,
                hide_index=True,
                disabled=["id", "subject_name", "chapter_name", "source", "item_date"],
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True),
                    "item_date": st.column_config.TextColumn("Date"),
                    "planned_hours": st.column_config.NumberColumn("Planned Hours", min_value=0.0, max_value=24.0, step=0.5),
                    "time_block": st.column_config.SelectboxColumn("Time Block", options=TIME_BLOCKS),
                    "status": st.column_config.SelectboxColumn("Status", options=SCHEDULE_STATUSES),
                    "notes": st.column_config.TextColumn("Notes"),
                    "source": st.column_config.TextColumn("Source"),
                },
            )

            if st.button("💾 Save Schedule Changes"):
                updates = edited[["id", "planned_hours", "time_block", "status", "notes"]].copy()
                bulk_update_schedule(conn, updates)
                st.success("Saved.")
                st.rerun()

        st.markdown("### Last 14 Days Logs")
        logs = fetch_logs(conn, today_local() - timedelta(days=14), today_local())
        if logs.empty:
            st.info("No logs yet.")
        else:
            st.dataframe(
                logs[["id", "log_date", "subject_name", "chapter_name", "planned_hours", "actual_hours", "quality", "notes"]],
                use_container_width=True,
                hide_index=True,
            )
            del_id = st.selectbox("Delete log entry", [None] + logs["id"].tolist(), format_func=lambda x: "Select..." if x is None else f"#{x}")
            if st.button("Delete Selected Log", type="secondary", disabled=del_id is None):
                delete_log(conn, int(del_id))
                st.success("Deleted.")
                st.rerun()


def ui_analytics(conn: sqlite3.Connection) -> None:
    st.subheader("Analytics: Progress, Weak Areas, and Trends")

    subjects = fetch_subjects(conn)
    chapters = fetch_chapters(conn)
    exams = fetch_exams(conn)
    av = fetch_availability(conn)
    perf = fetch_performance(conn)

    if subjects.empty:
        st.info("Add subjects first.")
        return

    # Progress per subject
    logs_30d = fetch_logs(conn, today_local() - timedelta(days=30), today_local())
    hours_30d = logs_30d.groupby("subject_id")["actual_hours"].sum().to_dict() if not logs_30d.empty else {}

    progress_rows = []
    for _, s in subjects.iterrows():
        s_id = int(s["id"])
        comp = subject_completion_now(chapters, s_id) if not chapters.empty else 0.0
        progress_rows.append(
            {
                "subject_id": s_id,
                "subject_name": str(s["name"]),
                "completion_pct": comp * 100.0,
                "hours_30d": float(hours_30d.get(s_id, 0.0)),
                "target_score": float(s["target_score"]),
            }
        )
    progress = pd.DataFrame(progress_rows).sort_values("completion_pct", ascending=False)

    c1, c2 = st.columns([1.1, 1.0])
    with c1:
        st.markdown("### Subject-wise Progress")
        fig = px.bar(progress, x="subject_name", y="completion_pct", hover_data=["hours_30d", "target_score"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Planned vs Actual (Last 30 Days)")
        if logs_30d.empty:
            st.info("No logs in the last 30 days.")
        else:
            pva = logs_30d.copy()
            pva["log_date"] = pd.to_datetime(pva["log_date"])
            daily = pva.groupby("log_date")[["planned_hours", "actual_hours"]].sum().reset_index()
            fig2 = px.line(daily, x="log_date", y=["planned_hours", "actual_hours"])
            st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("### Weak Area Detection")
        if chapters.empty:
            st.info("No chapters yet.")
        else:
            # Weak score proxy: low completion + higher difficulty + lower recent scores
            perf_norm = perf.copy()
            if not perf_norm.empty:
                perf_norm["pct"] = (perf_norm["score"] / perf_norm["max_score"]).clip(0, 1)
            recent_by_subject = (
                perf_norm.sort_values("assessment_date", ascending=False)
                .groupby("subject_id")["pct"]
                .head(5)
                .groupby(level=0)
                .mean()
                .to_dict()
                if not perf_norm.empty
                else {}
            )
            ch = chapters.copy()
            ch["recent_norm"] = ch["subject_id"].apply(lambda sid: float(recent_by_subject.get(int(sid), 0.65)))
            ch["weakness"] = 1.0 - ch["recent_norm"].clip(0, 1)
            ch["completion_gap"] = (1.0 - ch["completion"].clip(0, 1))
            ch["difficulty_norm"] = (ch["difficulty"] - 1) / 4.0
            ch["weak_score"] = 0.45 * ch["weakness"] + 0.35 * ch["completion_gap"] + 0.20 * ch["difficulty_norm"]
            weak = ch[(ch["status"] != "done") & (ch["completion"] < 0.999)].sort_values("weak_score", ascending=False).head(12)
            if weak.empty:
                st.success("No weak areas detected (or all chapters complete).")
            else:
                st.dataframe(
                    weak[["subject_name", "name", "difficulty", "completion", "weak_score"]].rename(columns={"name": "chapter"}),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Score Trend Over Time")
        if perf.empty:
            st.info("No performance records yet.")
        else:
            p = perf.copy()
            p["pct"] = (p["score"] / p["max_score"]).clip(0, 1) * 100.0
            p["assessment_date"] = pd.to_datetime(p["assessment_date"])
            fig3 = px.line(p.sort_values("assessment_date"), x="assessment_date", y="pct", color="subject_name", markers=True)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Progress Table")
    st.dataframe(progress, use_container_width=True, hide_index=True)

    # Optional: exam readiness summary
    if not exams.empty:
        st.markdown("### Exam Readiness (Pacing)")
        rows = []
        for _, ex in exams.iterrows():
            ex_date = parse_iso(str(ex["exam_date"]))
            if ex_date < today_local():
                continue
            s_id = int(ex["subject_id"])
            req_daily, rem_hours, days_left = required_daily_hours_for_exam(conn, chapters, s_id, ex_date)
            rows.append(
                {
                    "subject": str(ex["subject_name"]),
                    "exam": str(ex["name"]),
                    "exam_date": iso(ex_date),
                    "days_left": days_left,
                    "remaining_hours": rem_hours,
                    "required_h_per_day": float(req_daily) if math.isfinite(req_daily) else None,
                }
            )
        if rows:
            df = pd.DataFrame(rows).sort_values(["exam_date", "required_h_per_day"], ascending=[True, False])
            st.dataframe(df, use_container_width=True, hide_index=True)


def ui_predict(conn: sqlite3.Connection) -> None:
    st.subheader("Predict: Expected Exam Scores, What-if Simulations, and Risk Alerts")

    subjects = fetch_subjects(conn)
    chapters = fetch_chapters(conn)
    exams = fetch_exams(conn)
    av = fetch_availability(conn)

    if subjects.empty:
        st.info("Add subjects first.")
        return

    st.markdown("### Train Prediction Model")
    train_df = build_training_dataset(conn)
    st.caption(
        "Model trains on your historical performance + your logged study hours and completion snapshots. "
        "Needs ~10+ performance records for stable training."
    )
    st.dataframe(train_df.head(15), use_container_width=True, hide_index=True)

    pipe = train_model(train_df)
    if pipe is None:
        st.warning("Not enough data to train yet (need roughly 10+ performance records). Keep logging results and study hours.")
        predicted = None
    else:
        st.success("Model trained.")
        st.markdown("### What-if Simulation")
        mult = st.slider("Study effort multiplier for next ~14 days", 0.5, 2.0, 1.0, 0.05)
        predicted = predict_upcoming_scores(conn, pipe, what_if_multiplier=float(mult))

        st.markdown("### Predicted Scores (Upcoming)")
        st.dataframe(
            predicted[["subject_name", "predicted_score", "hours_14d", "completion", "days_to_exam"]].copy(),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(predicted, x="subject_name", y="predicted_score", hover_data=["hours_14d", "completion", "days_to_exam"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Risk Alerts")
    pred_df = predicted if isinstance(predicted, pd.DataFrame) else None
    alerts = build_risk_alerts(conn, subjects, chapters, exams, av, pred_df)
    if not alerts:
        st.success("No major risks detected based on your current plan and data.")
    else:
        for a in alerts:
            st.warning(a)


def ui_reports(conn: sqlite3.Connection) -> None:
    st.subheader("Reports: Downloadable PDF Study Report")

    subjects = fetch_subjects(conn)
    chapters = fetch_chapters(conn)
    exams = fetch_exams(conn)
    av = fetch_availability(conn)

    # Train model if possible
    train_df = build_training_dataset(conn)
    pipe = train_model(train_df)
    predictions = predict_upcoming_scores(conn, pipe, what_if_multiplier=1.0) if pipe else pd.DataFrame()

    # Progress summary
    logs_30d = fetch_logs(conn, today_local() - timedelta(days=30), today_local())
    hours_30d = logs_30d.groupby("subject_id")["actual_hours"].sum().to_dict() if not logs_30d.empty else {}
    progress_rows = []
    for _, s in subjects.iterrows():
        s_id = int(s["id"])
        comp = subject_completion_now(chapters, s_id) if not chapters.empty else 0.0
        progress_rows.append(
            {
                "subject_id": s_id,
                "subject_name": str(s["name"]),
                "completion_pct": comp * 100.0,
                "hours_30d": float(hours_30d.get(s_id, 0.0)),
            }
        )
    progress = pd.DataFrame(progress_rows).sort_values("completion_pct", ascending=False)

    # Next 7 days schedule
    start = today_local()
    end = today_local() + timedelta(days=7)
    schedule_7d = fetch_schedule(conn, start, end)

    # Alerts
    alerts = build_risk_alerts(conn, subjects, chapters, exams, av, predictions if not predictions.empty else None)

    st.markdown("### Preview")
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.markdown("**Next 7 days schedule**")
        st.dataframe(
            schedule_7d[["item_date", "time_block", "subject_name", "chapter_name", "planned_hours", "status", "source"]]
            if not schedule_7d.empty
            else pd.DataFrame(),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown("**Risk alerts**")
        if not alerts:
            st.success("No risk alerts.")
        else:
            for a in alerts:
                st.warning(a)

        st.markdown("**Predictions**")
        if predictions is None or predictions.empty:
            st.info("Not enough data for predictions yet.")
        else:
            st.dataframe(predictions[["subject_name", "predicted_score", "completion", "hours_14d"]], use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Download PDF")
    if st.button("Generate PDF Report", type="primary"):
        pdf = generate_pdf_report(
            title="Study Planner Report",
            generated_on=datetime.now(),
            schedule_df=schedule_7d,
            progress_df=progress,
            alerts=alerts,
            predictions_df=predictions if not predictions.empty else None,
        )
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf,
            file_name=f"study_report_{iso(today_local())}.pdf",
            mime="application/pdf",
        )


def ui_settings(conn: sqlite3.Connection) -> None:
    st.subheader("Settings: Planner Weights & Advanced Controls")

    st.markdown("### Planning Engine Parameters")
    c1, c2, c3 = st.columns(3)

    urgency_alpha = safe_float(get_setting(conn, "urgency_alpha", "1.4"), 1.4)
    wd = safe_float(get_setting(conn, "weight_difficulty", "0.35"), 0.35)
    ww = safe_float(get_setting(conn, "weight_weakness", "0.35"), 0.35)
    wc = safe_float(get_setting(conn, "weight_completion_gap", "0.30"), 0.30)

    with c1:
        urgency_alpha_new = st.slider("Urgency Alpha (exam proximity aggressiveness)", 0.5, 3.0, float(urgency_alpha), 0.05)
    with c2:
        wd_new = st.slider("Weight: Difficulty", 0.0, 1.0, float(wd), 0.01)
        ww_new = st.slider("Weight: Weakness (low marks)", 0.0, 1.0, float(ww), 0.01)
        wc_new = st.slider("Weight: Completion Gap", 0.0, 1.0, float(wc), 0.01)
    with c3:
        min_chunk = safe_float(get_setting(conn, "min_chunk_hours", "0.5"), 0.5)
        min_chunk_new = st.slider("Min time chunk (hours)", 0.25, 2.0, float(min_chunk), 0.25)

        overwrite_default = (get_setting(conn, "auto_overwrite_future_auto", "true").lower() == "true")
        overwrite_default_new = st.checkbox("Default overwrite future auto items", value=overwrite_default)

        buffer = safe_float(get_setting(conn, "risk_buffer_factor", "0.90"), 0.90)
        buffer_new = st.slider("Risk buffer (lower = stricter alerts)", 0.6, 1.0, float(buffer), 0.01)

    # Normalize weights to sum 1 (keeps behavior stable)
    wsum = wd_new + ww_new + wc_new
    if wsum <= 1e-9:
        st.error("At least one weight must be > 0.")
        return
    wd_new /= wsum
    ww_new /= wsum
    wc_new /= wsum

    if st.button("Save Settings", type="primary"):
        set_setting(conn, "urgency_alpha", f"{urgency_alpha_new:.4f}")
        set_setting(conn, "weight_difficulty", f"{wd_new:.4f}")
        set_setting(conn, "weight_weakness", f"{ww_new:.4f}")
        set_setting(conn, "weight_completion_gap", f"{wc_new:.4f}")
        set_setting(conn, "min_chunk_hours", f"{min_chunk_new:.2f}")
        set_setting(conn, "auto_overwrite_future_auto", "true" if overwrite_default_new else "false")
        set_setting(conn, "risk_buffer_factor", f"{buffer_new:.2f}")
        st.success("Saved.")
        st.rerun()

    st.divider()
    st.markdown("### Advanced: Manual Schedule Item (Power User)")
    subjects = fetch_subjects(conn)
    chapters = fetch_chapters(conn)
    if subjects.empty:
        st.info("Add subjects first.")
        return

    with st.form("manual_item", clear_on_submit=True):
        item_date = st.date_input("Date", value=today_local())
        s_id = select_subject(subjects, "Subject", key="manual_subj")
        chapters_for_subject = chapters[chapters["subject_id"] == s_id] if s_id is not None else pd.DataFrame()
        chap_options = [None] + (chapters_for_subject["id"].tolist() if not chapters_for_subject.empty else [])
        ch_id = st.selectbox(
            "Chapter (optional)",
            chap_options,
            format_func=lambda x: "None" if x is None else chapters_for_subject[chapters_for_subject["id"] == x]["name"].iloc[0],
        )
        planned = st.number_input("Planned Hours", min_value=0.25, max_value=24.0, value=1.0, step=0.25)
        time_block = st.selectbox("Time Block", TIME_BLOCKS, index=TIME_BLOCKS.index("Any"))
        status = st.selectbox("Status", SCHEDULE_STATUSES, index=0)
        notes = st.text_input("Notes", value="Manual item")
        submitted = st.form_submit_button("Add Manual Schedule Item")
    if submitted:
        if s_id is None:
            st.error("Select a subject.")
        else:
            insert_schedule_item(
                conn=conn,
                item_date=item_date,
                subject_id=int(s_id),
                chapter_id=int(ch_id) if ch_id is not None else None,
                planned_hours=float(planned),
                time_block=time_block,
                status=status,
                notes=notes,
                source="manual",
            )
            conn.commit()
            st.success("Added.")
            st.rerun()


if __name__ == "__main__":
    main()
