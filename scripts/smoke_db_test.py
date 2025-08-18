"""
Runs a quick Neon/SQLite API contract test.
Usage:
  python -m scripts.smoke_db_test
"""
import os, time, uuid
from data.store import init_db, save_snapshot, load_snapshot

def main():
    init_db()
    user_id = f"smoke-{uuid.uuid4()}"
    scene = "Probe scene " + time.strftime("%Y-%m-%d %H:%M:%S")
    choices = ["Go left", "Go right"]
    history = [{"role":"assistant","content":scene}]
    save_snapshot(user_id, scene, choices, history)
    snap = load_snapshot(user_id)
    backend = "Neon" if os.getenv("DATABASE_URL") else "SQLite"
    ok = bool(snap and snap.get("scene") == scene and snap.get("choices") == choices)
    print(f"[{backend}] save/load:", "OK" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
